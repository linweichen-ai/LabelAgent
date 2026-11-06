"""Opt-in capture of raw SAM results inside a private diagnostic module.

Importing this helper never loads weights, starts workers, or makes requests.
Calling capture_segment may run the supplied module's predictor: real GPU use
must be coordinated outside this helper. It must not be used to patch a live
service module. Captured arrays are copies; the original segment function gets
the exact predictor return object and performs its existing processing.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import uuid
import weakref

import numpy as np


_PRIVATE_MODULES = weakref.WeakSet()
_ACTIVE_MODULES = weakref.WeakSet()


def load_private_segment(source_path, expected_sha256):
    """Load an exact on-disk source copy without importing a service namespace."""
    path = Path(source_path).resolve()
    source_bytes = path.read_bytes()
    actual = hashlib.sha256(source_bytes).hexdigest()
    if actual != expected_sha256:
        raise ValueError("Segment source SHA256 does not match the frozen source")
    spec = importlib.util.spec_from_file_location("_sam_raw_capture_" + uuid.uuid4().hex, path)
    if spec is None or spec.loader is None:
        raise ValueError("Segment source cannot be loaded as a private module")
    module = importlib.util.module_from_spec(spec)
    # Execute the same verified byte buffer, not a second read or cached pyc.
    exec(compile(source_bytes, str(path), "exec"), module.__dict__)
    if not callable(getattr(module, "segment", None)) or not callable(getattr(module, "_predictor", None)):
        raise ValueError("Private module lacks the expected segment/predictor interface")
    module._capture_source_path = str(path)
    module._capture_source_sha256 = actual
    _PRIVATE_MODULES.add(module)
    return module


def _array_record(value):
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    array = np.array(value, copy=True)
    return {"array": array, "shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


class _PredictorTap:
    def __init__(self, model, records):
        self._model = model
        self._records = records

    def __getattr__(self, name):
        return getattr(self._model, name)

    def predict_inst(self, *args, **kwargs):
        returned = self._model.predict_inst(*args, **kwargs)
        # The current source consumes NumPy-compatible masks/scores. Keep all
        # mask candidates, including every disconnected component and small
        # pixel region, before any source contour conversion or filtering.
        record = {"raw_return": returned, "multimask_output": kwargs.get("multimask_output"),
                  "mask_input_present": "mask_input" in kwargs,
                  "all_returned_candidates_retained": False}
        self._records.append(record)
        try:
            if not isinstance(returned, (tuple, list)) or len(returned) < 2:
                raise ValueError("Unexpected predictor return protocol; raw capture failed")
            record.update(masks=_array_record(returned[0]), scores=_array_record(returned[1]),
                          point_coords=_array_record(kwargs["point_coords"]) if "point_coords" in kwargs else None,
                          point_labels=_array_record(kwargs["point_labels"]) if "point_labels" in kwargs else None,
                          all_returned_candidates_retained=True)
        except Exception as error:
            record["capture_error"] = {"type": type(error).__name__, "message": str(error)[:1000]}
            raise
        return returned


def capture_segment(private_module, image_bytes, points, prior_mask=None):
    """Run the unmodified private segment function and retain pre-filter arrays.

    On segment errors, raw predictions already produced remain in the returned
    diagnostic record. An error result is never a valid segmentation. No output
    arrays or polygons are intersected, simplified, or selected by this helper.
    """
    if private_module not in _PRIVATE_MODULES:
        raise ValueError("Only a module loaded by load_private_segment may be captured")
    if private_module in _ACTIVE_MODULES:
        raise RuntimeError("A capture is already active on this private module")
    original_factory = private_module._predictor
    calls = []
    record = {"source_path": private_module._capture_source_path,
              "source_sha256": private_module._capture_source_sha256,
              "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
              "points": copy.deepcopy(points),
              "prior_mask_sha256": hashlib.sha256(prior_mask).hexdigest() if prior_mask is not None else None,
              "predictions": calls, "response": None, "error": None,
              "live_service_patched": False, "array_capture_before_source_postprocessing": True}
    previous_profile = sys.getprofile()
    segment_code = private_module.segment.__code__

    def observe_selection(frame, event, arg):
        if frame.f_code is segment_code and event == "return":
            best = frame.f_locals.get("best")
            if isinstance(best, (int, np.integer)) and not isinstance(best, bool):
                record["selected_index"] = int(best)
                record["selected_prediction_call"] = len(calls) - 1
                record["selected_index_source"] = "unmodified segment function local variable best at return/exception"
        if previous_profile is not None:
            previous_profile(frame, event, arg)

    def tapped_factory():
        model, processor = original_factory()
        return _PredictorTap(model, calls), processor

    _ACTIVE_MODULES.add(private_module)
    private_module._predictor = tapped_factory
    try:
        sys.setprofile(observe_selection)
        record["response"] = private_module.segment(image_bytes, points, prior_mask=prior_mask)
        record["status"] = "segment_returned"
    except Exception as error:
        record["status"] = "segment_raised"
        record["error"] = {"type": type(error).__name__, "message": str(error)[:1000]}
    finally:
        sys.setprofile(previous_profile)
        private_module._predictor = original_factory
        _ACTIVE_MODULES.discard(private_module)
    record["predict_inst_calls"] = len(calls)
    record["predictor_factory_restored"] = private_module._predictor is original_factory
    record["profile_hook_restored"] = sys.getprofile() is previous_profile
    record["selected_index_observed"] = "selected_index" in record
    return record


def save_capture(record, output_dir):
    """Persist arrays without dtype coercion, including failed postprocessing."""
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "capture.json").exists():
        raise FileExistsError("An existing capture must remain immutable")
    manifest = {key: value for key, value in record.items() if key != "predictions"}
    manifest["predictions"] = []
    for index, prediction in enumerate(record["predictions"]):
        info = {key: value for key, value in prediction.items()
                if key not in {"raw_return", "masks", "scores", "point_coords", "point_labels"}}
        arrays = {}
        for key in ("masks", "scores", "point_coords", "point_labels"):
            value = prediction.get(key)
            if value is not None:
                arrays[key] = value["array"]
                info[key] = {name: field for name, field in value.items() if name != "array"}
        file = destination / f"prediction_{index:03d}.npz"
        np.savez_compressed(file, **arrays)
        info.update(array_file=file.name, array_file_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        if "capture_error" in prediction:
            # A native fallback keeps the received return object on copy errors;
            # it never converts floating masks into binary uint8. This is an
            # explicitly failed capture, not eligible segmentation evidence.
            import pickle
            fallback = destination / f"prediction_{index:03d}_native_return.pickle"
            try:
                with fallback.open("wb") as stream:
                    pickle.dump(prediction["raw_return"], stream, protocol=pickle.HIGHEST_PROTOCOL)
                info["failed_capture_native_return_file"] = fallback.name
                info["failed_capture_native_return_sha256"] = hashlib.sha256(fallback.read_bytes()).hexdigest()
            except Exception as error:
                info["native_return_persistence_error"] = {"type": type(error).__name__, "message": str(error)[:1000]}
        manifest["predictions"].append(info)
    manifest["raw_capture_complete"] = bool(record["predictions"]) and all(
        item.get("all_returned_candidates_retained", False) for item in record["predictions"])
    (destination / "capture.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--points-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execute-isolated-inference", action="store_true",
                        help="Explicitly execute the private predictor; omitted means prepare only")
    args = parser.parse_args(argv)
    source = Path(args.source).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != args.source_sha256:
        raise ValueError("Source SHA256 mismatch")
    image_bytes = Path(args.input_image).read_bytes()
    points = json.loads(Path(args.points_json).read_text(encoding="utf-8-sig"))
    if not isinstance(points, list):
        raise ValueError("points-json must contain a list of source-image pixel points")
    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    plan = {"status": "prepared_only", "source": str(source), "source_sha256": digest,
            "input_image_sha256": hashlib.sha256(image_bytes).hexdigest(), "points": points,
            "model_calls": 0, "live_service_patched": False,
            "execution_requested": args.execute_isolated_inference}
    (destination / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.execute_isolated_inference:
        print(json.dumps(plan, ensure_ascii=False))
        return 0
    private = load_private_segment(source, digest)
    result = capture_segment(private, image_bytes, points)
    manifest = save_capture(result, destination)
    print(json.dumps({"status": manifest["status"], "predict_inst_calls": manifest["predict_inst_calls"],
                      "raw_capture_complete": manifest["raw_capture_complete"]}, ensure_ascii=False))
    return 0 if (manifest["status"] == "segment_returned" and manifest["raw_capture_complete"]
                 and manifest["selected_index_observed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
