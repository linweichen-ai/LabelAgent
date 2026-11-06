"""CPU synthetic protocol tests; no SAM weights, GPU calls, or network."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

_here = Path(__file__).resolve()
_candidates = [_here.with_name("sam_raw_capture.py")]
for _root in _here.parents:
    _candidates.append(_root / "tools/labelagent_experiments/sam_raw_capture.py")
_helper = next(path for path in _candidates if path.is_file())
_spec = importlib.util.spec_from_file_location("isolated_raw_capture_test_helper", _helper)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
capture_segment, load_private_segment, save_capture = _module.capture_segment, _module.load_private_segment, _module.save_capture


SYNTHETIC_SOURCE = '''
import numpy as np
class Model:
    def __init__(self):
        self.masks=np.zeros((2,24,32),dtype=np.uint8)
        self.masks[0,2:9,2:9]=1
        self.masks[0,13:19,19:28]=1
        self.masks[0,21,30]=1
        self.masks[1,3:22,4:25]=1
        self.scores=np.array([.4,.9],dtype=np.float32)
        self.returned=(self.masks,self.scores,None)
        self.calls=0
    def predict_inst(self,*args,**kwargs):
        self.calls+=1
        return self.returned
model=Model()
fail_after_prediction=False
response={"synthetic_only":True,"parts":[],"score":.9}
def _predictor():return model,None
def segment(image_bytes,points,prior_mask=None):
    predictor,processor=_predictor()
    returned=predictor.predict_inst({"image_bytes":image_bytes},
        point_coords=np.array([[p["x"],p["y"]] for p in points],dtype=np.float32),
        point_labels=np.array([p["label"] for p in points],dtype=np.int32),
        multimask_output=True)
    assert returned is model.returned
    best=1
    if fail_after_prediction:raise ValueError("synthetic postprocessing failure")
    return response
'''


class RawCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sam-raw-capture-test-")
        folder = Path(self.temporary.name).resolve()
        self.assertEqual(folder.parent, Path(tempfile.gettempdir()).resolve())
        self.file = folder / "synthetic_segment.py"
        self.file.write_text(SYNTHETIC_SOURCE, encoding="utf-8")
        self.digest = hashlib.sha256(self.file.read_bytes()).hexdigest()
        self.module = load_private_segment(self.file, self.digest)
        self.points = [{"x": 4, "y": 4, "label": 1}]

    def tearDown(self):
        self.temporary.cleanup()

    def test_all_raw_candidates_components_and_small_pixels_are_copied(self):
        result = capture_segment(self.module, b"synthetic bytes, not a real image", self.points)
        self.assertEqual(result["status"], "segment_returned")
        self.assertIs(result["response"], self.module.response)
        self.assertEqual(result["predict_inst_calls"], 1)
        self.assertEqual(result["selected_index"], 1)
        captured = result["predictions"][0]["masks"]["array"]
        np.testing.assert_array_equal(captured, self.module.model.masks)
        self.assertEqual(captured.shape, (2, 24, 32))
        self.assertEqual(int(captured[0].sum()), 49 + 54 + 1)
        self.assertEqual(int(captured[0, 21, 30]), 1)
        captured[:] = 0
        self.assertEqual(int(self.module.model.masks[0].sum()), 104)

    def test_existing_function_receives_original_return_and_factory_is_restored(self):
        factory = self.module._predictor
        original = self.module.model.masks.copy()
        for _ in range(2):
            result = capture_segment(self.module, b"synthetic", self.points)
            self.assertTrue(result["predictor_factory_restored"])
            self.assertIs(self.module._predictor, factory)
            self.assertEqual(result["predict_inst_calls"], 1)
            np.testing.assert_array_equal(self.module.model.masks, original)
        self.assertEqual(self.module.model.calls, 2)

    def test_postprocessing_error_retains_raw_capture_but_is_not_success(self):
        self.module.fail_after_prediction = True
        factory = self.module._predictor
        result = capture_segment(self.module, b"synthetic", self.points)
        self.assertEqual(result["status"], "segment_raised")
        self.assertEqual(result["error"]["type"], "ValueError")
        self.assertIsNone(result["response"])
        self.assertEqual(result["predict_inst_calls"], 1)
        self.assertIs(self.module._predictor, factory)
        self.assertEqual(result["selected_index"], 1)
        np.testing.assert_array_equal(result["predictions"][0]["masks"]["array"], self.module.model.masks)
        folder = Path(self.temporary.name) / "error-capture"
        manifest = save_capture(result, folder)
        self.assertEqual(manifest["status"], "segment_raised")
        with np.load(folder / "prediction_000.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["masks"], self.module.model.masks)

    def test_source_digest_mismatch_and_nonprivate_modules_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_private_segment(self.file, "0" * 64)
        with self.assertRaisesRegex(ValueError, "Only a module"):
            capture_segment(types.ModuleType("source.sam3_service.point_segment"), b"synthetic", self.points)

    def test_hash_verified_bytes_execute_even_if_disk_changes_with_same_size_and_mtime(self):
        original = self.file.read_bytes()
        replacement = original.replace(b"synthetic_only", b"different_only")
        self.assertEqual(len(original), len(replacement))
        stamp = self.file.stat()
        actual_sha256 = hashlib.sha256
        def hash_then_change_disk(data):
            digest = actual_sha256(data)
            self.file.write_bytes(replacement)
            os.utime(self.file, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            return digest
        with mock.patch.object(_module.hashlib, "sha256", side_effect=hash_then_change_disk):
            private = load_private_segment(self.file, self.digest)
        self.assertTrue(private.response["synthetic_only"])
        self.assertNotIn("different_only", private.response)
        self.assertEqual(self.file.read_bytes(), replacement)
        self.assertEqual(private._capture_source_sha256, self.digest)

    def test_point_and_image_provenance_is_exact_without_mutating_prompt(self):
        result = capture_segment(self.module, b"synthetic", self.points, prior_mask=b"synthetic prior")
        self.assertEqual(result["image_sha256"], hashlib.sha256(b"synthetic").hexdigest())
        self.assertEqual(result["prior_mask_sha256"], hashlib.sha256(b"synthetic prior").hexdigest())
        self.assertEqual(result["source_sha256"], self.digest)
        self.assertEqual(result["points"], self.points)
        result["points"][0]["x"] = 999
        self.assertEqual(self.points[0]["x"], 4)
        np.testing.assert_array_equal(result["predictions"][0]["point_coords"]["array"], [[4, 4]])

    def test_float_masks_are_not_silently_binarized_and_selection_is_traced(self):
        self.module.model.masks = self.module.model.masks.astype(np.float32) * .25
        self.module.model.returned = (self.module.model.masks, self.module.model.scores, None)
        result = capture_segment(self.module, b"synthetic", self.points)
        folder = Path(self.temporary.name) / "float-capture"
        manifest = save_capture(result, folder)
        self.assertEqual(manifest["selected_index"], 1)
        self.assertEqual(manifest["selected_prediction_call"], 0)
        with np.load(folder / "prediction_000.npz", allow_pickle=False) as saved:
            self.assertEqual(saved["masks"].dtype, np.dtype("float32"))
            np.testing.assert_array_equal(saved["masks"], self.module.model.masks)
            self.assertEqual(float(saved["masks"][0, 21, 30]), .25)

    def test_copy_failure_is_explicit_and_received_native_return_is_preserved(self):
        original = _module._array_record
        counter = 0
        def fail_score_copy(value):
            nonlocal counter
            counter += 1
            if counter == 2:
                raise RuntimeError("synthetic capture-copy failure")
            return original(value)
        with mock.patch.object(_module, "_array_record", side_effect=fail_score_copy):
            result = capture_segment(self.module, b"synthetic", self.points)
        self.assertEqual(result["status"], "segment_raised")
        self.assertFalse(result["selected_index_observed"])
        folder = Path(self.temporary.name) / "failed-copy"
        manifest = save_capture(result, folder)
        self.assertFalse(manifest["raw_capture_complete"])
        fallback = folder / manifest["predictions"][0]["failed_capture_native_return_file"]
        import pickle
        with fallback.open("rb") as stream:
            restored = pickle.load(stream)
        np.testing.assert_array_equal(restored[0], self.module.model.masks)
        np.testing.assert_array_equal(restored[1], self.module.model.scores)

    def test_cli_defaults_to_prepare_and_does_not_import_a_model_source(self):
        folder = Path(self.temporary.name)
        source = folder / "must_not_import.py"
        source.write_text("raise RuntimeError('This source must not be imported in prepare mode')\n", encoding="utf-8")
        (folder / "input.png").write_bytes(b"synthetic input bytes")
        (folder / "points.json").write_text(json.dumps(self.points), encoding="utf-8")
        run = subprocess.run([sys.executable, "-B", "-X", "utf8", str(_helper),
            "--source", str(source), "--source-sha256", hashlib.sha256(source.read_bytes()).hexdigest(),
            "--input-image", str(folder / "input.png"), "--points-json", str(folder / "points.json"),
            "--output-dir", str(folder / "prepared")], capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual(result["status"], "prepared_only")
        self.assertEqual(result["model_calls"], 0)
        self.assertFalse((folder / "prepared/capture.json").exists())

    def test_execute_cli_requires_success_raw_capture_and_observed_selection(self):
        folder = Path(self.temporary.name)
        (folder / "input.png").write_bytes(b"synthetic input bytes")
        (folder / "points.json").write_text(json.dumps(self.points), encoding="utf-8")
        cases = [
            ("postprocessing-error", SYNTHETIC_SOURCE.replace("fail_after_prediction=False", "fail_after_prediction=True"), 1),
            ("missing-selection", SYNTHETIC_SOURCE.replace("best=1", "selection_not_exposed=1"), 1),
            ("complete-evidence", SYNTHETIC_SOURCE, 0),
        ]
        for name, source, expected in cases:
            with self.subTest(name=name):
                file = folder / (name + ".py")
                file.write_text(source, encoding="utf-8")
                destination = folder / name
                run = subprocess.run([sys.executable, "-B", "-X", "utf8", str(_helper),
                    "--source", str(file), "--source-sha256", hashlib.sha256(file.read_bytes()).hexdigest(),
                    "--input-image", str(folder / "input.png"), "--points-json", str(folder / "points.json"),
                    "--output-dir", str(destination), "--execute-isolated-inference"],
                    capture_output=True, text=True, encoding="utf-8", timeout=30)
                self.assertEqual(run.returncode, expected, run.stdout + run.stderr)
                record = json.loads((destination / "capture.json").read_text(encoding="utf-8"))
                self.assertEqual(record["predict_inst_calls"], 1)
                self.assertTrue(record["raw_capture_complete"])
                with np.load(destination / "prediction_000.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["masks"], self.module.model.masks)


if __name__ == "__main__":
    unittest.main()
