"""Run an isolated, candidate-only LabelAgent smoke test on new images.

The tool reuses model connection settings from an existing LabelAgent state,
but creates all uploads, candidates, and run state in a temporary workspace.
It never changes the source state, human truth, or training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Candidate-only smoke test for a new LabelAgent image batch.")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--source-state", required=True, type=Path,
                        help="Existing state.json used only for model settings and labels.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-rounds", type=int, default=2)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def checked_json(response, operation: str) -> dict:
    payload = response.get_json(silent=True)
    if response.status_code >= 400:
        detail = payload if payload is not None else response.get_data(as_text=True)
        raise RuntimeError(f"{operation} failed ({response.status_code}): {detail}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"{operation} returned a non-object response")
    return payload


def main() -> int:
    args = parse_args()
    images = [path.expanduser().resolve() for path in args.images]
    source_state_path = args.source_state.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not 1 <= args.max_rounds <= 100:
        raise ValueError("--max-rounds must be between 1 and 100")
    if not source_state_path.is_file():
        raise FileNotFoundError(source_state_path)
    for image in images:
        if not image.is_file():
            raise FileNotFoundError(image)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_state = json.loads(source_state_path.read_text(encoding="utf-8"))
    config = dict(source_state.get("config") or {})
    labels = list(source_state.get("labels") or [])
    required_config = ("llm_url", "llm_model", "sam3_url")
    missing = [key for key in required_config if not config.get(key)]
    if missing or not labels:
        raise ValueError(f"source state lacks required settings: {missing or ['labels']}")

    with tempfile.TemporaryDirectory(prefix="labelagent-new-data-") as temporary:
        isolated_work = Path(temporary).resolve()
        os.environ["LABELAGENT_WORK_DIR"] = str(isolated_work)
        # Import only after setting the isolated work directory.
        from agent import server, workflow

        client = server.app.test_client()
        checked_json(client.post("/api/labels", json={"labels": labels}), "set labels")
        checked_json(client.post("/api/config", json={
            key: value for key, value in config.items()
            if key in {"llm_url", "llm_model", "llm_key", "llm_prompt_profile", "sam3_url"}
        }), "set model config")

        started = time.perf_counter()
        with ExitStack() as stack:
            files = [(stack.enter_context(path.open("rb")), path.name) for path in images]
            upload = checked_json(client.post(
                "/api/uploads", data={"files": files},
                content_type="multipart/form-data"), "batch upload")
        upload_ms = elapsed_ms(started)
        added = upload.get("added") or []
        if len(added) != len(images):
            raise RuntimeError(f"batch upload registered {len(added)} of {len(images)} files")

        runs = []
        for image, item in zip(images, added):
            load_started = time.perf_counter()
            loaded = checked_json(client.post(f"/api/files/{item['id']}/load"),
                                  f"load {image.name}")
            load_ms = elapsed_ms(load_started)
            compatibility = loaded.get("data_compatibility") or {}

            infer_started = time.perf_counter()
            response = checked_json(client.post("/api/reference_draft", json={
                "frame": 0,
                "auto_detect_visible": True,
                "max_rounds": args.max_rounds,
            }), f"candidate inference {image.name}")
            inference_ms = elapsed_ms(infer_started)
            candidates = response.get("candidates") or []
            discovery = response.get("visible_label_discovery") or {}
            counts = dict(sorted(Counter(
                shape.get("label", "<missing>") for shape in candidates
                if isinstance(shape, dict)).items()))

            stem = image.stem
            result_path = output_dir / f"{stem}-candidate.json"
            result_path.write_text(json.dumps(response, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
            overlay_path = output_dir / f"{stem}-overlay.jpg"
            overlay_path.write_bytes(workflow.candidate_overlay(image.read_bytes(), candidates))
            runs.append({
                "input": str(image),
                "input_sha256": sha256(image),
                "bytes": image.stat().st_size,
                "load_elapsed_ms": load_ms,
                "load_timings_ms": loaded.get("load_timings_ms"),
                "compatibility": compatibility,
                "inference_elapsed_ms": inference_ms,
                "candidate_count": len(candidates),
                "candidate_counts_by_label": counts,
                "visible_labels": discovery.get("visible_labels", []),
                "uncertain_labels": discovery.get("uncertain_labels", []),
                "unknown_candidates": discovery.get("unknown_candidates", []),
                "open_set_scan_complete": discovery.get("open_set_scan_complete"),
                "review_queue_count": len((response.get("agent") or {}).get("review_queue") or []),
                "candidate_only": response.get("candidate_only") is True,
                "human_truth": response.get("human_truth") is True,
                "approved_for_training": response.get("approved_for_training") is True,
                "result_json": str(result_path),
                "result_sha256": sha256(result_path),
                "overlay": str(overlay_path),
                "overlay_sha256": sha256(overlay_path),
            })

        summary = {
            "schema_version": "labelagent.new_dataset_smoke.v1",
            "isolated_workspace": True,
            "source_state_unchanged_by_design": True,
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "training_data_modified": False,
            "image_count": len(images),
            "labels": labels,
            "model": config.get("llm_model"),
            "prompt_profile": config.get("llm_prompt_profile"),
            "max_rounds": args.max_rounds,
            "batch_upload_elapsed_ms": upload_ms,
            "runs": runs,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "image_count": len(runs),
        "candidate_counts": [run["candidate_count"] for run in runs],
        "inference_elapsed_ms": [run["inference_elapsed_ms"] for run in runs],
        "isolated_workspace_removed": True,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
