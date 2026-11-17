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
from urllib.parse import urlsplit


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
    parser.add_argument("--llm-url",
                        help="Explicit VLM endpoint override; source state is never changed.")
    parser.add_argument("--llm-model",
                        help="Explicit VLM model-id override; source state is never changed.")
    parser.add_argument("--llm-fast-url",
                        help="Optional small-image VLM endpoint; source state is never changed.")
    parser.add_argument("--llm-fast-model",
                        help="Optional small-image VLM model-id; source state is never changed.")
    parser.add_argument(
        "--verify-audit-reuse", action="store_true",
        help="After inference, repeat the exact persisted class audit and require a server cache hit.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def model_config(source: dict, llm_url: str | None = None,
                 llm_model: str | None = None, llm_fast_url: str | None = None,
                 llm_fast_model: str | None = None) -> dict:
    config = dict(source or {})
    if llm_url:
        parsed = urlsplit(llm_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None):
            raise ValueError("llm-url must be an HTTP(S) endpoint without credentials")
        config["llm_url"] = llm_url.rstrip("/")
    if llm_model:
        if not llm_model.strip():
            raise ValueError("llm-model must not be blank")
        config["llm_model"] = llm_model.strip()
    if llm_fast_url:
        parsed = urlsplit(llm_fast_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None):
            raise ValueError("llm-fast-url must be an HTTP(S) endpoint without credentials")
        config["llm_fast_url"] = llm_fast_url.rstrip("/")
    if llm_fast_model:
        if not llm_fast_model.strip():
            raise ValueError("llm-fast-model must not be blank")
        config["llm_fast_model"] = llm_fast_model.strip()
    return config


def checked_json(response, operation: str) -> dict:
    payload = response.get_json(silent=True)
    if response.status_code >= 400:
        detail = payload if payload is not None else response.get_data(as_text=True)
        raise RuntimeError(f"{operation} failed ({response.status_code}): {detail}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"{operation} returned a non-object response")
    return payload


def model_call_kind(record: dict) -> str:
    """Name a model call from its bounded prompt without changing the pipeline."""
    instruction = str((record.get("input") or {}).get("instruction") or "")
    if instruction.startswith("Inspect this factory image and assess every known label"):
        return "visible_label_discovery"
    if instruction.startswith("Independently classify exactly one highlighted machine candidate"):
        return "candidate_class_blind_audit"
    if (instruction.startswith("No human-confirmed reference image")
            or instruction.startswith("Use the supplied reference contours")
            or "Annotation audit" in instruction or "candidate masks" in instruction):
        return "agent_review"
    if instruction.startswith("Describe only the annotated objects"):
        return "reference_description"
    return "other_vlm_call"


def summarize_model_trace(trace: object) -> list[dict]:
    """Keep timing and protocol evidence while excluding prompts and model prose."""
    if not isinstance(trace, list):
        return []
    summary = []
    for position, record in enumerate(trace, 1):
        if not isinstance(record, dict):
            continue
        output = record.get("output") or {}
        local_vlm = output.get("local_vlm")
        raw_text = output.get("raw_text")
        parsed = output.get("parsed")
        images = (record.get("input") or {}).get("images") or []
        summary.append({
            "sequence": position,
            "kind": model_call_kind(record),
            "service": record.get("service"),
            "status": record.get("status"),
            "elapsed_ms": record.get("elapsed_ms"),
            "busy_retry_count": output.get("busy_retry_count", 0),
            "model_route": output.get("model_route"),
            "fast_fallback_reason": output.get("fast_fallback_reason"),
            "finish_reason": output.get("finish_reason"),
            "raw_text_chars": len(raw_text) if isinstance(raw_text, str) else None,
            "parsed_json_chars": (len(json.dumps(parsed, ensure_ascii=False,
                                                   separators=(",", ":")))
                                  if isinstance(parsed, (dict, list)) else None),
            "image_count": len(images) if isinstance(images, list) else None,
            "max_tokens": (record.get("input") or {}).get("max_tokens"),
            "local_vlm": local_vlm if isinstance(local_vlm, dict) else None,
        })
    return summary


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
    config = model_config(source_state.get("config") or {},
                          args.llm_url, args.llm_model,
                          args.llm_fast_url, args.llm_fast_model)
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
            if key in {"llm_url", "llm_model", "llm_key", "llm_prompt_profile", "sam3_url",
                       "llm_fast_url", "llm_fast_model", "llm_fast_key",
                       "llm_fast_max_pixels", "llm_fast_max_long_edge",
                       "llm_fast_max_extra_pixels", "llm_fast_max_extra_long_edge",
                       "llm_fast_timeout_seconds", "llm_fast_cooldown_seconds"}
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
            debug_path = isolated_work / "model_debug.json"
            debug = (json.loads(debug_path.read_text(encoding="utf-8"))
                     if debug_path.is_file() else {})
            call_timings = summarize_model_trace(debug.get("trace"))
            audit_reuse = None
            if args.verify_audit_reuse:
                expected_artifact = response.get("candidate_artifact_sha256")
                expected_frame = response.get("input_sha256")
                if (not isinstance(expected_artifact, str)
                        or not isinstance(expected_frame, str)):
                    raise RuntimeError("candidate response lacks audit binding hashes")
                audit_started = time.perf_counter()
                repeated = checked_json(client.post(
                    "/api/reference_candidates/audit", json={
                        "frame": 0,
                        "topic": server.state.get("topic"),
                        "expected_candidate_artifact_sha256": expected_artifact,
                        "expected_frame_sha256": expected_frame,
                    }), f"repeat candidate audit {image.name}")
                audit_elapsed = elapsed_ms(audit_started)
                debug_after = (json.loads(debug_path.read_text(encoding="utf-8"))
                               if debug_path.is_file() else {})
                if repeated.get("candidate_class_audit_cache_hit") is not True:
                    raise RuntimeError("exact repeated audit did not report a server cache hit")
                if repeated.get("agent") != response.get("agent"):
                    raise RuntimeError("reused audit differs from the persisted audit")
                if debug_after != debug:
                    raise RuntimeError("exact repeated audit unexpectedly changed model trace")
                if (repeated.get("candidate_only") is not True
                        or repeated.get("human_truth") is not False
                        or repeated.get("approved_for_training") is not False):
                    raise RuntimeError("reused audit crossed the candidate-only boundary")
                audit_reuse = {
                    "verified": True,
                    "elapsed_ms": audit_elapsed,
                    "candidate_class_audit_cache_hit": True,
                    "agent_response_identical": True,
                    "model_trace_unchanged": True,
                    "candidate_only": repeated.get("candidate_only") is True,
                    "human_truth_false": repeated.get("human_truth") is False,
                    "approved_for_training_false": (
                        repeated.get("approved_for_training") is False),
                }
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
                "model_call_count": len(call_timings),
                "model_call_elapsed_ms": round(sum(
                    call.get("elapsed_ms") or 0 for call in call_timings), 3),
                "model_call_timings": call_timings,
                "audit_reuse": audit_reuse,
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
