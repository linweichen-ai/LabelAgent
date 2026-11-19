"""Run an isolated, candidate-only LabelAgent smoke test on new images.

The tool reuses model connection settings from an existing LabelAgent state,
but creates all uploads, candidates, and run state in a temporary workspace.
It never changes the source state, human truth, or training data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import time
import urllib.request
import uuid
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
    parser.add_argument(
        "--capture-vlm-cache-stats", action="store_true",
        help="Capture bounded Local VLM cache counters before and after the isolated flow.")
    parser.add_argument(
        "--defer-background", action="store_true",
        help=("Publish the initial machine candidates before Agent refinement and "
              "high-risk class audit, matching the interactive workbench flow."))
    parser.add_argument(
        "--complete-background", action="store_true",
        help=("After publishing deferred candidates, reproducibly finish Agent "
              "refinement and high-risk class audit and record both latencies."))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def vlm_health_url(llm_url: str) -> str:
    parsed = urlsplit(llm_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-len("/v1")]
    return parsed._replace(path=path + "/health", query="", fragment="").geturl()


def bounded_vlm_health(llm_url: str) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(vlm_health_url(llm_url), timeout=10) as response:
        payload = json.load(response)
    stats = payload.get("response_cache_stats")
    if not isinstance(stats, dict) or not all(
            isinstance(key, str) and type(value) is int and value >= 0
            for key, value in stats.items()):
        raise RuntimeError("Local VLM health lacks bounded cache counters")
    return {
        "status": payload.get("status"),
        "model_loaded": payload.get("model_loaded"),
        "response_cache_capacity": payload.get("response_cache_capacity"),
        "response_cache_entries": payload.get("response_cache_entries"),
        "exact_requests_inflight": payload.get("exact_requests_inflight"),
        "response_cache_stats": dict(sorted(stats.items())),
    }


def cache_stat_delta(before: dict, after: dict) -> dict:
    before_stats = before.get("response_cache_stats") or {}
    after_stats = after.get("response_cache_stats") or {}
    names = set(before_stats) | set(after_stats)
    delta = {name: after_stats.get(name, 0) - before_stats.get(name, 0)
             for name in sorted(names)}
    if any(type(value) is not int or value < 0 for value in delta.values()):
        raise RuntimeError("Local VLM cache counters moved backwards")
    return delta


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


def read_model_trace(path: Path, *, stage: str) -> list[dict]:
    """Read one bounded server trace and label the pipeline stage."""
    payload = (json.loads(path.read_text(encoding="utf-8"))
               if path.is_file() else {})
    rows = summarize_model_trace(payload.get("trace"))
    for row in rows:
        row["pipeline_stage"] = stage
    return rows


def grouped_audit_timing(response: dict, elapsed: float) -> dict:
    """Represent a blind-audit endpoint without inventing per-call traces."""
    audits = list(((response.get("agent") or {}).get(
        "candidate_class_audits") or []))
    return {
        "sequence": 1,
        "pipeline_stage": "background_candidate_class_audit",
        "kind": "candidate_class_blind_audit_group",
        "service": "GPT",
        "status": "succeeded",
        "elapsed_ms": elapsed,
        "request_count": len(audits),
        "execution_modes": sorted({
            str(item.get("execution_mode")) for item in audits
            if isinstance(item, dict) and item.get("execution_mode")
        }),
        "content_free_summary": True,
    }


def main() -> int:
    args = parse_args()
    if args.complete_background and not args.defer_background:
        raise ValueError("--complete-background requires --defer-background")
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
    vlm_health_before = (bounded_vlm_health(config["llm_url"])
                         if args.capture_vlm_cache_stats else None)

    with tempfile.TemporaryDirectory(prefix="labelagent-new-data-") as temporary:
        isolated_work = Path(temporary).resolve()
        os.environ["LABELAGENT_WORK_DIR"] = str(isolated_work)
        # Import only after setting the isolated work directory.
        from agent import server, workflow

        client = server.app.test_client()
        bootstrap = checked_json(client.get("/api/bootstrap"), "read label binding")
        checked_json(client.post("/api/labels", json={
            "labels": labels,
            "path": bootstrap.get("path", ""),
            "topic": bootstrap.get("topic"),
            "file_activation_generation": bootstrap.get(
                "file_activation_generation"),
            "expected_label_state_sha256": bootstrap.get("label_state_sha256"),
        }), "set labels")
        bootstrap = checked_json(client.get("/api/bootstrap"), "read config binding")
        config_payload = {
            key: value for key, value in config.items()
            if key in {"llm_url", "llm_model", "llm_key", "llm_prompt_profile", "sam3_url",
                       "sam3_point_url", "ego_bottom_point_fallback",
                       "llm_fast_url", "llm_fast_model", "llm_fast_key",
                       "llm_fast_max_pixels", "llm_fast_max_long_edge",
                       "llm_fast_max_extra_pixels", "llm_fast_max_extra_long_edge",
                       "llm_fast_timeout_seconds", "llm_fast_cooldown_seconds"}
        }
        config_payload.update({
            "expected_path": bootstrap.get("path", ""),
            "expected_topic": bootstrap.get("topic"),
            "expected_file_activation_generation": bootstrap.get(
                "file_activation_generation"),
            "expected_config_state_sha256": bootstrap.get("config_state_sha256"),
        })
        checked_json(client.post("/api/config", json=config_payload), "set model config")

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
            draft_request = {
                "frame": 0,
                "auto_detect_visible": True,
                "max_rounds": args.max_rounds,
            }
            if args.defer_background:
                draft_request.update({
                    "defer_agent_review": True,
                    "defer_class_audit": True,
                })
            response = checked_json(client.post(
                "/api/reference_draft", json=draft_request),
                f"candidate inference {image.name}")
            candidate_visible_ms = elapsed_ms(infer_started)
            debug_path = isolated_work / "model_debug.json"
            call_timings = read_model_trace(
                debug_path, stage="initial_candidate_publication")
            background = None
            if args.complete_background:
                background_started = time.perf_counter()
                topic = server.state.get("topic")
                if not isinstance(topic, str):
                    raise RuntimeError("isolated server lacks an active topic")
                artifact_sha = response.get("candidate_artifact_sha256")
                input_sha = response.get("input_sha256")
                if not isinstance(artifact_sha, str) or not isinstance(input_sha, str):
                    raise RuntimeError("deferred candidate response lacks binding hashes")
                refine_started = time.perf_counter()
                refined = checked_json(client.post(
                    "/api/reference_candidates/refine", json={
                        "frame": 0,
                        "topic": topic,
                        "refinement_id": uuid.uuid4().hex,
                        "expected_candidate_artifact_sha256": artifact_sha,
                        "expected_frame_sha256": input_sha,
                    }), f"background Agent refinement {image.name}")
                refine_ms = elapsed_ms(refine_started)
                call_timings.extend(read_model_trace(
                    debug_path, stage="background_agent_refinement"))
                if refined.get("stopped") is True:
                    raise RuntimeError("background Agent refinement stopped")
                final_response = copy.deepcopy(response)
                final_response["candidates"] = refined.get("candidates")
                final_response["agent"] = refined.get("agent")
                final_response["candidate_artifact_sha256"] = refined.get(
                    "candidate_artifact_sha256")
                final_response["agent_review_deferred"] = False
                final_response["agent_review_status"] = "complete"
                audit_ms = 0.0
                audit_cache_hit = None
                if ((refined.get("agent") or {}).get(
                        "candidate_class_audit_status") == "pending"):
                    audit_started = time.perf_counter()
                    audited = checked_json(client.post(
                        "/api/reference_candidates/audit", json={
                            "frame": 0,
                            "topic": topic,
                            "expected_candidate_artifact_sha256": refined.get(
                                "candidate_artifact_sha256"),
                            "expected_frame_sha256": refined.get("input_sha256"),
                        }), f"background candidate audit {image.name}")
                    audit_ms = elapsed_ms(audit_started)
                    audit_cache_hit = audited.get(
                        "candidate_class_audit_cache_hit")
                    call_timings.append(grouped_audit_timing(audited, audit_ms))
                    final_response["candidates"] = audited.get("candidates")
                    final_response["agent"] = audited.get("agent")
                    final_response["candidate_artifact_sha256"] = audited.get(
                        "candidate_artifact_sha256")
                    final_response["candidate_class_audit_deferred"] = False
                    final_response["candidate_class_audit_status"] = "complete"
                response = final_response
                background = {
                    "completed": True,
                    "agent_refinement_elapsed_ms": refine_ms,
                    "candidate_class_audit_elapsed_ms": audit_ms,
                    "candidate_class_audit_cache_hit": audit_cache_hit,
                    "elapsed_ms": elapsed_ms(background_started),
                    "candidate_only": response.get("candidate_only") is True,
                    "human_truth": response.get("human_truth") is True,
                    "approved_for_training": response.get(
                        "approved_for_training") is True,
                }
            full_pipeline_ms = elapsed_ms(infer_started)
            candidates = response.get("candidates") or []
            discovery = response.get("visible_label_discovery") or {}
            debug = (json.loads(debug_path.read_text(encoding="utf-8"))
                     if debug_path.is_file() else {})
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
                "inference_elapsed_ms": full_pipeline_ms,
                "candidate_visible_elapsed_ms": candidate_visible_ms,
                "background_completion": background,
                "background_deferred": args.defer_background,
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
                "ego_bottom_point_fallback": copy.deepcopy(
                    (response.get("agent") or {}).get("ego_bottom_point_fallback")),
                "candidate_only": response.get("candidate_only") is True,
                "human_truth": response.get("human_truth") is True,
                "approved_for_training": response.get("approved_for_training") is True,
                "result_json": str(result_path),
                "result_sha256": sha256(result_path),
                "overlay": str(overlay_path),
                "overlay_sha256": sha256(overlay_path),
            })

        vlm_cache_observability = None
        if vlm_health_before is not None:
            vlm_health_after = bounded_vlm_health(config["llm_url"])
            vlm_cache_observability = {
                "content_free": True,
                "health_url": vlm_health_url(config["llm_url"]),
                "before": vlm_health_before,
                "after": vlm_health_after,
                "delta": cache_stat_delta(vlm_health_before, vlm_health_after),
            }
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
            "background_deferred": args.defer_background,
            "background_completed": args.complete_background,
            "batch_upload_elapsed_ms": upload_ms,
            "vlm_cache_observability": vlm_cache_observability,
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
        "candidate_visible_elapsed_ms": [
            run["candidate_visible_elapsed_ms"] for run in runs],
        "isolated_workspace_removed": True,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
