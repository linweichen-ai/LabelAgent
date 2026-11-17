"""Measure production visible-label discovery latency without writing workbench state.

The benchmark reuses the exact LabelAgent discovery prompt, schema, registry,
model route and token budget from a workbench state file.  It only reads image
bytes and writes one receipt under ``artifacts``; candidates, human truth and
training data are never touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.agent import workflow
from source.agent.media import open_media


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def read_health(url: str | None) -> dict | None:
    if not url:
        return None
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("VLM health did not return an object")
    return {
        "status": payload.get("status"),
        "model_loaded": payload.get("model_loaded"),
        "exact_requests_inflight": payload.get("exact_requests_inflight"),
        "response_cache_entries": payload.get("response_cache_entries"),
        "response_cache_stats": payload.get("response_cache_stats"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--health-url")
    args = parser.parse_args()

    output = args.output.resolve()
    artifact_root = (ROOT / "artifacts").resolve()
    if not output.is_relative_to(artifact_root):
        raise ValueError("output must be inside repository artifacts")
    if output.exists():
        raise FileExistsError(output)

    state_path = args.state.resolve()
    images = [path.resolve() for path in args.images]
    for path in [state_path, *images]:
        if not path.is_file():
            raise FileNotFoundError(path)

    state_sha256_before = file_sha256(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    config = dict(state.get("config") or {})
    labels = list(state.get("labels") or [])
    if not config.get("llm_url") or not config.get("llm_model") or not labels:
        raise ValueError("state lacks primary model settings or labels")

    health_before = read_health(args.health_url)
    rows = []
    for position, path in enumerate(images, 1):
        source_bytes = path.read_bytes()
        media = open_media(path)
        topics = list(media.topics)
        if len(topics) != 1 or media.topics[topics[0]].get("count") != 1:
            raise ValueError(f"benchmark input is not one still image: {path}")
        # Match the browser/server route exactly: StillImage applies EXIF
        # transpose, RGB conversion and deterministic PNG encoding before the
        # workflow receives bytes.  Using source file bytes directly can have
        # identical pixels but a different exact request and cache key.
        media_topic = topics[0]
        image = media.frame(media_topic, 0)
        trace_by_id = {}

        def capture(event):
            trace_by_id[event["id"]] = event

        started = time.perf_counter()
        with workflow.trace_calls(capture):
            discovery = workflow.discover_visible_labels(config, image, labels)
        wall_ms = (time.perf_counter() - started) * 1000
        calls = [event for event in trace_by_id.values()
                 if event.get("service") == "GPT"]
        if not calls or calls[-1].get("status") != "succeeded":
            raise RuntimeError(f"missing successful VLM trace for {path}")
        call = calls[-1]
        model_output = call.get("output") or {}
        local = model_output.get("local_vlm") or {}
        stats = local.get("inference_stats") or {}
        row = {
            "image": str(path),
            "source_file_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "source_encoded_bytes": len(source_bytes),
            "production_image_sha256": hashlib.sha256(image).hexdigest(),
            "production_encoded_bytes": len(image),
            "production_media_topic": media_topic,
            "wall_ms": rounded(wall_ms),
            "trace_elapsed_ms": call.get("elapsed_ms"),
            "model_route": model_output.get("model_route"),
            "finish_reason": model_output.get("finish_reason"),
            "cache_hit": local.get("cache_hit"),
            "coalesced": local.get("coalesced"),
            "coalesced_wait_seconds": local.get("coalesced_wait_seconds"),
            "prompt_tokens": stats.get("prompt_tokens"),
            "completion_tokens": stats.get("completion_tokens"),
            "preprocess_seconds": stats.get("preprocess_seconds"),
            "constraint_setup_seconds": stats.get("constraint_setup_seconds"),
            "generation_seconds": stats.get("generation_seconds"),
            "actual_image_token_counts": stats.get("actual_image_token_counts"),
            "image_grid_thw": stats.get("image_grid_thw"),
            "visible_labels": discovery.get("visible_labels") or [],
            "uncertain_labels": discovery.get("uncertain_labels") or [],
            "unknown_candidate_count": len(
                discovery.get("unknown_candidates") or []),
            "open_set_scan_complete": discovery.get("open_set_scan_complete"),
            "discovery_sha256": json_sha256(discovery),
        }
        rows.append(row)
        print(json.dumps(
            {"progress": f"{position}/{len(images)}", **row},
            ensure_ascii=False, separators=(",", ":")), flush=True)

    health_after = read_health(args.health_url)
    wall_values = [float(row["wall_ms"]) for row in rows]
    generation_values = [float(row["generation_seconds"]) for row in rows
                         if isinstance(row["generation_seconds"], (int, float))]
    median_ms = statistics.median(wall_values)
    result = {
        "schema_version": "labelagent.visible-discovery-latency.v1",
        "read_only": True,
        "production_contract": {
            "model": config.get("llm_model"),
            "prompt_profile": config.get("llm_prompt_profile", "full"),
            "configured_max_tokens": config.get("llm_max_tokens"),
            "labels": labels,
            "fast_route_configured": bool(
                config.get("llm_fast_url") and config.get("llm_fast_model")),
        },
        "rows": rows,
        "summary": {
            "image_count": len(rows),
            "wall_min_ms": rounded(min(wall_values)),
            "wall_p50_ms": rounded(median_ms),
            "wall_p95_ms": rounded(percentile(wall_values, 95)),
            "wall_max_ms": rounded(max(wall_values)),
            "generation_p50_seconds": rounded(
                statistics.median(generation_values)
                if generation_values else None),
            "generation_p95_seconds": rounded(
                percentile(generation_values, 95)),
            "over_25_seconds": sum(value > 25000 for value in wall_values),
            "over_twice_median": sum(
                value > 2 * median_ms for value in wall_values),
            "cache_hits": sum(row["cache_hit"] is True for row in rows),
            "coalesced_requests": sum(
                row["coalesced"] is True for row in rows),
            "all_primary_route": all(
                row["model_route"] == "primary" for row in rows),
            "all_stop_finish": all(
                row["finish_reason"] == "stop" for row in rows),
            "candidate_written": False,
            "human_truth_modified": False,
            "training_data_modified": False,
            "source_state_modified": file_sha256(state_path) != state_sha256_before,
        },
        "health_before": health_before,
        "health_after": health_after,
        "state": str(state_path),
        "state_sha256": state_sha256_before,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("RESULT=" + json.dumps(result["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
