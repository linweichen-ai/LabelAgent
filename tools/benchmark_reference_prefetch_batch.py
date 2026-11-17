"""Measure prefetch and click-to-candidate latency on an isolated workbench."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path


def request_json(url, body=None):
    data = None if body is None else json.dumps(
        body, ensure_ascii=False, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data is not None else {})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1200) as response:
        payload = json.load(response)
    return payload, (time.perf_counter() - started) * 1000


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def percentile(values, percent):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path for the complete row-level benchmark receipt.",
    )
    args = parser.parse_args()
    base = args.url.rstrip("/")
    initial, _ = request_json(base + "/api/state/summary")
    initial_state_sha256 = sha256_file(args.state_file)
    initial_truth = set(initial.get("truth_frames") or [])
    initial_candidates = set((initial.get("candidate_counts") or {}).keys())
    requested = [int(frame) for frame in args.frames]
    occupied = [frame for frame in requested
                if str(frame) in initial_truth or str(frame) in initial_candidates]
    if occupied:
        raise SystemExit(f"benchmark frames are not blank: {occupied}")

    rows = []
    for position, frame in enumerate(requested, 1):
        before = sha256_file(args.state_file)
        prefetch, prefetch_ms = request_json(
            base + "/api/reference_discovery/prefetch", {
                "frame": frame,
                "expected_path": initial["path"],
                "expected_topic": initial["topic"],
            })
        after_prefetch = sha256_file(args.state_file)
        if before != after_prefetch:
            raise RuntimeError(f"prefetch wrote isolated state for frame {frame}")
        draft, draft_ms = request_json(base + "/api/reference_draft", {
            "frame": frame,
            "auto_detect_visible": True,
            "defer_agent_review": True,
            "defer_class_audit": True,
            "max_rounds": 6,
        })
        current, _ = request_json(base + "/api/state/summary")
        if str(frame) in set(current.get("truth_frames") or []):
            raise RuntimeError(f"frame {frame} was promoted to truth")
        if str(frame) not in set((current.get("candidate_counts") or {}).keys()):
            raise RuntimeError(f"frame {frame} candidates were not isolated")
        sam = prefetch.get("sam_prefetch") or {}
        roi = sam.get("slope_roi_prefetch") or {}
        cache = (((draft.get("agent") or {}).get("sam_generation") or {})
                 .get("diagnostics", {}).get("response_cache", {}))
        row = {
            "frame": frame,
            "prefetch_wall_ms": round(prefetch_ms, 3),
            "discovery_elapsed_ms": prefetch.get("discovery_elapsed_ms"),
            "sam_prefetch_elapsed_ms": sam.get("elapsed_ms"),
            "slope_roi_prefetch_elapsed_ms": roi.get("elapsed_ms"),
            "slope_roi_prefetch_completed": roi.get("completed", 0),
            "draft_wall_ms": round(draft_ms, 3),
            "candidate_count": len(draft.get("candidates") or []),
            "selected_labels": sam.get("selected_labels") or [],
            "draft_sam_cache_hit": cache.get("hit"),
            "prefetch_state_unchanged": before == after_prefetch,
            "candidate_only": draft.get("candidate_only"),
            "human_truth": draft.get("human_truth"),
            "approved_for_training": draft.get("approved_for_training"),
        }
        rows.append(row)
        print(json.dumps({"progress": f"{position}/{len(requested)}", **row},
                         ensure_ascii=False, separators=(",", ":")), flush=True)

    draft_values = [row["draft_wall_ms"] for row in rows]
    prefetch_values = [row["prefetch_wall_ms"] for row in rows]
    discovery_values = [row["discovery_elapsed_ms"] for row in rows
                        if isinstance(row["discovery_elapsed_ms"], (int, float))]
    result = {
        "schema_version": "labelagent.reference-prefetch-batch-benchmark.v1",
        "frames": requested,
        "rows": rows,
        "summary": {
            "frame_count": len(rows),
            "prefetch_p50_ms": round(statistics.median(prefetch_values), 3),
            "prefetch_p95_ms": round(percentile(prefetch_values, 95), 3),
            "discovery_p50_ms": round(statistics.median(discovery_values), 3),
            "discovery_p95_ms": round(percentile(discovery_values, 95), 3),
            "draft_p50_ms": round(statistics.median(draft_values), 3),
            "draft_p95_ms": round(percentile(draft_values, 95), 3),
            "draft_min_ms": round(min(draft_values), 3),
            "draft_max_ms": round(max(draft_values), 3),
            "all_prefetch_state_unchanged": all(
                row["prefetch_state_unchanged"] for row in rows),
            "all_draft_multi_cache_hits": all(
                row["draft_sam_cache_hit"] is True for row in rows),
            "all_candidate_only": all(row["candidate_only"] is True for row in rows),
            "all_human_truth_false": all(row["human_truth"] is False for row in rows),
            "all_training_approval_false": all(
                row["approved_for_training"] is False for row in rows),
        },
        "initial_state_sha256": initial_state_sha256,
        "final_isolated_state_sha256": sha256_file(args.state_file),
        "note": "The final state hash includes isolated machine candidates from this benchmark."
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print("RESULT=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
