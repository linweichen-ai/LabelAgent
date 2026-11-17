"""Capture exact, state-safe visible-label discovery receipts across frames."""

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
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--state-file", required=True, type=Path)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()

    base = args.url.rstrip("/")
    initial, _ = request_json(base + "/api/state/summary")
    initial_state_sha256 = sha256_file(args.state_file)
    initial_truth = set(initial.get("truth_frames") or [])
    initial_candidates = set((initial.get("candidate_counts") or {}).keys())
    occupied = [frame for frame in args.frames
                if str(frame) in initial_truth or str(frame) in initial_candidates]
    if occupied:
        raise SystemExit(f"benchmark frames are not blank: {occupied}")

    rows = []
    for position, frame in enumerate(args.frames, 1):
        before = sha256_file(args.state_file)
        response, wall_ms = request_json(
            base + "/api/reference_discovery/prefetch", {
                "frame": frame,
                "expected_path": initial["path"],
                "expected_topic": initial["topic"],
            })
        after = sha256_file(args.state_file)
        if before != after:
            raise RuntimeError(f"prefetch wrote isolated state for frame {frame}")
        if response.get("prefetched") is not True:
            raise RuntimeError(
                f"frame {frame} was not prefetched: {response.get('code')}")
        row = {
            "frame": frame,
            "wall_ms": round(wall_ms, 3),
            "discovery_elapsed_ms": response.get("discovery_elapsed_ms"),
            "input_sha256": response.get("input_sha256"),
            "discovery_sha256": response.get("discovery_sha256"),
            "visible_labels": response.get("visible_labels") or [],
            "uncertain_labels": response.get("uncertain_labels") or [],
            "retry_count": response.get("retry_count"),
            "sam_selected_labels": ((response.get("sam_prefetch") or {})
                                    .get("selected_labels") or []),
            "state_unchanged": before == after,
            "candidate_only": response.get("candidate_only"),
            "human_truth": response.get("human_truth"),
            "approved_for_training": response.get("approved_for_training"),
        }
        rows.append(row)
        print(json.dumps({"progress": f"{position}/{len(args.frames)}", **row},
                         ensure_ascii=False, separators=(",", ":")), flush=True)

    latencies = [row["discovery_elapsed_ms"] for row in rows]
    result = {
        "schema_version": "labelagent.reference-discovery-benchmark.v1",
        "variant": args.variant,
        "frames": args.frames,
        "rows": rows,
        "summary": {
            "frame_count": len(rows),
            "discovery_p50_ms": round(statistics.median(latencies), 3),
            "discovery_p95_ms": round(percentile(latencies, 95), 3),
            "all_state_unchanged": all(row["state_unchanged"] for row in rows),
            "all_candidate_only": all(row["candidate_only"] is True for row in rows),
            "all_human_truth_false": all(row["human_truth"] is False for row in rows),
            "all_training_approval_false": all(
                row["approved_for_training"] is False for row in rows),
        },
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": sha256_file(args.state_file),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("RESULT=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
