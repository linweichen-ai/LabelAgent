#!/usr/bin/env python3
"""Benchmark cold validation versus reuse of verified pallet review bytes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.agent.pallet_candidate_review import ENV_NAME, load_review_page


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")
    if not 10 <= args.iterations <= 10000:
        raise ValueError("iterations must be between 10 and 10000")
    os.environ[ENV_NAME] = str(args.bundle.resolve())
    started = time.perf_counter_ns()
    first = load_review_page()
    cold_ms = (time.perf_counter_ns() - started) / 1_000_000
    warm_ms = []
    for _ in range(args.iterations):
        started = time.perf_counter_ns()
        page = load_review_page()
        warm_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        if (page["candidate_count"] != first["candidate_count"]
                or page["assets"].keys() != first["assets"].keys()
                or any(page["assets"][key] != first["assets"][key]
                       for key in first["assets"])):
            raise RuntimeError("cached readback changed verified content")
    median = statistics.median(warm_ms)
    report = {
        "schema_version": "labelagent.pallet_candidate_review_cache_benchmark.v1",
        "status": "local_read_only_benchmark_not_deployed",
        "iterations": args.iterations,
        "candidate_count": first["candidate_count"],
        "asset_count": len(first["assets"]),
        "cold_validation_ms": round(cold_ms, 6),
        "warm_median_ms": round(median, 6),
        "warm_p95_ms": round(sorted(warm_ms)[int(len(warm_ms) * .95) - 1], 6),
        "warm_max_ms": round(max(warm_ms), 6),
        "cold_to_warm_median_ratio": round(cold_ms / median, 3) if median else None,
        "verified_content_stable": True,
        "human_truth_writes": 0,
        "training_data_writes": 0,
        "production_deployed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
