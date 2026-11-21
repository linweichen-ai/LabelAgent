#!/usr/bin/env python3
"""Merge non-overlapping read-only pallet boundary audit segments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def merge(inputs: list[Path]) -> dict:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    if not reports:
        raise ValueError("at least one input report is required")
    identity_keys = (
        "workbench", "path", "topic", "file_activation_generation",
        "total_frame_count",
    )
    baseline = {key: reports[0][key] for key in identity_keys}
    if any({key: report[key] for key in identity_keys} != baseline
           for report in reports[1:]):
        raise ValueError("audit segments do not share the same live data scope")
    inventory = [row for report in reports for row in report["frame_inventory"]]
    frames = [int(row["frame"]) for row in inventory]
    if len(frames) != len(set(frames)):
        raise ValueError("audit segments contain overlapping frames")
    expected = list(range(baseline["total_frame_count"]))
    if sorted(frames) != expected:
        missing = sorted(set(expected) - set(frames))
        raise ValueError(f"audit segments do not cover every frame; missing={missing}")
    results = [row for report in reports for row in report["results"]]
    violations = [row for row in results if not row["gate_passed"]]
    adjusted = [row for row in results if row["changed_pixels"] > 0]
    return {
        "schema_version": "labelagent.pallet_boundary_notch_collection_audit.v1",
        **baseline,
        "frame_range": [0, baseline["total_frame_count"]],
        "frame_count": baseline["total_frame_count"],
        "candidate_count": sum(row["candidate_count"] for row in inventory),
        "pallet_candidate_count": len(results),
        "adjusted_count": len(adjusted),
        "v6_adjusted_count": sum(
            row["geometry_cleanup_version"] == 6 for row in adjusted),
        "filled_image_boundary_notch_count": sum(
            row["filled_image_boundary_notch_count"] for row in results),
        "component_merge_count": sum(
            row["component_merge_count"] for row in results),
        "maximum_change_ratio": max(
            (row["change_ratio"] for row in results), default=0),
        "minimum_structural_hole_retention": min(
            (row["minimum_structural_hole_retention"] for row in results),
            default=1),
        "maximum_component_merge_count": max(
            (row["component_merge_count"] for row in results), default=0),
        "violation_count": len(violations),
        "gate_passed": not violations,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "top_changed": sorted(
            adjusted, key=lambda row: row["change_ratio"], reverse=True)[:30],
        "violations": violations,
        "frame_inventory": sorted(inventory, key=lambda row: int(row["frame"])),
        "results": sorted(
            results, key=lambda row: (int(row["frame"]), str(row["shape_id"]))),
        "segments": [
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "frame_range": report["frame_range"],
            }
            for path, report in zip(inputs, reports)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = merge(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    summary = {key: report[key] for key in (
        "frame_count", "candidate_count", "pallet_candidate_count",
        "adjusted_count", "v6_adjusted_count",
        "filled_image_boundary_notch_count", "component_merge_count",
        "maximum_change_ratio", "minimum_structural_hole_retention",
        "maximum_component_merge_count", "violation_count", "gate_passed",
    )}
    summary["output_sha256"] = hashlib.sha256(
        args.output.read_bytes()).hexdigest()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
