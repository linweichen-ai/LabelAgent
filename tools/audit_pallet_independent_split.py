#!/usr/bin/env python3
"""Audit a frozen pallet split report against previously tuned frame IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def audit(report: dict, excluded_frames: set[str]) -> dict:
    if report.get("schema_version") != "labelagent.pallet_box3d_overlap_pairs.v1":
        raise ValueError("unsupported pallet split report schema")
    if report.get("production_deployed") is not False:
        raise ValueError("independent audit requires an offline report")
    if report.get("human_pixel_truth") is not False:
        raise ValueError("this audit is for candidate-only reports")

    rows = report.get("rows")
    if not isinstance(rows, list):
        raise ValueError("report rows must be a list")

    overlap = []
    row_summaries = []
    for row in rows:
        frame_id = str(row.get("frame_id", ""))
        if frame_id in excluded_frames:
            overlap.append(frame_id)
        split_metrics = row.get("depth_split_metrics") or []
        failed = [str(item.get("obj_id")) for item in split_metrics
                  if not item.get("candidate_pass")]
        occluded = [str(item.get("obj_id")) for item in split_metrics
                    if item.get("occluded_unobservable_candidate")]
        row_summaries.append({
            "sequence": str(row.get("sequence", "")),
            "frame_id": frame_id,
            "camera": int(row.get("camera", -1)),
            "all_instances_localized": bool(row.get("all_instances_localized")),
            "raw_conflicts_separated": bool(row.get("all_conflicts_separated")),
            "depth_split_candidate_pass": bool(
                row.get("depth_split_candidate_pass")),
            "depth_or_occlusion_handled_candidate": bool(
                row.get("depth_or_occlusion_handled_candidate")),
            "instance_count": len(split_metrics),
            "failed_obj_ids": failed,
            "occluded_unobservable_obj_ids": occluded,
        })

    unique_frames = sorted({item["frame_id"] for item in row_summaries})
    all_localized = bool(row_summaries) and all(
        item["all_instances_localized"] for item in row_summaries)
    all_handled = bool(row_summaries) and all(
        item["depth_or_occlusion_handled_candidate"] for item in row_summaries)
    split_pass_count = sum(
        item["depth_split_candidate_pass"] for item in row_summaries)
    independent_gate_pass = bool(
        row_summaries
        and not overlap
        and all_localized
        and all_handled
        and report.get("workbench_writes") == 0
        and report.get("approved_for_training") is False
    )
    return {
        "schema_version": "labelagent.pallet_independent_split_audit.v1",
        "status": "independent_candidate_audit_not_pixel_truth",
        "excluded_tuning_frame_count": len(excluded_frames),
        "evaluated_sequence_count": len({item["sequence"] for item in row_summaries}),
        "evaluated_frame_count": len(unique_frames),
        "evaluated_row_count": len(row_summaries),
        "overlapping_tuning_frames": sorted(set(overlap)),
        "all_instances_localized": all_localized,
        "depth_split_pass_row_count": split_pass_count,
        "depth_or_occlusion_handled_row_count": sum(
            item["depth_or_occlusion_handled_candidate"]
            for item in row_summaries),
        "all_rows_safely_handled": all_handled,
        "independent_candidate_gate_pass": independent_gate_pass,
        "human_pixel_accuracy_measurable": False,
        "false_positive_rate_measurable": False,
        "approved_for_training": False,
        "production_deployed": False,
        "row_summaries": row_summaries,
        "limitations": [
            "3D boxes localize instances but are not human pixel masks",
            "occluded-unobservable candidates stay suppressed for human review",
            "passing this audit does not authorize training or deployment",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that a split report is frame-independent and conservative.")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--excluded-frame", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise ValueError("output file already exists")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = audit(report, {str(value) for value in args.excluded_frame})
    result["input_report"] = str(args.report.resolve())
    result["input_report_sha256"] = digest(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "independent_candidate_gate_pass": result[
            "independent_candidate_gate_pass"],
        "evaluated_frame_count": result["evaluated_frame_count"],
        "depth_split_pass_row_count": result["depth_split_pass_row_count"],
        "depth_or_occlusion_handled_row_count": result[
            "depth_or_occlusion_handled_row_count"],
        "output": str(args.output.resolve()),
        "output_sha256": digest(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
