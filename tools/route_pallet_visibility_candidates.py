#!/usr/bin/env python3
"""Route pallet candidates from single-frame and temporal LiDAR evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_pallet_box3d_roi_sam import digest


def temporal_key(report: dict) -> tuple[str, int, str, str]:
    return (str(report["sequence"]), int(report["camera"]),
            str(report["center_frame_id"]), str(report["target_obj_id"]))


def propagation_sources(report: dict) -> list[str]:
    return [str(row["frame_id"]) for row in report.get("rows", [])
            if row.get("evidence_eligible")
            and int(row.get("target_visible_points", 0)) >= 3]


def route_instance(instance: dict, temporal: dict | None) -> dict:
    classification = instance["lidar_visibility_classification"]
    base = {
        "preserve_3d_identity": True,
        "write_human_truth": False,
        "write_training_data": False,
        "production_deploy": False,
        "requires_review": True,
        "propagation_source_frame_ids": [],
    }
    if classification == "visible_lidar_supported":
        return {**base, "route": "depth_split_candidate",
                "allow_current_2d_candidate": True,
                "suppress_current_2d_candidate": False,
                "reason": "stable_single_frame_lidar_visibility"}
    if classification == "occluded_lidar_supported":
        return {**base, "route": "single_frame_occlusion_defer_2d",
                "allow_current_2d_candidate": False,
                "suppress_current_2d_candidate": True,
                "reason": "single_frame_lidar_supports_occlusion"}
    if (classification == "no_lidar_return_insufficient_evidence" and temporal
            and temporal.get("temporal_occlusion_supported_for_center") is True):
        return {**base, "route": "temporal_occlusion_defer_2d",
                "allow_current_2d_candidate": False,
                "suppress_current_2d_candidate": True,
                "propagation_source_frame_ids": propagation_sources(temporal),
                "reason": "adjacent_valid_frames_support_occlusion_and_reappearance"}
    return {**base, "route": "insufficient_evidence_review",
            "allow_current_2d_candidate": False,
            "suppress_current_2d_candidate": True,
            "reason": "no_safe_2d_candidate_route"}


def split_metric_index(report: dict) -> dict[tuple[str, int, str, str], dict]:
    if (report.get("schema_version") != "labelagent.pallet_box3d_overlap_pairs.v1"
            or report.get("production_deployed") is not False):
        raise ValueError("requires a frozen non-deployed split report")
    result = {}
    for image in report.get("rows", []):
        for metric in image.get("depth_split_metrics", []):
            key = (str(image["sequence"]), int(image["camera"]),
                   str(image["frame_id"]), str(metric["obj_id"]))
            if key in result:
                raise ValueError(f"duplicate split metric for {key}")
            result[key] = metric
    return result


def apply_split_gate(decision: dict, metric: dict | None) -> dict:
    """Suppress a visible instance when its actual split mask is not usable."""
    if not decision["allow_current_2d_candidate"]:
        return decision
    if metric is None:
        raise ValueError("allowed candidate has no matching split metric")
    if metric.get("candidate_pass") is True:
        return decision
    return {
        **decision,
        "route": "split_candidate_failed_review",
        "allow_current_2d_candidate": False,
        "suppress_current_2d_candidate": True,
        "reason": "lidar_visible_but_depth_split_mask_failed_gate",
    }


def validate_single(report: dict) -> None:
    if (report.get("schema_version") != "labelagent.pallet_lidar_visibility.v1"
            or report.get("production_deployed") is not False
            or report.get("human_pixel_truth") is not False):
        raise ValueError("requires a frozen non-deployed single-frame report")


def validate_temporal(report: dict) -> None:
    if (report.get("schema_version") != "labelagent.pallet_temporal_visibility.v1"
            or report.get("production_deployed") is not False
            or report.get("human_pixel_truth") is not False):
        raise ValueError("requires a frozen non-deployed temporal report")


def route(single_path: Path, temporal_paths: list[Path], output: Path,
          split_path: Path | None = None) -> dict:
    if output.exists():
        raise ValueError("output already exists")
    single = json.loads(single_path.read_text(encoding="utf-8")); validate_single(single)
    temporal_reports = []
    for path in temporal_paths:
        report = json.loads(path.read_text(encoding="utf-8")); validate_temporal(report)
        temporal_reports.append((path, report))
    temporal_map = {}
    for path, report in temporal_reports:
        key = temporal_key(report)
        if key in temporal_map:
            raise ValueError(f"duplicate temporal evidence for {key}")
        temporal_map[key] = {"path": path, "report": report}
    split_metrics = None
    if split_path is not None:
        split_report = json.loads(split_path.read_text(encoding="utf-8"))
        split_metrics = split_metric_index(split_report)
    rows = []
    for image in single["rows"]:
        sequence, camera, frame = (str(image["sequence"]), int(image["camera"]),
                                   str(image["frame_id"]))
        for instance in image["instances"]:
            key = (sequence, camera, frame, str(instance["obj_id"]))
            temporal_item = temporal_map.get(key)
            decision = route_instance(instance,
                                      temporal_item["report"] if temporal_item else None)
            if split_metrics is not None:
                decision = apply_split_gate(decision, split_metrics.get(key))
            rows.append({
                "sequence": sequence, "camera": camera, "frame_id": frame,
                "obj_id": instance["obj_id"],
                "single_frame_classification": instance[
                    "lidar_visibility_classification"],
                "single_frame_box_points": instance["box_points"],
                "single_frame_visible_points": instance["default_visible_points"],
                "temporal_report_path": (str(temporal_item["path"].resolve())
                                         if temporal_item else None),
                "temporal_report_sha256": (digest(temporal_item["path"])
                                           if temporal_item else None),
                **decision,
            })
    route_counts = {}
    for row in rows:
        route_counts[row["route"]] = route_counts.get(row["route"], 0) + 1
    report = {
        "schema_version": "labelagent.pallet_visibility_candidate_routes.v1",
        "status": "offline_candidate_routes_not_deployed",
        "single_frame_report_sha256": digest(single_path),
        "split_report_sha256": digest(split_path) if split_path else None,
        "temporal_report_count": len(temporal_reports),
        "candidate_count": len(rows), "route_counts": route_counts,
        "current_2d_candidates_allowed": sum(
            row["allow_current_2d_candidate"] for row in rows),
        "current_2d_candidates_suppressed": sum(
            row["suppress_current_2d_candidate"] for row in rows),
        "review_required": sum(row["requires_review"] for row in rows),
        "human_truth_writes": 0, "training_data_writes": 0,
        "workbench_writes": 0, "production_deployed": False,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-frame-report", required=True, type=Path)
    parser.add_argument("--temporal-report", action="append", default=[], type=Path)
    parser.add_argument("--split-report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = route(args.single_frame_report, args.temporal_report, args.output,
                   args.split_report)
    print(json.dumps({key: report[key] for key in (
        "candidate_count", "route_counts", "current_2d_candidates_allowed",
        "current_2d_candidates_suppressed", "review_required",
        "human_truth_writes", "training_data_writes", "workbench_writes")},
        ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
