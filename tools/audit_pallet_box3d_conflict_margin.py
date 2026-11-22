#!/usr/bin/env python3
"""Audit the projected-3D-box overlap margin without running SAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_pallet_box3d_roi_sam import digest, load_camera, project_box


def maximum_projected_overlap(target: dict, projected: list[dict]) -> dict:
    target_area = int(target["mask"].sum())
    best = {"maximum_overlap": 0.0, "other_obj_id": None,
            "target_overlap": 0.0, "other_overlap": 0.0}
    for other in projected:
        if other is target:
            continue
        other_area = int(other["mask"].sum())
        overlap = int((target["mask"] & other["mask"]).sum())
        target_fraction = overlap / target_area if target_area else 0.0
        other_fraction = overlap / other_area if other_area else 0.0
        maximum = max(target_fraction, other_fraction)
        if maximum > best["maximum_overlap"]:
            best = {"maximum_overlap": maximum,
                    "other_obj_id": other["obj_id"],
                    "target_overlap": target_fraction,
                    "other_overlap": other_fraction}
    return best


def threshold_counts(values: list[float], thresholds: list[float]) -> dict[str, int]:
    return {format(threshold, ".2f"): sum(value >= threshold for value in values)
            for threshold in thresholds}


def audit(source_report: Path, manifest_path: Path, labels_root: Path,
          calibration: Path, output: Path, thresholds: list[float]) -> dict:
    if output.exists():
        raise ValueError("output already exists")
    source = json.loads(source_report.read_text(encoding="utf-8"))
    if (source.get("scope") != "3d_box_localization_agreement_not_pixel_boundary_accuracy"
            or source.get("production_deployed") is not False):
        raise ValueError("requires the frozen non-deployed 3D ROI report")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {(str(row["sequence"]), int(str(row["camera"]).split("_")[-1])): row
             for row in manifest["files"]}
    root = Path(manifest["local_root"])
    rows = []
    for source_row in source["rows"]:
        if not source_row.get("localization_pass"):
            continue
        sequence = str(source_row["sequence"]); camera_number = int(source_row["camera"])
        item = files[(sequence, camera_number)]; frame = str(item["frame_id"])
        image_path = root / item["relative_path"]
        if digest(image_path) != item["sha256"]:
            raise ValueError("held-out image changed")
        label = json.loads((labels_root / sequence / f"{frame}.json").read_text(
            encoding="utf-8"))
        camera = load_camera(calibration, camera_number)
        projected = [value for obj in label["objects"]
                     if obj.get("obj_type") == "Pallet" and not obj.get("ignore")
                     if (value := project_box(obj, camera)) is not None]
        target = next(row for row in projected
                      if str(row["obj_id"]) == str(source_row["target_obj_id"]))
        overlap = maximum_projected_overlap(target, projected)
        rows.append({"sequence": sequence, "frame_id": frame,
                     "camera": camera_number, "target_obj_id": target["obj_id"],
                     "source_instance_isolation_pass": bool(
                         source_row.get("instance_isolation_pass")), **overlap})
    isolated = [row["maximum_overlap"] for row in rows
                if row["source_instance_isolation_pass"]]
    failed = [row["maximum_overlap"] for row in rows
              if not row["source_instance_isolation_pass"]]
    max_isolated = max(isolated) if isolated else None
    min_failed = min(failed) if failed else None
    margin = (min_failed - max_isolated
              if min_failed is not None and max_isolated is not None else None)
    values = [row["maximum_overlap"] for row in rows]
    report = {
        "schema_version": "labelagent.pallet_box3d_conflict_margin.v1",
        "status": "offline_geometry_audit_not_deployed",
        "source_report_sha256": digest(source_report),
        "manifest_sha256": digest(manifest_path),
        "localized_row_count": len(rows),
        "source_isolated_row_count": len(isolated),
        "source_failed_isolation_row_count": len(failed),
        "maximum_overlap_among_source_isolated": max_isolated,
        "minimum_overlap_among_source_failed": min_failed,
        "separation_margin": margin,
        "trigger_counts_by_threshold": threshold_counts(values, thresholds),
        "workbench_writes": 0, "human_pixel_truth": False,
        "approved_for_training": False, "production_deployed": False,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--labels-root", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--thresholds", default="0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    args = parser.parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    report = audit(args.source_report, args.manifest, args.labels_root,
                   args.calibration, args.output, thresholds)
    print(json.dumps({key: report[key] for key in (
        "localized_row_count", "maximum_overlap_among_source_isolated",
        "minimum_overlap_among_source_failed", "separation_margin",
        "trigger_counts_by_threshold", "workbench_writes")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
