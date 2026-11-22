#!/usr/bin/env python3
"""Use adjacent frozen labels and PCDs to test a single-frame occlusion hypothesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.compare_boxes3d import Box
from tools.evaluate_pallet_box3d_roi_sam import digest, load_camera, project_box
from tools.evaluate_projected_support import project_visible
from tools.slope_lidar_geometry import read_xyz_pcd


def temporal_occlusion_gate(center_visible: int, visible_before: int,
                            visible_after: int, present_frames: int,
                            total_frames: int, median_with_overlap: float,
                            median_without_overlap: float) -> bool:
    return bool(center_visible == 0 and visible_before >= 1 and visible_after >= 1
                and present_frames == total_frames
                and median_without_overlap >= median_with_overlap + 5)


def temporal_presence_gate(center_visible: int, visible_before: int,
                           visible_after: int, present_frames: int,
                           total_frames: int, attr_consistent: bool,
                           scale_max_range: float) -> bool:
    """Confirm tracked presence only; never infer a pixel boundary."""
    return bool(center_visible >= 1 and visible_before >= 1 and visible_after >= 1
                and present_frames == total_frames and attr_consistent
                and scale_max_range <= 1e-6)


def temporal_input_paths(manifest_path: Path, manifest: dict, data_root: Path,
                         calibration: Path) -> list[Path]:
    camera_number = int(manifest["camera"])
    paths = [manifest_path, calibration / "lidar2camera.yaml",
             calibration / f"camera{camera_number}_calib.yaml"]
    for item in manifest["frames"]:
        frame = str(item["frame_id"])
        paths.extend((data_root / "labels" / f"{frame}.json",
                      data_root / "pcd" / f"{frame}.pcd"))
    return paths


def freeze_temporal_inputs(manifest_path: Path, manifest: dict, data_root: Path,
                           calibration: Path) -> dict[str, str]:
    paths = temporal_input_paths(manifest_path, manifest, data_root, calibration)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"temporal inputs missing: {missing}")
    return {str(path.resolve()): digest(path) for path in paths}


def verify_temporal_inputs(input_sha256: dict[str, str]) -> None:
    drifted = [path for path, expected in input_sha256.items()
               if not Path(path).is_file() or digest(Path(path)) != expected]
    if drifted:
        raise ValueError(f"temporal inputs changed during evaluation: {drifted}")


def temporal_identity_counts(rows: list[dict]) -> tuple[int, int]:
    return sum(bool(row.get("target_present")) for row in rows), len(rows)


def visible_point_median(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return float(np.median([row["target_visible_points"] for row in rows]))


def projected_overlap(target: dict | None, other: dict | None) -> float:
    if target is None or other is None:
        return 0.0
    area = int(target["mask"].sum())
    return int((target["mask"] & other["mask"]).sum()) / area if area else 0.0


def render_chart(rows: list[dict], center_frame: str, output: Path,
                 target_id: str, occluder_id: str) -> None:
    width, height = 1000, 480
    canvas = np.full((height, width, 3), 247, np.uint8)
    left, right, top, bottom = 70, 950, 55, 390
    cv2.line(canvas, (left, bottom), (right, bottom), (70, 80, 88), 1)
    cv2.line(canvas, (left, top), (left, bottom), (70, 80, 88), 1)
    max_visible = max(20, max(row.get("target_visible_points", 0) for row in rows))
    points_visible, points_overlap = [], []
    for index, row in enumerate(rows):
        x = int(left + index * (right - left) / max(1, len(rows) - 1))
        y = int(bottom - row.get("target_visible_points", 0) / max_visible * (bottom - top))
        oy = int(bottom - row.get("target_occluder_projected_overlap", 0) * (bottom - top))
        if row.get("target_ignored"):
            cv2.drawMarker(canvas, (x, bottom), (135, 135, 135),
                           cv2.MARKER_TILTED_CROSS, 8, 1, cv2.LINE_AA)
        else:
            points_visible.append((x, y))
        points_overlap.append((x, oy))
        if row["frame_id"] == center_frame:
            cv2.line(canvas, (x, top), (x, bottom), (45, 45, 210), 2)
        cv2.putText(canvas, str(index - rows.index(next(r for r in rows if r["frame_id"] == center_frame))),
                    (x - 8, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, .42,
                    (55, 65, 72), 1, cv2.LINE_AA)
        label = ("ignore" if row.get("target_ignored")
                 else str(row.get("target_visible_points", 0)))
        label_y = bottom - 10 if row.get("target_ignored") else y - 9
        label_color = (110, 110, 110) if row.get("target_ignored") else (20, 130, 80)
        cv2.putText(canvas, label, (x - 14, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, label_color, 1, cv2.LINE_AA)
    cv2.polylines(canvas, [np.array(points_visible)], False, (20, 150, 90), 2,
                  cv2.LINE_AA)
    cv2.polylines(canvas, [np.array(points_overlap)], False, (225, 145, 25), 2,
                  cv2.LINE_AA)
    cv2.putText(canvas, f"obj{target_id} visible LiDAR points", (75, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 130, 80), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"projected overlap with nearer obj{occluder_id}", (390, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (225, 145, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, "relative annotated frame (0 = center)", (345, 445),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (55, 65, 72), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), canvas)


def evaluate(manifest_path: Path, data_root: Path, calibration: Path,
             output: Path) -> dict:
    if output.exists():
        raise ValueError("output directory already exists")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sequence = str(manifest["sequence"]); camera_number = int(manifest["camera"])
    target_id = str(manifest["target_obj_id"]); occluder_id = str(manifest["occluder_obj_id"])
    center_frame = str(manifest["center_frame_id"])
    input_sha256 = freeze_temporal_inputs(
        manifest_path, manifest, data_root, calibration)
    camera = load_camera(calibration, camera_number)
    rows, attrs, scales = [], [], []
    for item in manifest["frames"]:
        frame = str(item["frame_id"]); label_path = data_root / "labels" / f"{frame}.json"
        pcd_path = data_root / "pcd" / f"{frame}.pcd"
        if (label_path.stat().st_size != int(item["label_size"])
                or pcd_path.stat().st_size != int(item["pcd_size"])
                or digest(label_path) != item["label_sha256"]
                or digest(pcd_path) != item["pcd_sha256"]):
            raise ValueError("temporal input differs from frozen manifest")
        label = json.loads(label_path.read_text(encoding="utf-8"))
        objects = {str(obj.get("obj_id")): obj for obj in label["objects"]
                   if obj.get("obj_type") == "Pallet"}
        target = objects.get(target_id); occluder = objects.get(occluder_id)
        if target is None:
            rows.append({"frame_id": frame, "target_present": False}); continue
        attrs.append(target.get("obj_attr")); scales.append([
            float(target["obj_psr"]["scale"][axis]) for axis in ("x", "y", "z")])
        points = read_xyz_pcd(pcd_path); box = Box(target)
        member = (np.abs((points - box.center) @ box.rotation)
                  <= box.size / 2 + 1e-8).all(axis=1)
        indices, _ = project_visible(points, camera, cell_size=4,
                                     depth_slack=.12, near=.2)
        target_projection = project_box(target, camera)
        occluder_projection = project_box(occluder, camera) if occluder else None
        rows.append({
            "frame_id": frame, "target_present": True,
            "target_ignored": bool(target.get("ignore")),
            "evidence_eligible": not bool(target.get("ignore")),
            "target_attr": target.get("obj_attr"),
            "target_box_points": int(member.sum()),
            "target_visible_points": int(member[indices].sum()),
            "target_projectable": target_projection is not None,
            "occluder_projectable": occluder_projection is not None,
            "target_camera_depth": (target_projection or {}).get("camera_depth"),
            "occluder_camera_depth": (occluder_projection or {}).get("camera_depth"),
            "target_occluder_projected_overlap": projected_overlap(
                target_projection, occluder_projection),
        })
    center_index = next(index for index, row in enumerate(rows)
                        if row["frame_id"] == center_frame)
    evidence_rows = [row for row in rows if row.get("evidence_eligible")]
    supported = [row["target_visible_points"] >= 3 for row in evidence_rows]
    overlap_rows = [row for row in evidence_rows
                    if row["target_occluder_projected_overlap"] >= .5]
    clear_rows = [row for row in evidence_rows
                  if row["target_occluder_projected_overlap"] < .5]
    median_overlap = visible_point_median(overlap_rows)
    median_clear = visible_point_median(clear_rows)
    present, total = temporal_identity_counts(rows)
    eligible_present = len(evidence_rows)
    before = sum(row.get("evidence_eligible") and row.get("target_visible_points", 0) >= 3
                 for row in rows[:center_index])
    after = sum(row.get("evidence_eligible") and row.get("target_visible_points", 0) >= 3
                for row in rows[center_index + 1:])
    center_visible = rows[center_index].get("target_visible_points", 0)
    attr_consistent = bool(attrs) and len(set(attrs)) == 1
    scale_max_range = (float(np.ptp(np.asarray(scales), axis=0).max())
                       if scales else None)
    gate = bool(median_overlap is not None and median_clear is not None
                and temporal_occlusion_gate(
                    center_visible, before, after, present, total,
                    median_overlap, median_clear))
    presence_gate = bool(
        scale_max_range is not None and temporal_presence_gate(
            center_visible, before, after, present, total,
            attr_consistent, scale_max_range))
    verify_temporal_inputs(input_sha256)
    output.mkdir(parents=True)
    chart = output / f"obj{target_id}-temporal-visibility.png"
    render_chart(rows, center_frame, chart, target_id, occluder_id)
    report = {
        "schema_version": "labelagent.pallet_temporal_visibility.v1",
        "status": "offline_temporal_diagnostic_not_deployed",
        "manifest_sha256": input_sha256[str(manifest_path.resolve())],
        "input_sha256": input_sha256,
        "frozen_input_count": len(input_sha256),
        "sequence": sequence, "camera": camera_number,
        "target_obj_id": target_id, "occluder_obj_id": occluder_id,
        "center_frame_id": center_frame, "frame_count": len(rows),
        "target_present_frames": present,
        "evidence_eligible_frames": len(evidence_rows),
        "ignored_context_frames": sum(row.get("target_ignored", False) for row in rows),
        "target_supported_frames": sum(supported),
        "target_supported_before_center": before,
        "target_supported_after_center": after,
        "center_visible_points": center_visible,
        "median_visible_points_with_occluder_overlap": median_overlap,
        "median_visible_points_without_occluder_overlap": median_clear,
        "target_attr_consistent": attr_consistent,
        "target_scale_max_range": scale_max_range,
        "temporal_occlusion_supported_for_center": gate,
        "temporal_presence_supported_for_center": presence_gate,
        "interpretation": (
            "temporal_support_for_occlusion_not_pixel_truth" if gate else
            "temporal_support_for_presence_not_pixel_truth" if presence_gate else
            "temporal_evidence_insufficient"),
        "chart_path": str(chart.resolve()), "chart_sha256": digest(chart),
        "oss_writes": 0, "gpu_requests": 0, "workbench_writes": 0,
        "human_pixel_truth": False, "approved_for_training": False,
        "production_deployed": False, "rows": rows,
    }
    path = output / "results.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    report["report_path"] = str(path.resolve()); report["report_sha256"] = digest(path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate(args.manifest, args.data_root, args.calibration, args.output)
    print(json.dumps({key: report[key] for key in (
        "frame_count", "target_present_frames", "target_supported_frames",
        "target_supported_before_center", "target_supported_after_center",
        "center_visible_points", "median_visible_points_with_occluder_overlap",
        "median_visible_points_without_occluder_overlap",
        "temporal_occlusion_supported_for_center", "workbench_writes",
        "report_path", "report_sha256")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
