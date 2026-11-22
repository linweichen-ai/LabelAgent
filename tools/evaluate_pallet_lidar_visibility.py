#!/usr/bin/env python3
"""Validate ambiguous pallet visibility with frozen LiDAR returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.compare_boxes3d import Box
from tools.evaluate_pallet_box3d_overlap_pairs import projected_conflicts
from tools.evaluate_pallet_box3d_roi_sam import digest, load_camera, project_box
from tools.evaluate_projected_support import project_visible
from tools.slope_lidar_geometry import read_xyz_pcd


GRID = [(cell, slack) for cell in (2, 4, 8) for slack in (.05, .12, .20)]


def classify_visibility(box_points: int, in_frame_points: int,
                        visible_counts: list[int], nearer_overlap: float) -> str:
    if box_points == 0:
        return "no_lidar_return_insufficient_evidence"
    if visible_counts and min(visible_counts) >= 3:
        return "visible_lidar_supported"
    if (in_frame_points >= 5 and visible_counts and max(visible_counts) <= 1
            and nearer_overlap >= .7):
        return "occluded_lidar_supported"
    return "sparse_or_parameter_sensitive"


def in_camera_frame(points: np.ndarray, camera: dict, near=.2) -> np.ndarray:
    transform = camera["transform"]
    xyz = points @ transform[:3, :3].T + transform[:3, 3]
    fx, fy, cx, cy = camera["intrinsics"]
    uv = xyz[:, :2] / xyz[:, 2, None] * [fx, fy] + [cx, cy]
    return ((xyz[:, 2] > near) & (uv[:, 0] >= 0)
            & (uv[:, 0] < camera["width"] - .5) & (uv[:, 1] >= 0)
            & (uv[:, 1] < camera["height"] - .5))


def membership(points: np.ndarray, obj: dict) -> np.ndarray:
    box = Box(obj)
    return (np.abs((points - box.center) @ box.rotation)
            <= box.size / 2 + 1e-8).all(axis=1)


def render(image: Image.Image, projected: list[dict], pixels: np.ndarray,
           visible_memberships: dict[str, np.ndarray], rows: list[dict],
           output: Path) -> None:
    canvas = np.asarray(image).copy()
    colors = [(35, 210, 125), (255, 160, 35), (90, 125, 255), (220, 75, 180)]
    legend = []
    for index, projected_row in enumerate(projected):
        key = str(projected_row["obj_id"]); color = colors[index % len(colors)]
        selected = pixels[visible_memberships[key]]
        for x, y in selected:
            cv2.circle(canvas, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)
        contour = cv2.findContours(projected_row["mask"].astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(canvas, contour, -1, color, 1, cv2.LINE_AA)
        detail = next(row for row in rows if str(row["obj_id"]) == key)
        cx, cy = map(int, projected_row["center"])
        cv2.putText(canvas, key, (max(0, cx - 6), max(15, cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
        status = {
            "visible_lidar_supported": "visible",
            "occluded_lidar_supported": "occluded",
            "no_lidar_return_insufficient_evidence": "no-return",
            "sparse_or_parameter_sensitive": "uncertain",
        }[detail["lidar_visibility_classification"]]
        legend.append((color, f"obj {key}: box {detail['box_points']}  "
                       f"visible {detail['default_visible_points']}  {status}"))
    if legend:
        overlay = canvas.copy(); height = 12 + len(legend) * 20
        cv2.rectangle(overlay, (8, 8), (312, height), (12, 20, 27), -1)
        canvas = cv2.addWeighted(overlay, .78, canvas, .22, 0)
        for index, (color, text) in enumerate(legend):
            y = 25 + index * 20
            cv2.circle(canvas, (19, y - 4), 4, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, text, (31, y), cv2.FONT_HERSHEY_SIMPLEX,
                        .42, (242, 246, 248), 1, cv2.LINE_AA)
    Image.fromarray(canvas).save(output)


def evaluate(source_report: Path, manifest_path: Path, pcd_manifest_path: Path,
             labels_root: Path, calibration: Path, pcd_root: Path,
             output: Path) -> dict:
    if output.exists():
        raise ValueError("output directory already exists")
    source = json.loads(source_report.read_text(encoding="utf-8"))
    if (source.get("scope") != "3d_box_localization_agreement_not_pixel_boundary_accuracy"
            or source.get("production_deployed") is not False):
        raise ValueError("requires the frozen non-deployed 3D ROI report")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pcd_manifest = json.loads(pcd_manifest_path.read_text(encoding="utf-8"))
    files = {(str(row["sequence"]), int(str(row["camera"]).split("_")[-1])): row
             for row in manifest["files"]}
    pcd_files = {str(row["sequence"]): row for row in pcd_manifest["files"]}
    image_root = Path(manifest["local_root"])
    output.mkdir(parents=True); visuals = output / "visuals"; visuals.mkdir()
    rows = []
    ambiguous = [row for row in source["rows"] if row.get("localization_pass")
                 and not row.get("instance_isolation_pass")]
    for source_row in ambiguous:
        sequence = str(source_row["sequence"]); camera_number = int(source_row["camera"])
        item = files[(sequence, camera_number)]; frame = str(item["frame_id"])
        image_path = image_root / item["relative_path"]
        if digest(image_path) != item["sha256"]:
            raise ValueError("held-out image changed")
        pcd_item = pcd_files[sequence]; pcd_path = pcd_root / pcd_item["filename"]
        if frame != str(pcd_item["frame_id"]) or digest(pcd_path) != pcd_item["sha256"]:
            raise ValueError("PCD does not match frozen manifest")
        points = read_xyz_pcd(pcd_path)
        label = json.loads((labels_root / sequence / f"{frame}.json").read_text(
            encoding="utf-8"))
        objects = {str(obj.get("obj_id")): obj for obj in label["objects"]}
        camera = load_camera(calibration, camera_number)
        projected = [value for obj in label["objects"]
                     if obj.get("obj_type") == "Pallet" and not obj.get("ignore")
                     if (value := project_box(obj, camera)) is not None]
        target = next(row for row in projected
                      if str(row["obj_id"]) == str(source_row["target_obj_id"]))
        selected = [target, *(row[0] for row in projected_conflicts(target, projected))]
        in_frame = in_camera_frame(points, camera)
        visible_by_grid = {}
        for cell, slack in GRID:
            indices, pixels = project_visible(points, camera, cell_size=cell,
                                              depth_slack=slack, near=.2)
            visible_by_grid[(cell, slack)] = (indices, pixels)
        default_indices, default_pixels = visible_by_grid[(4, .12)]
        detail_rows, default_memberships = [], {}
        for projected_row in selected:
            key = str(projected_row["obj_id"]); member = membership(points, objects[key])
            visible_counts = []
            for cell, slack in GRID:
                indices, _ = visible_by_grid[(cell, slack)]
                visible_counts.append(int(member[indices].sum()))
            nearer = [other for other in selected
                      if float(other["camera_depth"]) < float(projected_row["camera_depth"])]
            projected_pixels = int(projected_row["mask"].sum())
            nearer_overlap = max((int((projected_row["mask"] & other["mask"]).sum())
                                  / projected_pixels for other in nearer), default=0.0)
            default_member = member[default_indices]
            default_memberships[key] = default_member
            detail_rows.append({
                "obj_id": projected_row["obj_id"],
                "camera_depth": projected_row["camera_depth"],
                "box_points": int(member.sum()),
                "in_frame_box_points": int((member & in_frame).sum()),
                "default_visible_points": int(default_member.sum()),
                "default_visible_pixels": int(len(np.unique(
                    default_pixels[default_member], axis=0))) if default_member.any() else 0,
                "visible_counts_grid_order": visible_counts,
                "minimum_visible_points": min(visible_counts),
                "maximum_visible_points": max(visible_counts),
                "maximum_nearer_projected_box_overlap": nearer_overlap,
                "lidar_visibility_classification": classify_visibility(
                    int(member.sum()), int((member & in_frame).sum()),
                    visible_counts, nearer_overlap),
            })
        visual = visuals / f"{sequence}-camera{camera_number}-lidar-visibility.png"
        render(Image.open(image_path).convert("RGB"), selected, default_pixels,
               default_memberships, detail_rows, visual)
        rows.append({"sequence": sequence, "frame_id": frame,
                     "camera": camera_number, "target_obj_id": target["obj_id"],
                     "pcd_path": str(pcd_path.resolve()), "pcd_sha256": digest(pcd_path),
                     "point_count": len(points), "instances": detail_rows,
                     "visual_path": str(visual.resolve()), "visual_sha256": digest(visual)})
    classifications = [item["lidar_visibility_classification"]
                       for row in rows for item in row["instances"]]
    report = {
        "schema_version": "labelagent.pallet_lidar_visibility.v1",
        "status": "offline_lidar_visibility_diagnostic_not_deployed",
        "source_report_sha256": digest(source_report),
        "pcd_manifest_sha256": digest(pcd_manifest_path),
        "grid": [{"cell_size": cell, "depth_slack": slack} for cell, slack in GRID],
        "ambiguous_image_count": len(rows),
        "visible_lidar_supported_instances": classifications.count(
            "visible_lidar_supported"),
        "occluded_lidar_supported_instances": classifications.count(
            "occluded_lidar_supported"),
        "insufficient_evidence_instances": classifications.count(
            "no_lidar_return_insufficient_evidence"),
        "workbench_writes": 0, "gpu_requests": 0, "human_pixel_truth": False,
        "approved_for_training": False, "production_deployed": False,
        "rows": rows,
    }
    path = output / "results.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    report["report_path"] = str(path.resolve()); report["report_sha256"] = digest(path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pcd-manifest", required=True, type=Path)
    parser.add_argument("--labels-root", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--pcd-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate(args.source_report, args.manifest, args.pcd_manifest,
                      args.labels_root, args.calibration, args.pcd_root, args.output)
    print(json.dumps({key: report[key] for key in (
        "ambiguous_image_count", "visible_lidar_supported_instances",
        "occluded_lidar_supported_instances", "insufficient_evidence_instances",
        "workbench_writes", "gpu_requests", "report_path", "report_sha256")},
        ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
