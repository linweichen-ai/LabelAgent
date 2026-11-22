#!/usr/bin/env python3
"""Try LiDAR-return point prompts only for frozen failed pallet splits."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.compare_boxes3d import Box
from tools.evaluate_pallet_box3d_overlap_pairs import (
    depth_split_masks, index_manifest_files, mask_separation,
    projected_conflicts,
)
from tools.evaluate_pallet_box3d_roi_sam import (
    association_metrics, digest, load_camera, localization_gate, project_box,
)
from tools.evaluate_pallet_lidar_centered_roi_sam import _crop_box, _result_mask
from tools.evaluate_pallet_lidar_visibility import index_pcd_files
from tools.evaluate_projected_support import project_visible
from tools.evaluate_slope_lidar_sam import request_json
from tools.slope_lidar_geometry import read_xyz_pcd


def membership(points: np.ndarray, obj: dict) -> np.ndarray:
    box = Box(obj)
    return (np.abs((points - box.center) @ box.rotation)
            <= box.size / 2 + 1e-8).all(axis=1)


def spread_points(points: np.ndarray, limit: int) -> list[list[int]]:
    unique = np.unique(np.asarray(points, dtype=np.int32), axis=0)
    if not len(unique) or limit <= 0:
        return []
    # np.unique is lexicographically sorted, giving a deterministic endpoint;
    # farthest-point sampling then covers the visible return extent.
    selected = [0]
    while len(selected) < min(limit, len(unique)):
        distance = np.min(
            ((unique[:, None, :] - unique[selected][None, :, :]) ** 2).sum(axis=2),
            axis=1)
        distance[selected] = -1
        selected.append(int(np.argmax(distance)))
    return unique[selected].tolist()


def unique_instance_pixels(points: np.ndarray, visible_indices: np.ndarray,
                           visible_pixels: np.ndarray, target_obj: dict,
                           conflict_objects: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    target = membership(points, target_obj)[visible_indices]
    conflicts = [membership(points, obj)[visible_indices]
                 for obj in conflict_objects]
    other = np.logical_or.reduce(conflicts) if conflicts else np.zeros_like(target)
    positive = visible_pixels[target & ~other]
    negative = visible_pixels[other & ~target]
    return positive, negative


def prompt_groups(positive: list[list[int]], negative: list[list[int]], roi,
                  scale: int = 1):
    def local(point, label):
        # Preserve the original pixel-center location when the ROI is enlarged.
        x = round((point[0] - roi[0] + .5) * scale - .5)
        y = round((point[1] - roi[1] + .5) * scale - .5)
        return {"x": int(x), "y": int(y),
                "label": label}
    positives = [local(point, 1) for point in positive]
    negatives = [local(point, 0) for point in negative]
    groups = []
    if positives:
        groups.append(("one_positive_all_negative", [positives[0], *negatives]))
    if len(positives) >= 2:
        groups.append(("three_positive_all_negative", [*positives[:3], *negatives]))
    if len(positives) >= 2 and len(negatives) >= 2:
        groups.append(("three_positive_two_negative",
                       [*positives[:3], *negatives[:2]]))
    return groups


def downsample_mask(mask: np.ndarray, width: int, height: int,
                    threshold: float = .5) -> np.ndarray:
    """Map a model-scale mask back without accepting sparse subpixel spill."""
    if mask.shape == (height, width):
        return mask.astype(bool, copy=True)
    coverage = cv2.resize(mask.astype(np.float32), (width, height),
                          interpolation=cv2.INTER_AREA)
    return coverage >= threshold


def accepted_candidate(metrics: dict, positive_coverage: float,
                       negative_clear_fraction: float, split_metrics: list[dict],
                       separations: list[dict]) -> bool:
    return bool(
        localization_gate(metrics)
        and positive_coverage >= .8
        and negative_clear_fraction >= .8
        and split_metrics
        and all(row["candidate_pass"] for row in split_metrics)
        and separations
        and all(row["separated_at_0_15"] for row in separations)
    )


DEFAULT_ROI_SCALE = 2


def render_panel(image, proposal, split, positives, negatives, roi, title):
    base = image.copy(); overlay = image.copy(); split_view = image.copy()
    overlay_array = np.asarray(overlay).copy()
    overlay_array[proposal] = (np.asarray(overlay_array[proposal], np.uint16)
                               + np.array([255, 160, 35])) // 2
    split_array = np.asarray(split_view).copy()
    split_array[split] = (np.asarray(split_array[split], np.uint16)
                          + np.array([45, 210, 135])) // 2
    views = [base, Image.fromarray(overlay_array.astype(np.uint8)),
             Image.fromarray(split_array.astype(np.uint8))]
    panel = Image.new("RGB", (960, 286), "white")
    for index, view in enumerate(views):
        crop = view.crop(tuple(roi)).resize((320, 240))
        draw = ImageDraw.Draw(crop)
        sx, sy = 320 / (roi[2] - roi[0]), 240 / (roi[3] - roi[1])
        for x, y in positives:
            draw.ellipse(((x - roi[0]) * sx - 3, (y - roi[1]) * sy - 3,
                          (x - roi[0]) * sx + 3, (y - roi[1]) * sy + 3), fill="lime")
        for x, y in negatives:
            draw.ellipse(((x - roi[0]) * sx - 3, (y - roi[1]) * sy - 3,
                          (x - roi[0]) * sx + 3, (y - roi[1]) * sy + 3), fill="red")
        panel.paste(crop, (index * 320, 46))
    ImageDraw.Draw(panel).text((8, 8), title, fill="black")
    ImageDraw.Draw(panel).text(
        (8, 27), "image / SAM proposal / depth-split target; green positive, red negative",
        fill="black")
    return panel


def evaluate(split_report_path: Path, manifest_path: Path, pcd_manifest_path: Path,
             labels_root: Path, calibration: Path, pcd_root: Path, output: Path,
             sam_url: str, timeout: float = 120,
             roi_scale: int = DEFAULT_ROI_SCALE) -> dict:
    if output.exists():
        raise ValueError("output directory already exists")
    if roi_scale not in (1, 2, 4):
        raise ValueError("roi_scale must be one of 1, 2 or 4")
    split_report = json.loads(split_report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pcd_manifest = json.loads(pcd_manifest_path.read_text(encoding="utf-8"))
    if (split_report.get("production_deployed") is not False
            or manifest.get("evaluation_only") is not True):
        raise ValueError("requires frozen offline evaluation inputs")
    inputs = {str(path.resolve()): digest(path) for path in (
        split_report_path, manifest_path, pcd_manifest_path)}
    files = index_manifest_files(manifest["files"])
    pcd_files = index_pcd_files(pcd_manifest["files"])
    image_root = Path(manifest["local_root"])
    output.mkdir(parents=True); masks_dir = output / "masks"; masks_dir.mkdir()
    panels, rows = [], []
    health_before = request_json(sam_url.rstrip("/") + "/health", timeout=15)
    for split_row in split_report["rows"]:
        failed = [metric for metric in split_row["depth_split_metrics"]
                  if not metric["candidate_pass"]]
        for failed_metric in failed:
            sequence = str(split_row["sequence"]); frame = str(split_row["frame_id"])
            camera_number = int(split_row["camera"]); target_id = str(failed_metric["obj_id"])
            item = files[(sequence, camera_number, frame)]
            image_path = image_root / item["relative_path"]
            pcd_item = pcd_files[(sequence, frame)]
            pcd_path = pcd_root / pcd_item["filename"]
            if digest(image_path) != item["sha256"] or digest(pcd_path) != pcd_item["sha256"]:
                raise ValueError("frozen image or PCD changed")
            label_path = labels_root / sequence / f"{frame}.json"
            label = json.loads(label_path.read_text(encoding="utf-8"))
            objects = {str(obj.get("obj_id")): obj for obj in label["objects"]}
            camera = load_camera(calibration, camera_number)
            projected = [value for obj in label["objects"]
                         if obj.get("obj_type") == "Pallet" and not obj.get("ignore")
                         if (value := project_box(obj, camera)) is not None]
            target = next(row for row in projected if str(row["obj_id"]) == target_id)
            conflicts = [row[0] for row in projected_conflicts(target, projected)]
            selected = [target, *conflicts]
            points = read_xyz_pcd(pcd_path)
            visible_indices, visible_pixels = project_visible(
                points, camera, cell_size=4, depth_slack=.12, near=.2)
            positive_pixels, negative_pixels = unique_instance_pixels(
                points, visible_indices, visible_pixels, objects[target_id],
                [objects[str(row["obj_id"])] for row in conflicts])
            positives = spread_points(positive_pixels, 3)
            negatives = spread_points(negative_pixels, 4)
            bbox = [min(row["bbox"][0] for row in selected),
                    min(row["bbox"][1] for row in selected),
                    max(row["bbox"][2] for row in selected),
                    max(row["bbox"][3] for row in selected)]
            image = Image.open(image_path).convert("RGB"); width, height = image.size
            roi = _crop_box(bbox, width, height, minimum_side=240, padding=24)
            positives = [point for point in positives
                         if roi[0] <= point[0] < roi[2] and roi[1] <= point[1] < roi[3]]
            negatives = [point for point in negatives
                         if roi[0] <= point[0] < roi[2] and roi[1] <= point[1] < roi[3]]
            groups = prompt_groups(positives, negatives, roi, roi_scale)
            crop = image.crop(tuple(roi))
            model_crop = (crop if roi_scale == 1 else crop.resize(
                (crop.width * roi_scale, crop.height * roi_scale),
                Image.Resampling.LANCZOS))
            stream = io.BytesIO(); model_crop.save(stream, "PNG")
            encoded = base64.b64encode(stream.getvalue()).decode("ascii")
            payload = request_json(sam_url.rstrip("/") + "/segment_point_groups_batch", {
                "image": encoded, "prompt_groups": [points for _, points in groups]},
                timeout=timeout) if groups else {"items": []}
            other_masks = {}
            for conflict in conflicts:
                path = Path(split_row["depth_split_masks"][str(conflict["obj_id"])]["path"])
                other_masks[conflict["obj_id"]] = cv2.imread(
                    str(path), cv2.IMREAD_GRAYSCALE) > 0
            attempts = []
            for result_item in payload.get("items") or []:
                index = int(result_item.get("index", 0)); policy = groups[index][0]
                if "result" not in result_item:
                    attempts.append({"policy": policy, "error": result_item.get("error")})
                    continue
                value = result_item["result"]
                model_local = _result_mask(
                    value, model_crop.width, model_crop.height)
                local = downsample_mask(model_local, crop.width, crop.height)
                proposal = np.zeros((height, width), dtype=bool)
                proposal[roi[1]:roi[3], roi[0]:roi[2]] = local
                metrics = association_metrics(
                    proposal, target["mask"], [row["mask"] for row in conflicts],
                    target["center"])
                positive_coverage = (sum(proposal[y, x] for x, y in positives)
                                     / len(positives) if positives else 0.0)
                negative_clear = (sum(not proposal[y, x] for x, y in negatives)
                                  / len(negatives) if negatives else 0.0)
                candidate_masks = {target["obj_id"]: proposal, **other_masks}
                split_masks, split_metrics = depth_split_masks(selected, candidate_masks)
                separations = [mask_separation(
                    split_masks[target["obj_id"]], split_masks[row["obj_id"]])
                    for row in conflicts]
                accepted = accepted_candidate(
                    metrics, positive_coverage, negative_clear,
                    split_metrics, separations)
                # Keep Windows artifact paths below legacy MAX_PATH; the full
                # policy name remains in the JSON receipt.
                mask_path = masks_dir / f"f{frame[-8:]}-c{camera_number}-o{target_id}-p{index}.png"
                Image.fromarray(split_masks[target["obj_id"]].astype(np.uint8) * 255).save(mask_path)
                attempt = {
                    "policy": policy, "sam_score": value.get("score"),
                    "roi_scale": roi_scale,
                    "model_mask_pixels": int(model_local.sum()),
                    "sam_mask_pixels": int(proposal.sum()), "metrics": metrics,
                    "positive_coverage": positive_coverage,
                    "negative_clear_fraction": negative_clear,
                    "split_metrics": split_metrics, "separations": separations,
                    "split_target_pixels": next(
                        row["split_pixels"] for row in split_metrics
                        if str(row["obj_id"]) == target_id),
                    "accepted": accepted, "mask_path": str(mask_path.resolve()),
                    "mask_sha256": digest(mask_path),
                }
                attempts.append(attempt)
                panels.append(render_panel(
                    image, proposal, split_masks[target["obj_id"]], positives, negatives,
                    roi, f"frame {frame} camera {camera_number} obj {target_id} {policy} accepted={accepted}"))
            accepted_attempts = [item for item in attempts if item.get("accepted")]
            accepted_attempts.sort(key=lambda item: (
                float(item["negative_clear_fraction"]),
                float(item["positive_coverage"]),
                -int(item["split_target_pixels"]),
            ), reverse=True)
            selected_attempt = accepted_attempts[0] if accepted_attempts else None
            rows.append({
                "sequence": sequence, "frame_id": frame, "camera": camera_number,
                "target_obj_id": failed_metric["obj_id"],
                "roi_scale": roi_scale,
                "model_input_size": [model_crop.width, model_crop.height],
                "visible_unique_positive_count": int(len(positive_pixels)),
                "visible_unique_negative_count": int(len(negative_pixels)),
                "selected_positive_points": positives,
                "selected_negative_points": negatives,
                "prompt_group_count": len(groups), "attempts": attempts,
                "recovered": selected_attempt is not None,
                "selected_policy": (selected_attempt or {}).get("policy"),
                "selected_mask_path": (selected_attempt or {}).get("mask_path"),
                "selected_mask_sha256": (selected_attempt or {}).get("mask_sha256"),
                "selected_mask_pixels": (selected_attempt or {}).get(
                    "split_target_pixels", 0),
            })
    sheet = Image.new("RGB", (960, max(1, sum(panel.height for panel in panels))), "white")
    top = 0
    for panel in panels:
        sheet.paste(panel, (0, top)); top += panel.height
    sheet_path = output / "lidar-conflict-point-review.jpg"; sheet.save(sheet_path, quality=90)
    health_after = request_json(sam_url.rstrip("/") + "/health", timeout=15)
    if any(digest(Path(path)) != value for path, value in inputs.items()):
        raise RuntimeError("frozen input changed during evaluation")
    report = {
        "schema_version": "labelagent.pallet_lidar_conflict_points.v1",
        "status": "offline_candidate_diagnostic_not_deployed",
        "roi_scale": roi_scale,
        "input_sha256": inputs, "failed_candidate_count": len(rows),
        "recovered_candidate_count": sum(row["recovered"] for row in rows),
        "sam_attempt_count": sum(len(row["attempts"]) for row in rows),
        "health_before": health_before, "health_after": health_after,
        "contact_sheet": str(sheet_path.resolve()),
        "contact_sheet_sha256": digest(sheet_path),
        "human_pixel_truth": False, "approved_for_training": False,
        "production_deployed": False, "workbench_writes": 0,
        "training_data_writes": 0, "oss_writes": 0, "rows": rows,
    }
    report_path = output / "results.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    report["report_path"] = str(report_path.resolve())
    report["report_sha256"] = digest(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-report", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pcd-manifest", required=True, type=Path)
    parser.add_argument("--labels-root", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--pcd-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sam-url", default="http://127.0.0.1:28766")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--roi-scale", type=int, choices=(1, 2, 4), default=DEFAULT_ROI_SCALE,
        help=("enlarge the frozen ROI before SAM; 2x is the independently "
              "repeated offline incumbent and all gates stay unchanged"))
    args = parser.parse_args()
    report = evaluate(
        args.split_report, args.manifest, args.pcd_manifest, args.labels_root,
        args.calibration, args.pcd_root, args.output, args.sam_url, args.timeout,
        args.roi_scale)
    print(json.dumps({key: report[key] for key in (
        "failed_candidate_count", "recovered_candidate_count", "sam_attempt_count",
        "workbench_writes", "contact_sheet", "contact_sheet_sha256",
        "report_path", "report_sha256")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
