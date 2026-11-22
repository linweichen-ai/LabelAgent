#!/usr/bin/env python3
"""Cross-sequence 3D-box-localized pallet ROI SAM diagnostic, read-only."""

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
from PIL import Image, ImageDraw, ImageFont
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.compare_boxes3d import Box
from tools.evaluate_pallet_lidar_centered_roi_sam import (
    _crop_box, _distance_to_mask, _result_mask, select_associated_result,
)
from tools.evaluate_projected_support import camera_parameters
from tools.evaluate_slope_lidar_sam import request_json


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_camera(calibration: Path, camera: int) -> dict:
    extrinsics = yaml.safe_load((calibration / "lidar2camera.yaml").read_text(encoding="utf-8"))
    text = (calibration / f"camera{camera}_calib.yaml").read_text(encoding="utf-8")
    intrinsic = yaml.safe_load("\n".join(
        line for line in text.splitlines() if not line.startswith("%YAML:")))
    return camera_parameters(intrinsic, extrinsics[f"lidar_to_camera_{camera}"])


def project_box(obj: dict, camera: dict) -> dict | None:
    box = Box(obj)
    transform = camera["transform"]
    xyz = box.corners @ transform[:3, :3].T + transform[:3, 3]
    if not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= .2):
        return None
    fx, fy, cx, cy = camera["intrinsics"]
    uv = xyz[:, :2] / xyz[:, 2, None] * [fx, fy] + [cx, cy]
    center_xyz = box.center @ transform[:3, :3].T + transform[:3, 3]
    if center_xyz[2] <= .2:
        return None
    center = center_xyz[:2] / center_xyz[2] * [fx, fy] + [cx, cy]
    raw = [float(uv[:, 0].min()), float(uv[:, 1].min()),
           float(uv[:, 0].max()), float(uv[:, 1].max())]
    width, height = camera["width"], camera["height"]
    clipped = [max(0., raw[0]), max(0., raw[1]),
               min(width - 1., raw[2]), min(height - 1., raw[3])]
    raw_area = max(0., raw[2] - raw[0]) * max(0., raw[3] - raw[1])
    clipped_area = max(0., clipped[2] - clipped[0]) * max(0., clipped[3] - clipped[1])
    if raw_area <= 0 or clipped_area < 80 or clipped_area / raw_area < .6:
        return None
    hull = np.zeros((height, width), dtype=np.uint8)
    polygon = cv2.convexHull(np.rint(uv).astype(np.int32))
    cv2.fillConvexPoly(hull, polygon, 1)
    if not hull.any():
        return None
    return {
        "obj_id": obj.get("obj_id"), "obj_attr": obj.get("obj_attr"),
        "bbox": clipped, "raw_bbox": raw,
        "center": [float(center[0]), float(center[1])],
        "clipped_fraction": clipped_area / raw_area,
        "projected_pixels": int(hull.sum()), "mask": hull.astype(bool),
    }


def select_target(objects: list[dict], camera: dict, policy="largest_empty") -> tuple[dict, list[dict]] | tuple[None, list[dict]]:
    projected = []
    for obj in objects:
        if obj.get("obj_type") != "Pallet" or obj.get("ignore"):
            continue
        row = project_box(obj, camera)
        if row is not None:
            projected.append(row)
    eligible = [row for row in projected
                if 0 <= row["center"][0] < camera["width"]
                and 0 <= row["center"][1] < camera["height"]]
    if not eligible:
        return None, projected
    for row in eligible:
        area = int(row["mask"].sum())
        overlaps = [float((row["mask"] & other["mask"]).sum() / area)
                    for other in projected if other is not row] if area else []
        row["maximum_other_projected_box_fraction"] = max(overlaps, default=0.0)
        row["isolation_fraction"] = 1.0 - row["maximum_other_projected_box_fraction"]
    if policy == "largest_empty":
        eligible.sort(key=lambda row: (
            row.get("obj_attr") == "empty", row["projected_pixels"],
            -abs(row["center"][0] - camera["width"] / 2)), reverse=True)
    elif policy == "isolated":
        eligible.sort(key=lambda row: (
            row["isolation_fraction"] >= .75, row["isolation_fraction"],
            row["projected_pixels"], row.get("obj_attr") == "empty",
            -abs(row["center"][0] - camera["width"] / 2)), reverse=True)
    else:
        raise ValueError(f"unknown target policy: {policy}")
    return eligible[0], projected


def association_metrics(proposal, target, other_masks, center):
    proposal_area, target_area = int(proposal.sum()), int(target.sum())
    intersection = int((proposal & target).sum())
    other_fractions = [float((proposal & mask).sum() / proposal_area)
                       for mask in other_masks] if proposal_area else []
    return {
        "proposal_pixels": proposal_area,
        "projected_box_pixels": target_area,
        "projected_box_coverage": intersection / target_area if target_area else None,
        "proposal_inside_projected_box": intersection / proposal_area if proposal_area else None,
        "proposal_to_projected_box_area_ratio": proposal_area / target_area if target_area else None,
        "center_distance_pixels": _distance_to_mask(proposal, *center),
        "maximum_other_projected_box_fraction": max(other_fractions, default=0.0),
    }


def localization_gate(metrics):
    return bool(
        metrics["center_distance_pixels"] is not None
        and metrics["center_distance_pixels"] <= 24
        and metrics["projected_box_coverage"] is not None
        and metrics["projected_box_coverage"] >= .05
        and metrics["proposal_inside_projected_box"] is not None
        and metrics["proposal_inside_projected_box"] >= .3)


def point_fallback_groups(target, roi, crop_width, crop_height):
    cx = int(round(target["center"][0] - roi[0]))
    cy = int(round(target["center"][1] - roi[1]))
    positive = {"x": min(crop_width - 1, max(0, cx)),
                "y": min(crop_height - 1, max(0, cy)), "label": 1}
    corner_negatives = [
        {"x": 4, "y": 4, "label": 0},
        {"x": max(0, crop_width - 5), "y": 4, "label": 0},
        {"x": 4, "y": max(0, crop_height - 5), "label": 0},
        {"x": max(0, crop_width - 5), "y": max(0, crop_height - 5), "label": 0},
    ]
    x0, y0, x1, y1 = target["bbox"]
    x0, x1 = int(round(x0 - roi[0])), int(round(x1 - roi[0]))
    y0, y1 = int(round(y0 - roi[1])), int(round(y1 - roi[1]))
    margin = 8
    ring_negatives = [
        {"x": positive["x"], "y": max(0, y0 - margin), "label": 0},
        {"x": positive["x"], "y": min(crop_height - 1, y1 + margin), "label": 0},
        {"x": max(0, x0 - margin), "y": positive["y"], "label": 0},
        {"x": min(crop_width - 1, x1 + margin), "y": positive["y"], "label": 0},
    ]
    return [
        ("center_only", [positive]),
        ("crop_corners", [positive, *corner_negatives]),
        ("box_ring", [positive, *ring_negatives]),
    ]


def render_panel(image, projected, proposal, crop_box, title):
    base = np.asarray(image).copy()
    overlay = base.copy(); overlay[projected] = (45, 210, 135)
    proposed = base.copy(); proposed[proposal] = (255, 170, 35)
    difference = base.copy()
    difference[projected & proposal] = (45, 210, 135)
    difference[proposal & ~projected] = (255, 210, 40)
    difference[projected & ~proposal] = (255, 70, 70)
    views = [Image.fromarray(value).crop(tuple(crop_box)).resize((320, 240))
             for value in (base, overlay, proposed, difference)]
    panel = Image.new("RGB", (1280, 288), "white")
    for index, view in enumerate(views):
        panel.paste(view, (index * 320, 48))
    draw = ImageDraw.Draw(panel); font = ImageFont.load_default()
    draw.text((8, 7), title, fill="black", font=font)
    draw.text((8, 27), "image / projected 3D box / ROI text-SAM / overlap diagnostic",
              fill="black", font=font)
    return panel


def evaluate(manifest_path: Path, labels_root: Path, calibration: Path,
             output: Path, sam_url: str, *, timeout: float = 120,
             target_policy: str = "largest_empty", prompt_mode: str = "single",
             point_fallback: bool = False,
             point_fallback_mode: str = "single") -> dict:
    if output.exists():
        raise ValueError("output directory already exists")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("evaluation_only") is not True
            or manifest.get("training_return_allowed") is not False):
        raise ValueError("requires a frozen evaluation-only manifest")
    inputs = {str(manifest_path.resolve()): digest(manifest_path)}
    for path in sorted(calibration.glob("*.yaml")):
        inputs[str(path.resolve())] = digest(path)
    cameras = {camera: load_camera(calibration, camera) for camera in range(4)}
    root = Path(manifest["local_root"])
    health_before = request_json(sam_url.rstrip("/") + "/health", timeout=15)
    if health_before.get("status") != "ok" or health_before.get("model_loaded") is not True:
        raise ValueError("SAM service is not ready")
    output.mkdir(parents=True)
    rows, panels = [], []
    for item in manifest["files"]:
        sequence, frame = str(item["sequence"]), str(item["frame_id"])
        camera_number = int(str(item["camera"]).split("_")[-1])
        image_path = root / item["relative_path"]
        if digest(image_path) != item["sha256"]:
            raise ValueError(f"held-out image changed: {image_path}")
        label_path = labels_root / sequence / f"{frame}.json"
        inputs.setdefault(str(label_path.resolve()), digest(label_path))
        label = json.loads(label_path.read_text(encoding="utf-8"))
        if str(label.get("frame_id")) != frame:
            raise ValueError("label frame mismatch")
        target, projected = select_target(label.get("objects") or [], cameras[camera_number],
                                          policy=target_policy)
        row = {"sequence": sequence, "frame_id": frame, "camera": camera_number,
               "image_path": str(image_path), "image_sha256": item["sha256"],
               "target_available": target is not None, "sam_called": False,
               "localization_pass": False, "instance_isolation_pass": False,
               "point_fallback_called": False, "point_fallback_used": False,
               "human_pixel_truth": False,
               "candidate_only": True, "production_deployed": False}
        if target is None:
            rows.append(row); continue
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        roi = _crop_box(target["bbox"], width, height, minimum_side=160, padding=16)
        crop = image.crop(tuple(roi)); buffer = io.BytesIO(); crop.save(buffer, "PNG")
        started = time.monotonic()
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        if prompt_mode == "single":
            response = request_json(sam_url.rstrip("/") + "/segment_text", {
                "image": encoded, "prompt": "pallet"}, timeout=timeout)
            results = response.get("results") or []
        elif prompt_mode == "multi":
            response = request_json(sam_url.rstrip("/") + "/segment_text_multi", {
                "image": encoded, "prompts": {
                    "generic": "pallet", "wooden": "wooden pallet",
                    "plastic": "plastic pallet", "base": "pallet base"}}, timeout=timeout)
            results = []
            for prompt_name, payload in (response.get("prompts") or {}).items():
                for result in payload.get("results") or []:
                    results.append({**result, "prompt_name": prompt_name})
        else:
            raise ValueError(f"unknown prompt mode: {prompt_mode}")
        elapsed = time.monotonic() - started
        target_crop = target["mask"][roi[1]:roi[3], roi[0]:roi[2]]
        cx, cy = target["center"][0] - roi[0], target["center"][1] - roi[1]
        selected, local_mask, contains, distance, summaries = select_associated_result(
            results, target_crop, cx, cy, crop.width, crop.height)
        proposal = np.zeros((height, width), dtype=bool)
        row.update({"sam_called": True, "elapsed_seconds": round(elapsed, 4),
                    "target_obj_id": target["obj_id"], "target_obj_attr": target["obj_attr"],
                    "target_isolation_fraction": target["isolation_fraction"],
                    "projected_bbox": target["bbox"], "projected_center": target["center"],
                    "crop_box": roi, "model_result_count": len(results),
                    "selected_contains_projected_center": contains,
                    "selected_center_distance_pixels": distance,
                    "candidate_summaries": summaries})
        if selected is not None:
            proposal[roi[1]:roi[3], roi[0]:roi[2]] = local_mask
            other_masks = [item["mask"] for item in projected if item is not target]
            metrics = association_metrics(proposal, target["mask"], other_masks,
                                          target["center"])
            row.update({"sam_score": selected.get("score"),
                        "selected_prompt_name": selected.get("prompt_name", "generic"),
                        "metrics": metrics})
            row["localization_pass"] = localization_gate(metrics)
            row["instance_isolation_pass"] = bool(
                row["localization_pass"]
                and metrics["maximum_other_projected_box_fraction"] <= .25)
        if point_fallback and not row["localization_pass"]:
            row["point_fallback_called"] = True
            fallback_started = time.monotonic()
            try:
                groups = point_fallback_groups(target, roi, crop.width, crop.height)
                if point_fallback_mode == "single":
                    payload = request_json(sam_url.rstrip("/") + "/segment_points", {
                        "image": encoded, "points": groups[0][1]}, timeout=timeout)
                    items = [{"index": 0, "points": groups[0][1], "result": payload}]
                    diagnostics = {"image_encoded_once": True, "prompt_group_count": 1}
                elif point_fallback_mode == "batch":
                    payload = request_json(
                        sam_url.rstrip("/") + "/segment_point_groups_batch",
                        {"image": encoded, "prompt_groups": [points for _, points in groups]},
                        timeout=timeout)
                    items = payload.get("items") or []
                    diagnostics = payload.get("diagnostics") or {}
                else:
                    raise ValueError(f"unknown point fallback mode: {point_fallback_mode}")
                attempts, successful = [], []
                other_masks = [item["mask"] for item in projected if item is not target]
                for item in items:
                    index = int(item.get("index", 0))
                    policy = groups[index][0] if 0 <= index < len(groups) else f"group_{index}"
                    if "result" not in item:
                        attempts.append({"policy": policy, "error": item.get("error", "missing result")})
                        continue
                    point_result = item["result"]
                    point_local = _result_mask(point_result, crop.width, crop.height)
                    point_proposal = np.zeros((height, width), dtype=bool)
                    point_proposal[roi[1]:roi[3], roi[0]:roi[2]] = point_local
                    point_metrics = association_metrics(
                        point_proposal, target["mask"], other_masks, target["center"])
                    point_pass = localization_gate(point_metrics)
                    attempt = {"policy": policy, "score": point_result.get("score"),
                               "metrics": point_metrics, "localization_pass": point_pass,
                               "warnings": point_result.get("warnings") or []}
                    attempts.append(attempt)
                    successful.append((
                        point_pass,
                        float(point_metrics["proposal_inside_projected_box"] or 0),
                        float(point_metrics["projected_box_coverage"] or 0),
                        -float(point_metrics["maximum_other_projected_box_fraction"] or 0),
                        float(point_result.get("score", 0) or 0),
                        policy, point_result, point_proposal, point_metrics))
                successful.sort(key=lambda item: item[:5], reverse=True)
                selected_point = successful[0] if successful else None
                row["point_fallback"] = {
                    "mode": point_fallback_mode,
                    "elapsed_seconds": round(time.monotonic() - fallback_started, 4),
                    "diagnostics": diagnostics, "attempts": attempts,
                    "selected_policy": selected_point[5] if selected_point else None,
                    "localization_pass": bool(selected_point and selected_point[0]),
                }
                if selected_point and selected_point[0]:
                    policy, point_result, point_proposal, point_metrics = selected_point[5:]
                    proposal = point_proposal
                    row.update({
                        "point_fallback_used": True,
                        "selected_prompt_name": f"projected_center_point_{policy}",
                        "sam_score": point_result.get("score"), "metrics": point_metrics,
                        "localization_pass": True,
                        "instance_isolation_pass": (
                            point_metrics["maximum_other_projected_box_fraction"] <= .25),
                    })
            except Exception as error:
                row["point_fallback"] = {
                    "elapsed_seconds": round(time.monotonic() - fallback_started, 4),
                    "error_type": type(error).__name__, "error": str(error)[:1000],
                    "localization_pass": False,
                }
        panels.append(render_panel(
            image, target["mask"], proposal, roi,
            f"seq {sequence} camera {camera_number} obj {target['obj_id']} "
            f"attr={target['obj_attr']} loc={row['localization_pass']} "
            f"isolated={row['instance_isolation_pass']}"))
        rows.append(row)
    sheet = Image.new("RGB", (1280, sum(panel.height for panel in panels)), "white")
    top = 0
    for panel in panels:
        sheet.paste(panel, (0, top)); top += panel.height
    sheet_path = output / "cross-sequence-box3d-roi-sam.jpg"
    sheet.save(sheet_path, quality=90, optimize=True)
    health_after = request_json(sam_url.rstrip("/") + "/health", timeout=15)
    report = {
        "schema_version": "labelagent.pallet_box3d_roi_sam.v1",
        "status": "offline_localization_diagnostic_not_deployed",
        "scope": "3d_box_localization_agreement_not_pixel_boundary_accuracy",
        "target_selection": target_policy,
        "prompt_mode": prompt_mode,
        "point_fallback_enabled": point_fallback,
        "point_fallback_mode": point_fallback_mode,
        "input_sha256": inputs, "sequence_count": len(set(row["sequence"] for row in rows)),
        "image_count": len(rows), "target_available_count": sum(row["target_available"] for row in rows),
        "sam_frame_count": sum(row["sam_called"] for row in rows),
        "localization_pass_count": sum(row["localization_pass"] for row in rows),
        "instance_isolation_pass_count": sum(row["instance_isolation_pass"] for row in rows),
        "point_fallback_call_count": sum(row["point_fallback_called"] for row in rows),
        "point_fallback_used_count": sum(row["point_fallback_used"] for row in rows),
        "health_before": health_before, "health_after": health_after,
        "reference_labels_used": True, "human_pixel_truth": False,
        "workbench_writes": 0, "candidate_only": True,
        "approved_for_training": False, "production_deployed": False,
        "contact_sheet": str(sheet_path.resolve()), "contact_sheet_sha256": digest(sheet_path),
        "rows": rows,
    }
    report_path = output / "results.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    report["report_path"] = str(report_path.resolve()); report["report_sha256"] = digest(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--labels-root", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sam-url", default="http://127.0.0.1:28766")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--target-policy", choices=("largest_empty", "isolated"),
                        default="largest_empty")
    parser.add_argument("--prompt-mode", choices=("single", "multi"), default="single")
    parser.add_argument("--point-fallback", action="store_true")
    parser.add_argument("--point-fallback-mode", choices=("single", "batch"),
                        default="single")
    args = parser.parse_args()
    report = evaluate(args.manifest, args.labels_root, args.calibration, args.output,
                      args.sam_url, timeout=args.timeout,
                      target_policy=args.target_policy, prompt_mode=args.prompt_mode,
                      point_fallback=args.point_fallback,
                      point_fallback_mode=args.point_fallback_mode)
    keys = ("sequence_count", "image_count", "target_available_count", "sam_frame_count",
            "localization_pass_count", "instance_isolation_pass_count",
            "point_fallback_call_count", "point_fallback_used_count",
            "workbench_writes", "contact_sheet",
            "contact_sheet_sha256", "report_path", "report_sha256")
    print(json.dumps({key: report[key] for key in keys}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
