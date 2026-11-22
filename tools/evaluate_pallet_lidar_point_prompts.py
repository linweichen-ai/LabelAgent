#!/usr/bin/env python3
"""Evaluate label-free LiDAR prompt points for partial distant pallets.

Only raw point clouds, calibration, source images and frozen SAM candidates are
read.  Reference 3D-label directories are denied by an audit hook.  Returned
SAM masks remain offline candidates and are never written to the workbench.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.agent.edit_agent import _rasterize, valid_shape
from tools.audit_temporal_distant_pallet_recovery import _ranked_targets
from tools.evaluate_slope_lidar_sam import polygon_mask, request_json
from tools.lidar_candidate_evidence import (
    ReferenceReadAudit,
    fit_support_plane,
    projected_sensor,
    read_xyz,
    sensor_cameras,
)
from tools.review_distant_pallet_recovery import (
    _camera_name,
    _crop,
    _difference_overlay,
    _fit,
    _mapping,
    _mask_overlay,
    _mask_record,
    _public_metrics,
    match_expansion,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_spread(points: np.ndarray, count: int, *, anchor: np.ndarray | None = None
                   ) -> list[tuple[int, int]]:
    """Deterministically select spatially spread unique image points."""
    unique = np.unique(np.asarray(points, dtype=np.int32), axis=0)
    if not len(unique) or count <= 0:
        return []
    center = unique.mean(axis=0) if anchor is None else np.asarray(anchor, dtype=float)
    first = int(np.argmin(np.sum((unique - center) ** 2, axis=1)))
    selected = [first]
    while len(selected) < min(count, len(unique)):
        distances = np.min(
            np.sum((unique[:, None, :] - unique[selected][None, :, :]) ** 2, axis=2),
            axis=1)
        distances[selected] = -1
        selected.append(int(np.argmax(distances)))
    return [(int(unique[index, 0]), int(unique[index, 1])) for index in selected]


def select_lidar_prompt_points(candidate_mask: np.ndarray,
                               existing_pallet_union: np.ndarray,
                               pixels: np.ndarray, heights: np.ndarray, *,
                               low_height: tuple[float, float] = (.04, .45),
                               floor_band: float = .04,
                               seed_radius: int = 20,
                               component_radius: int = 18,
                               maximum_positive_points: int = 5,
                               maximum_negative_points: int = 4) -> dict:
    """Build conservative pallet points from raw low-height sensor returns."""
    mask = np.asarray(candidate_mask, dtype=bool)
    occupied = np.asarray(existing_pallet_union, dtype=bool)
    pixels = np.asarray(pixels, dtype=np.int32)
    heights = np.asarray(heights, dtype=float)
    if mask.ndim != 2 or occupied.shape != mask.shape:
        raise ValueError("candidate and existing pallet masks must align")
    if pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) != len(heights):
        raise ValueError("projected pixels and heights must align")
    if not mask.any():
        raise ValueError("candidate mask is empty")
    height, width = mask.shape
    in_bounds = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width)
                 & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
                 & np.isfinite(heights))
    pixels, heights = pixels[in_bounds], heights[in_bounds]
    low = (heights >= low_height[0]) & (heights <= low_height[1])
    floor = np.abs(heights) <= floor_band
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    span = max(x1 - x0 + 1, y1 - y0 + 1)
    padding = max(48, min(192, span * 2))
    roi = ((pixels[:, 0] >= max(0, x0 - padding))
           & (pixels[:, 0] <= min(width - 1, x1 + padding))
           & (pixels[:, 1] >= max(0, y0 - padding))
           & (pixels[:, 1] <= min(height - 1, y1 + padding)))
    seed_mask = cv2.dilate(
        mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (seed_radius * 2 + 1, seed_radius * 2 + 1))) > 0
    seed = low & roi & seed_mask[pixels[:, 1], pixels[:, 0]]
    if not np.any(seed):
        return {
            "ready": False,
            "reason": "no_low_height_seed_near_candidate",
            "low_points_in_roi": int(np.count_nonzero(low & roi)),
            "floor_points_in_roi": int(np.count_nonzero(floor & roi)),
        }
    low_raster = np.zeros_like(mask, dtype=np.uint8)
    low_pixels = pixels[low & roi]
    low_raster[low_pixels[:, 1], low_pixels[:, 0]] = 1
    linked = cv2.dilate(
        low_raster,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (component_radius * 2 + 1,
                                   component_radius * 2 + 1)))
    count, components = cv2.connectedComponents((linked > 0).astype(np.uint8), 8)
    seed_labels = components[pixels[seed, 1], pixels[seed, 0]]
    seed_labels = seed_labels[seed_labels > 0]
    if not len(seed_labels):
        return {"ready": False, "reason": "low_height_seed_has_no_component"}
    labels, frequencies = np.unique(seed_labels, return_counts=True)
    selected_label = int(labels[int(np.argmax(frequencies))])
    component_points = low_pixels[
        components[low_pixels[:, 1], low_pixels[:, 0]] == selected_label]
    if not len(component_points):
        return {"ready": False, "reason": "empty_low_height_component"}
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2], dtype=float)
    positives = _sample_spread(
        component_points, maximum_positive_points, anchor=center)
    component_binary = components == selected_label
    exclusion = cv2.dilate(
        (component_binary | occupied).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))) > 0
    negative_pixels = pixels[
        floor & roi & ~exclusion[pixels[:, 1], pixels[:, 0]]]
    negatives = _sample_spread(
        negative_pixels, maximum_negative_points, anchor=center)
    if len(negatives) < 2:
        return {
            "ready": False,
            "reason": "insufficient_ground_negative_points",
            "positive_points": len(positives),
            "negative_points": len(negatives),
            "low_component_points": int(len(component_points)),
        }
    points = ([{"x": x, "y": y, "label": 1} for x, y in positives]
              + [{"x": x, "y": y, "label": 0} for x, y in negatives])
    return {
        "ready": True,
        "reason": None,
        "points": points,
        "positive_points": len(positives),
        "negative_points": len(negatives),
        "low_points_in_roi": int(np.count_nonzero(low & roi)),
        "floor_points_in_roi": int(np.count_nonzero(floor & roi)),
        "low_component_points": int(len(component_points)),
        "low_component_bbox": [
            int(component_points[:, 0].min()), int(component_points[:, 1].min()),
            int(component_points[:, 0].max()), int(component_points[:, 1].max()),
        ],
        "roi_bbox": [
            max(0, x0 - padding), max(0, y0 - padding),
            min(width - 1, x1 + padding), min(height - 1, y1 + padding),
        ],
    }


def _shape_from_result(result: dict, identifier: str) -> dict:
    shape = {
        "id": identifier,
        "label": "pallet",
        "score": float(result.get("score", 0) or 0),
        "points": result.get("points") or [],
        "holes": result.get("holes") or [],
        "parts": result.get("parts") or [],
    }
    if not shape["parts"] and shape["points"]:
        shape["parts"] = [{"points": shape["points"], "holes": shape["holes"]}]
    return shape


def point_policies(points: list[dict]) -> list[tuple[str, list[dict]]]:
    """Try a conservative anchor first, then add spatial low-height support."""
    positives = [point for point in points if point.get("label") == 1]
    negatives = [point for point in points if point.get("label") == 0]
    if not positives or len(negatives) < 2:
        raise ValueError("point policies need one positive and two negatives")
    policies = [("anchor_ring", [positives[0], *negatives])]
    if len(positives) > 1:
        policies.append(("extent_ring", [positives[0], positives[1], *negatives]))
    policies.append(("all_low_ring", [*positives, *negatives]))
    output = []
    seen = set()
    for name, values in policies:
        key = tuple((point["x"], point["y"], point["label"]) for point in values)
        if key not in seen:
            output.append((name, values))
            seen.add(key)
    return output


def summarize_sam_attempts(rows: list[dict]) -> dict[str, int]:
    """Report frame and request counts without conflating the two."""
    attempts = [
        attempt
        for row in rows
        for attempt in row.get("sam_attempts", [])
    ]
    return {
        "sam_frame_count": sum(bool(row.get("sam_called")) for row in rows),
        "sam_attempt_count": len(attempts),
        "sam_successful_attempt_count": sum(
            "sam_mask_pixels" in attempt for attempt in attempts),
        "sam_http_error_count": sum(
            "http_status" in attempt for attempt in attempts),
        "sam_other_error_count": sum(
            "error" in attempt and "http_status" not in attempt
            for attempt in attempts),
    }


def _render_panel(image: Image.Image, current: np.ndarray, proposed: np.ndarray,
                  points: list[dict], title: str,
                  *, proposal_label: str = "lidar-point SAM") -> Image.Image:
    crop = _crop(current, proposed)
    original = image.copy()
    draw = ImageDraw.Draw(original)
    for point in points:
        color = (20, 220, 120) if point["label"] == 1 else (255, 70, 70)
        x, y = point["x"], point["y"]
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color,
                     outline=(10, 20, 25), width=2)
    views = (
        _fit(original.crop(crop), (360, 250)),
        _fit(_mask_overlay(image, current, (35, 205, 130)).crop(crop), (360, 250)),
        _fit(_mask_overlay(image, proposed, (255, 165, 35)).crop(crop), (360, 250)),
        _fit(_difference_overlay(image, current, proposed).crop(crop), (360, 250)),
    )
    panel = Image.new("RGB", (1440, 312), "white")
    for x, view in zip((0, 360, 720, 1080), views):
        panel.paste(view, (x, 62))
    font = ImageFont.load_default()
    header = ImageDraw.Draw(panel)
    header.text((8, 7), title, fill="black", font=font)
    header.text((8, 30), f"points / current / {proposal_label} / difference",
                fill="black", font=font)
    header.text((1090, 43), "green kept, yellow added, red missing",
                fill="black", font=font)
    return panel


def evaluate(dataset: Path, baseline_path: Path, geometry_audit_path: Path,
             output: Path, sam_url: str, *, limit: int = 4,
             timeout: float = 120) -> dict:
    if output.exists():
        raise ValueError("output directory already exists")
    dataset = dataset.resolve()
    manifest_path = dataset / "import_manifest.json"
    inputs = [manifest_path, baseline_path, geometry_audit_path]
    before = {str(path.resolve()): _sha256(path) for path in inputs}
    state_path = Path(r"D:\Apps\LabelAgent\work\state.json")
    state_before = _sha256(state_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline = _mapping(json.loads(baseline_path.read_text(encoding="utf-8")))
    geometry = json.loads(geometry_audit_path.read_text(encoding="utf-8"))
    targets = _ranked_targets(geometry, limit)
    camera_numbers = sorted({int(_camera_name(row["topic"]).split("_")[1])
                             for row in targets})
    reference_audit = ReferenceReadAudit(dataset)
    reference_audit.start()
    cameras, calibration_paths = sensor_cameras(dataset, manifest, camera_numbers)
    health_before = request_json(sam_url.rstrip("/") + "/health", timeout=15)
    if health_before.get("status") != "ok" or health_before.get("model_loaded") is not True:
        raise ValueError("point SAM service is not ready")
    inventories = {
        camera: sorted((dataset / "source" / "cameras" / f"camera_{camera}").glob("*.png"))
        for camera in camera_numbers
    }
    output.mkdir(parents=True)
    rows = []
    panels = []
    for rank, target in enumerate(targets, start=1):
        topic, frame = str(target["topic"]), int(target["frame"])
        camera = int(_camera_name(topic).split("_")[1])
        frame_shapes = baseline[topic][str(frame)]
        matches = [shape for shape in frame_shapes
                   if str(shape.get("id")) == str(target["shape_id"])]
        if len(matches) != 1:
            raise ValueError(f"target identity is not unique: {topic} {frame}")
        current_shape = matches[0]
        current = _rasterize(current_shape, 1920, 1536).astype(np.uint8)
        pallet_union = np.zeros_like(current)
        for shape in frame_shapes:
            if str(shape.get("label", "")).casefold() == "pallet":
                pallet_union |= _rasterize(shape, 1920, 1536).astype(np.uint8)
        xyz, lidar_path = read_xyz(dataset, manifest, frame)
        plane = fit_support_plane(xyz)
        sensor = projected_sensor(xyz, cameras[camera], plane)
        prompt = select_lidar_prompt_points(
            current, pallet_union, sensor["pixels"], sensor["heights"])
        image_path = inventories[camera][frame]
        row = {
            "rank": rank,
            "topic": topic,
            "frame": str(frame),
            "shape_id": target["shape_id"],
            "image_path": str(image_path),
            "image_sha256": _sha256(image_path),
            "lidar_path": lidar_path,
            "plane": plane,
            "prompt": prompt,
            "sam_called": False,
            "safe_expansion": False,
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
        }
        proposed = current
        if prompt["ready"]:
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            attempts = []
            successful = []
            for attempt_index, (policy, points) in enumerate(
                    point_policies(prompt["points"]), start=1):
                payload = {"image": encoded, "points": points}
                started = time.monotonic()
                try:
                    result = request_json(
                        sam_url.rstrip("/") + "/segment_points", payload,
                        timeout=timeout)
                    elapsed = time.monotonic() - started
                    mask = polygon_mask(result, 1920, 1536)
                    proposal = _shape_from_result(
                        result, f"lidar-point-{camera}-{frame}-{rank}-{attempt_index}")
                    if not valid_shape({**proposal, "id": 1}, 1920, 1536):
                        raise ValueError("SAM returned invalid pallet geometry")
                    matched, metrics = match_expansion(
                        current_shape, [proposal], 1920, 1536,
                        minimum_baseline_coverage=.5,
                        minimum_area_ratio=1.25,
                        maximum_minor_axis_ratio=1.5,
                        minimum_axis_elongation=2.0,
                        other_baselines=[
                            shape for shape in frame_shapes
                            if str(shape.get("id")) != str(target["shape_id"])],
                        maximum_other_baseline_fraction=.1)
                    positive_covered = all(mask[p["y"], p["x"]] != 0
                                           for p in points if p["label"] == 1)
                    negative_clear = all(mask[p["y"], p["x"]] == 0
                                         for p in points if p["label"] == 0)
                    attempt = {
                        "policy": policy,
                        "points": points,
                        "elapsed_seconds": round(elapsed, 4),
                        "sam_score": result.get("score"),
                        "sam_warnings": result.get("warnings") or [],
                        "sam_mask_pixels": int(mask.sum()),
                        "positive_points_covered": positive_covered,
                        "negative_points_clear": negative_clear,
                        "safe_expansion": (matched is not None and positive_covered
                                           and negative_clear),
                        "expansion_metrics": _public_metrics(metrics),
                    }
                    attempts.append(attempt)
                    successful.append((
                        bool(attempt["safe_expansion"]),
                        bool(positive_covered), bool(negative_clear),
                        float(result.get("score", 0) or 0),
                        -len(result.get("warnings") or []),
                        attempt, mask,
                    ))
                except urllib.error.HTTPError as error:
                    attempts.append({
                        "policy": policy,
                        "points": points,
                        "elapsed_seconds": round(time.monotonic() - started, 4),
                        "http_status": error.code,
                        "error": error.read().decode("utf-8", "replace")[:1000],
                        "safe_expansion": False,
                    })
                except Exception as error:
                    attempts.append({
                        "policy": policy,
                        "points": points,
                        "elapsed_seconds": round(time.monotonic() - started, 4),
                        "error_type": type(error).__name__,
                        "error": str(error)[:1000],
                        "safe_expansion": False,
                    })
            row["sam_called"] = True
            row["sam_attempts"] = attempts
            if successful:
                successful.sort(key=lambda item: item[:5], reverse=True)
                selected = successful[0]
                attempt, proposed = selected[5], selected[6]
                row.update({
                    "selected_policy": attempt["policy"],
                    "sam_elapsed_seconds": round(sum(
                        value["elapsed_seconds"] for value in attempts), 4),
                    "sam_score": attempt["sam_score"],
                    "sam_warnings": attempt["sam_warnings"],
                    "sam_mask_pixels": attempt["sam_mask_pixels"],
                    "positive_points_covered": attempt["positive_points_covered"],
                    "negative_points_clear": attempt["negative_points_clear"],
                    "safe_expansion": attempt["safe_expansion"],
                    "expansion_metrics": attempt["expansion_metrics"],
                })
            else:
                row["sam_all_attempts_failed"] = True
        image = Image.open(image_path).convert("RGB")
        panels.append(_render_panel(
            image, current.astype(bool), proposed.astype(bool),
            prompt.get("points") or [],
            f"#{rank} camera_{camera} frame {frame} ready={prompt['ready']} "
            f"safe_expansion={row['safe_expansion']}"))
        rows.append(row)
        (output / f"camera_{camera}-frame-{frame}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "rank": rank, "camera": camera, "frame": frame,
            "prompt_ready": prompt["ready"], "sam_called": row["sam_called"],
            "safe_expansion": row["safe_expansion"],
        }, ensure_ascii=False), flush=True)
    reference_audit.stop()
    sheet = Image.new("RGB", (1440, sum(panel.height for panel in panels)), "white")
    top = 0
    for panel in panels:
        sheet.paste(panel, (0, top))
        top += panel.height
    sheet_path = output / "pallet-lidar-point-review.jpg"
    sheet.save(sheet_path, quality=90, optimize=True)
    health_after = request_json(sam_url.rstrip("/") + "/health", timeout=15)
    if any(_sha256(Path(path)) != digest for path, digest in before.items()):
        raise RuntimeError("an input changed during LiDAR point-prompt evaluation")
    attempt_summary = summarize_sam_attempts(rows)
    report = {
        "schema_version": "labelagent.pallet_lidar_point_prompt.v2",
        "status": "candidate_only_not_deployed",
        "input_sha256": before,
        "calibration_paths": calibration_paths,
        "selected_count": len(rows),
        "prompt_ready_count": sum(row["prompt"]["ready"] for row in rows),
        **attempt_summary,
        "safe_expansion_count": sum(row["safe_expansion"] for row in rows),
        "health_before": health_before,
        "health_after": health_after,
        "reference_label_paths_opened": reference_audit.denied,
        "reference_labels_used": False,
        "workbench_writes": 0,
        "production_state_before_sha256": state_before,
        "production_state_after_sha256": _sha256(state_path),
        "contact_sheet": str(sheet_path.resolve()),
        "contact_sheet_sha256": _sha256(sheet_path),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "production_deployed": False,
        "rows": rows,
    }
    report_path = output / "results.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path.resolve())
    report["report_sha256"] = _sha256(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--geometry-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sam-url", default="http://127.0.0.1:28766")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    report = evaluate(
        args.dataset, args.baseline, args.geometry_audit, args.output,
        args.sam_url, limit=args.limit, timeout=args.timeout)
    print(json.dumps({key: report[key] for key in (
        "selected_count", "prompt_ready_count", "sam_frame_count",
        "sam_attempt_count", "sam_successful_attempt_count",
        "sam_http_error_count", "sam_other_error_count",
        "safe_expansion_count", "reference_labels_used", "workbench_writes",
        "production_state_before_sha256", "production_state_after_sha256",
        "contact_sheet", "contact_sheet_sha256", "report_path", "report_sha256",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
