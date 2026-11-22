#!/usr/bin/env python3
"""Evaluate LiDAR-localized pallet ROIs with text-SAM, without writeback."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.agent.edit_agent import _rasterize
from tools.evaluate_pallet_lidar_point_prompts import _render_panel
from tools.evaluate_slope_lidar_sam import polygon_mask, request_json
from tools.review_distant_pallet_recovery import _mapping


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _crop_box(roi, width, height, minimum_side=256, padding=24):
    x0, y0, x1, y1 = (int(value) for value in roi)
    x0, y0 = max(0, x0 - padding), max(0, y0 - padding)
    x1, y1 = min(width, x1 + padding), min(height, y1 + padding)
    target_width, target_height = max(minimum_side, x1 - x0), max(minimum_side, y1 - y0)
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = max(0, min(width - target_width, int(round(center_x - target_width / 2))))
    y0 = max(0, min(height - target_height, int(round(center_y - target_height / 2))))
    return [x0, y0, min(width, x0 + target_width), min(height, y0 + target_height)]


def _result_mask(result, width, height):
    return polygon_mask(result, width, height).astype(bool)


def _distance_to_mask(mask, x, y):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return float(np.sqrt(np.min((xs - x) ** 2 + (ys - y) ** 2)))


def select_associated_result(results, current, center_x, center_y, width, height):
    """Associate text masks to the incumbent instance and frozen 3D center."""
    ranked = []
    current_area = int(current.sum())
    summaries = []
    for index, result in enumerate(results):
        mask = _result_mask(result, width, height)
        contains = bool(0 <= center_x < width and 0 <= center_y < height
                        and mask[int(center_y), int(center_x)])
        coverage = float((mask & current).sum() / current_area) if current_area else 0.0
        distance = _distance_to_mask(mask, center_x, center_y)
        anchored = coverage >= .25
        near_center = distance is not None and distance <= 24
        score = float(result.get("score", 0) or 0)
        summaries.append({
            "index": index, "score": score, "contains_component_center": contains,
            "component_center_distance_pixels": distance,
            "current_coverage": coverage, "anchored_to_current": anchored,
            "prompt_name": result.get("prompt_name"),
        })
        ranked.append((anchored, near_center, coverage, score, -index,
                       result, mask, contains, distance))
    if not ranked:
        return None, None, False, None, summaries
    ranked.sort(key=lambda item: item[:5], reverse=True)
    selected = ranked[0]
    return selected[5], selected[6], selected[7], selected[8], summaries


def _mask_metrics(proposal, current, other_masks, points):
    proposal_area, current_area = int(proposal.sum()), int(current.sum())
    intersection = int((proposal & current).sum())
    union = int((proposal | current).sum())
    positive = [point for point in points if int(point["label"]) == 1]
    negative = [point for point in points if int(point["label"]) == 0]
    positive_covered = sum(bool(proposal[int(p["y"]), int(p["x"])]) for p in positive)
    negative_clear = sum(not bool(proposal[int(p["y"]), int(p["x"])]) for p in negative)
    other_fractions = [float((proposal & other).sum() / proposal_area)
                       for other in other_masks] if proposal_area else []
    return {
        "proposal_pixels": proposal_area,
        "current_pixels": current_area,
        "area_ratio": proposal_area / current_area if current_area else None,
        "current_coverage": intersection / current_area if current_area else None,
        "proposal_current_iou": intersection / union if union else 1.0,
        "positive_points": len(positive),
        "positive_points_covered": positive_covered,
        "positive_coverage": positive_covered / len(positive) if positive else None,
        "negative_points": len(negative),
        "negative_points_clear": negative_clear,
        "negative_clear_fraction": negative_clear / len(negative) if negative else None,
        "maximum_other_candidate_fraction": max(other_fractions, default=0.0),
    }


def evaluate(audit_path: Path, baseline_path: Path, output: Path,
             sam_url: str, *, timeout: float = 120) -> dict:
    if output.exists():
        raise ValueError("output directory already exists")
    audit_path, baseline_path = audit_path.resolve(), baseline_path.resolve()
    inputs = {str(path): _sha256(path) for path in (audit_path, baseline_path)}
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit.get("schema_version") != "labelagent.pallet_lidar_instance_prompt_audit.v1"
            or audit.get("sam_calls") != 0 or audit.get("human_truth") is not False):
        raise ValueError("requires the frozen candidate-only LiDAR prompt audit")
    baseline = _mapping(json.loads(baseline_path.read_text(encoding="utf-8")))
    state_path = Path(r"D:\Apps\LabelAgent\work\state.json")
    state_before = _sha256(state_path)
    health_before = request_json(sam_url.rstrip("/") + "/health", timeout=15)
    if health_before.get("status") != "ok" or health_before.get("model_loaded") is not True:
        raise ValueError("SAM service is not ready")
    output.mkdir(parents=True)
    rows, panels = [], []
    for source in audit["rows"]:
        topic, frame = str(source["topic"]), str(source["frame"])
        prompt = source["instance_prompt"]
        row = {
            "topic": topic, "frame": frame, "shape_id": source["shape_id"],
            "prompt_ready": bool(prompt["ready"]), "sam_called": False,
            "diagnostic_pass": False, "candidate_only": True, "human_truth": False,
            "approved_for_training": False,
        }
        image_path = Path(source["image_path"])
        if _sha256(image_path) != source["image_sha256"]:
            raise ValueError(f"source image changed: {image_path}")
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        shapes = baseline[topic][frame]
        matches = [shape for shape in shapes if str(shape.get("id")) == str(source["shape_id"])]
        if len(matches) != 1:
            raise ValueError(f"frozen candidate identity changed: {topic} {frame}")
        current_shape = matches[0]
        current = _rasterize(current_shape, width, height).astype(bool)
        proposal = current.copy()
        points = prompt.get("points") or []
        if prompt["ready"]:
            crop_box = _crop_box(prompt["roi_bbox"], width, height)
            crop = image.crop(tuple(crop_box))
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG")
            started = time.monotonic()
            response = request_json(
                sam_url.rstrip("/") + "/segment_text",
                {"image": base64.b64encode(buffer.getvalue()).decode("ascii"),
                 "prompt": "pallet"}, timeout=timeout)
            elapsed = time.monotonic() - started
            component = prompt["component_pixel_bbox"]
            center_x = (component[0] + component[2]) / 2 - crop_box[0]
            center_y = (component[1] + component[3]) / 2 - crop_box[1]
            current_crop = current[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
            selected, local_mask, contains_center, center_distance, candidate_summaries = (
                select_associated_result(response.get("results") or [], current_crop,
                                         center_x, center_y, crop.width, crop.height))
            row.update({
                "sam_called": True, "elapsed_seconds": round(elapsed, 4),
                "crop_box": crop_box, "crop_size": [crop.width, crop.height],
                "model_result_count": len(response.get("results") or []),
                "component_center_local": [center_x, center_y],
                "selected_contains_component_center": contains_center,
                "selected_component_center_distance_pixels": center_distance,
                "candidate_summaries": candidate_summaries,
                "diagnostics": response.get("diagnostics") or {},
            })
            if selected is not None:
                proposal = np.zeros_like(current)
                proposal[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]] = local_mask
                other_masks = [_rasterize(shape, width, height).astype(bool) for shape in shapes
                               if str(shape.get("id")) != str(source["shape_id"])
                               and str(shape.get("label")) == "pallet"]
                metrics = _mask_metrics(proposal, current, other_masks, points)
                row.update({"sam_score": selected.get("score"), "metrics": metrics})
                row["diagnostic_pass"] = bool(
                    center_distance is not None and center_distance <= 24
                    and metrics["current_coverage"] is not None
                    and metrics["current_coverage"] >= .5
                    and metrics["negative_clear_fraction"] is not None
                    and metrics["negative_clear_fraction"] >= .75
                    and metrics["maximum_other_candidate_fraction"] <= .1
                    and metrics["area_ratio"] is not None
                    and .5 <= metrics["area_ratio"] <= 3.0)
            else:
                row["empty_reason"] = (response.get("diagnostics") or {}).get("empty_reason")
        panels.append(_render_panel(
            image, current, proposal,
            [{"x": int(p["x"]), "y": int(p["y"]), "label": int(p["label"])}
             for p in points],
            f"{topic} frame {frame} center={row.get('selected_contains_component_center')} "
            f"pass={row['diagnostic_pass']}",
            proposal_label="LiDAR-localized ROI text-SAM",
        ))
        rows.append(row)
        (output / f"{topic.split('/')[2]}-frame-{frame}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sheet = Image.new("RGB", (1440, sum(panel.height for panel in panels)), "white")
    top = 0
    for panel in panels:
        sheet.paste(panel, (0, top)); top += panel.height
    sheet_path = output / "lidar-centered-roi-sam-review.jpg"
    sheet.save(sheet_path, quality=90, optimize=True)
    health_after = request_json(sam_url.rstrip("/") + "/health", timeout=15)
    if any(_sha256(Path(path)) != digest for path, digest in inputs.items()):
        raise RuntimeError("an input changed during evaluation")
    report = {
        "schema_version": "labelagent.pallet_lidar_centered_roi_sam.v1",
        "status": "offline_candidate_not_deployed", "input_sha256": inputs,
        "selected_count": len(rows),
        "prompt_ready_count": sum(row["prompt_ready"] for row in rows),
        "sam_frame_count": sum(row["sam_called"] for row in rows),
        "center_associated_count": sum(bool(row.get("selected_contains_component_center"))
                                       for row in rows),
        "diagnostic_pass_count": sum(row["diagnostic_pass"] for row in rows),
        "health_before": health_before, "health_after": health_after,
        "reference_labels_used": False, "workbench_writes": 0,
        "production_state_before_sha256": state_before,
        "production_state_after_sha256": _sha256(state_path),
        "contact_sheet": str(sheet_path.resolve()),
        "contact_sheet_sha256": _sha256(sheet_path),
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "production_deployed": False,
        "rows": rows,
    }
    report_path = output / "results.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    report["report_path"] = str(report_path.resolve())
    report["report_sha256"] = _sha256(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sam-url", default="http://127.0.0.1:28766")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    report = evaluate(args.audit, args.baseline, args.output, args.sam_url,
                      timeout=args.timeout)
    keys = ("selected_count", "prompt_ready_count", "sam_frame_count",
            "center_associated_count", "diagnostic_pass_count", "workbench_writes",
            "production_state_before_sha256", "production_state_after_sha256",
            "contact_sheet", "contact_sheet_sha256", "report_path", "report_sha256")
    print(json.dumps({key: report[key] for key in keys}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
