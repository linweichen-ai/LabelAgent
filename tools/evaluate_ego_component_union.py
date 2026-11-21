#!/usr/bin/env python3
"""Read-only A/B for a disconnected camera1 ego candidate.

The baseline sends every positive point in one SAM request.  The candidate
sends one positive point plus the same fixed negative points per request and
unions only prompt-valid component masks.  This is useful when a person or a
vehicle structure visually disconnects the visible ego body.

The output is local, candidate-only evidence.  It never writes workbench
state, human truth, training approval, or a production prompt.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from source.agent.edit_agent import _rasterize, valid_shape
from source.agent.ego_shadow import materialize_camera_template
from source.agent.workflow import _sam_request, candidate_overlay


DEVELOPMENT_FRAMES = (0, 38, 76, 114, 151)
HELD_OUT_FRAMES = (19, 57, 95, 133)
MAXIMUM_AREA_RATIO = 0.25


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_image(root: Path, frame: int) -> Path:
    matches = [path for path in root.glob(f"camera1-frame{frame}.*")
               if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one camera1 image for frame {frame}")
    return matches[0]


def _decode_response(response: dict, width: int, height: int) -> tuple[dict, np.ndarray, list[str]]:
    if not isinstance(response, dict):
        raise ValueError("point response must be an object")
    parts, score = response.get("parts"), response.get("score")
    warnings = response.get("warnings") or []
    if (not isinstance(parts, list) or not parts
            or isinstance(score, bool) or not isinstance(score, (int, float))
            or not np.isfinite(score) or not 0 <= score <= 1
            or not isinstance(warnings, list)
            or any(not isinstance(item, str) for item in warnings)):
        raise ValueError("point response is incomplete")
    shape = {"id": 1, "label": "ego", "parts": parts, "score": float(score)}
    if not valid_shape(shape, width, height):
        raise ValueError("point response geometry is invalid")
    mask = _rasterize(shape, width, height).astype(bool)
    if not mask.any():
        raise ValueError("point response mask is empty")
    return shape, mask, warnings


def _prompt_metrics(mask: np.ndarray, positives: list[dict], negatives: list[dict]) -> dict:
    positive_covered = sum(bool(mask[p["y"], p["x"]]) for p in positives)
    negative_violated = sum(bool(mask[p["y"], p["x"]]) for p in negatives)
    return {
        "positive_point_count": len(positives),
        "positive_points_covered": positive_covered,
        "negative_point_count": len(negatives),
        "negative_points_violated": negative_violated,
        "prompt_satisfied": (
            positive_covered == len(positives) and negative_violated == 0),
    }


def _geometry(mask: np.ndarray) -> dict:
    rows, columns = mask.nonzero()
    return {
        "area_pixels": int(mask.sum()),
        "area_ratio": round(float(mask.mean()), 6),
        "bbox": [int(columns.min()), int(rows.min()),
                 int(columns.max()) + 1, int(rows.max()) + 1],
        "touches_bottom": int(rows.max()) + 1 >= mask.shape[0] - 1,
    }


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum()) / union if union else 1.0


def evaluate_image(image: bytes, request_points) -> tuple[dict, dict, dict, dict, list[dict]]:
    with Image.open(io.BytesIO(image)) as decoded:
        width, height = decoded.size
    _, points = materialize_camera_template("camera1", width, height)
    positives = [point for point in points if point["label"] == 1]
    negatives = [point for point in points if point["label"] == 0]

    baseline_shape, baseline_mask, baseline_warnings = _decode_response(
        request_points(image, points), width, height)
    baseline = (_prompt_metrics(baseline_mask, positives, negatives)
                | _geometry(baseline_mask)
                | {"score": round(float(baseline_shape["score"]), 6),
                   "sam_warning_count": len(baseline_warnings)})
    baseline["candidate_available"] = (
        baseline["prompt_satisfied"] and not baseline_warnings
        and baseline["area_ratio"] <= MAXIMUM_AREA_RATIO)

    component_masks, component_parts, component_rows = [], [], []
    for index, positive in enumerate(positives):
        component_points = [positive, *negatives]
        try:
            shape, mask, warnings = _decode_response(
                request_points(image, component_points), width, height)
            metrics = (_prompt_metrics(mask, [positive], negatives)
                       | _geometry(mask)
                       | {"component_index": index,
                          "score": round(float(shape["score"]), 6),
                          "sam_warning_count": len(warnings)})
            metrics["component_available"] = (
                metrics["prompt_satisfied"] and not warnings
                and metrics["area_ratio"] <= MAXIMUM_AREA_RATIO)
            component_rows.append(metrics)
            if metrics["component_available"]:
                component_masks.append(mask)
                component_parts.extend(shape["parts"])
        except Exception as error:
            component_rows.append({
                "component_index": index,
                "component_available": False,
                "error": str(error)[:500],
            })

    union_mask = (np.logical_or.reduce(component_masks)
                  if component_masks else np.zeros((height, width), dtype=bool))
    if union_mask.any():
        union = (_prompt_metrics(union_mask, positives, negatives)
                 | _geometry(union_mask))
    else:
        union = {
            "area_pixels": 0,
            "area_ratio": 0.0,
            "bbox": None,
            "touches_bottom": False,
        } | _prompt_metrics(union_mask, positives, negatives)
    union["component_count"] = len(component_masks)
    union["candidate_available"] = (
        len(component_masks) == len(positives)
        and union["prompt_satisfied"]
        and union["area_ratio"] <= MAXIMUM_AREA_RATIO)
    union["iou_vs_baseline"] = round(_iou(union_mask, baseline_mask), 6)
    union["area_ratio_delta_vs_baseline"] = round(
        union["area_ratio"] - baseline["area_ratio"], 6)
    union_shape = {
        "id": 2,
        "label": "ego",
        "parts": component_parts,
        "score": min((row["score"] for row in component_rows
                      if row.get("component_available")), default=0.0),
    }
    return baseline, union, baseline_shape, union_shape, component_rows


def _contact_sheet(output: Path, rows: list[dict]) -> Path:
    cell_width, image_height, label_height = 480, 384, 30
    canvas = Image.new("RGB", (cell_width * 2,
                               (image_height + label_height) * len(rows)),
                       (15, 23, 31))
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    for row_index, row in enumerate(rows):
        for column, key in enumerate(("baseline_overlay", "union_overlay")):
            with Image.open(output / row[key]) as image:
                preview = image.convert("RGB")
                preview.thumbnail((cell_width, image_height))
            left = column * cell_width
            top = row_index * (image_height + label_height)
            canvas.paste(preview, (left + (cell_width - preview.width) // 2,
                                   top + (image_height - preview.height) // 2))
            result = row["baseline" if column == 0 else "component_union"]
            label = (f"{row['sample_id']} | {'baseline' if column == 0 else 'union'} "
                     f"available={result['candidate_available']} area={result['area_ratio']:.3f}")
            draw.rectangle((left, top + image_height,
                            left + cell_width, top + image_height + label_height),
                           fill=(25, 35, 45))
            draw.text((left + 6, top + image_height + 8), label,
                      fill=(83, 210, 161) if result["candidate_available"]
                      else (244, 130, 90), font=font)
    path = output / "contact_sheet.jpg"
    canvas.save(path, "JPEG", quality=88)
    return path


def evaluate_scope(image_root: Path, output: Path, request_points,
                   *, development_frames=DEVELOPMENT_FRAMES,
                   held_out_frames=HELD_OUT_FRAMES) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    if set(development_frames) & set(held_out_frames):
        raise ValueError("development and held-out frames must be disjoint")
    output.mkdir(parents=True)
    rows = []
    for split, frames in (("development", development_frames),
                          ("held_out", held_out_frames)):
        for frame in frames:
            source = _find_image(image_root, frame)
            image = source.read_bytes()
            baseline, union, baseline_shape, union_shape, components = evaluate_image(
                image, request_points)
            sample_id = f"camera1-frame{frame:03d}"
            baseline_path = output / f"{sample_id}-baseline.jpg"
            union_path = output / f"{sample_id}-component-union.jpg"
            baseline_path.write_bytes(candidate_overlay(image, [baseline_shape]))
            union_path.write_bytes(candidate_overlay(
                image, [union_shape] if union_shape["parts"] else []))
            rows.append({
                "sample_id": sample_id,
                "frame": frame,
                "split": split,
                "source_image": str(source.resolve()),
                "source_image_sha256": hashlib.sha256(image).hexdigest(),
                "baseline": baseline,
                "component_union": union,
                "components": components,
                "baseline_overlay": baseline_path.name,
                "baseline_overlay_sha256": _sha256(baseline_path),
                "union_overlay": union_path.name,
                "union_overlay_sha256": _sha256(union_path),
            })
    sheet = _contact_sheet(output, rows)

    def summarize(split: str) -> dict:
        selected = [row for row in rows if row["split"] == split]
        return {
            "sample_count": len(selected),
            "baseline_available_count": sum(
                row["baseline"]["candidate_available"] for row in selected),
            "component_union_available_count": sum(
                row["component_union"]["candidate_available"] for row in selected),
            "mean_baseline_area_ratio": round(float(np.mean([
                row["baseline"]["area_ratio"] for row in selected])), 6),
            "mean_component_union_area_ratio": round(float(np.mean([
                row["component_union"]["area_ratio"] for row in selected])), 6),
            "negative_violation_count": sum(
                row["component_union"]["negative_points_violated"] for row in selected),
        }

    report = {
        "schema_version": "labelagent.ego_component_union_ab.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "human_truth": False,
        "human_confirmed": False,
        "approved_for_training": False,
        "production_deployed": False,
        "formal_state_modified": False,
        "camera": "camera1",
        "strategy": "one_positive_plus_fixed_negatives_per_request_then_union",
        "color_dependent": False,
        "maximum_area_ratio": MAXIMUM_AREA_RATIO,
        "development_frames": list(development_frames),
        "held_out_frames": list(held_out_frames),
        "development": summarize("development"),
        "held_out": summarize("held_out"),
        "contact_sheet": sheet.name,
        "contact_sheet_sha256": _sha256(sheet),
        "samples": rows,
        "decision_boundary": (
            "Prompt-valid component unions remain assistant candidates. Human pixel truth, "
            "person/floor overlap gates, and an independent receipt are required before any "
            "training or production change."),
    }
    result = output / "results.json"
    result.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-sam-url", default="http://127.0.0.1:28766")
    arguments = parser.parse_args()
    if not arguments.point_sam_url.startswith("http://127.0.0.1:"):
        raise ValueError("point SAM URL must be loopback")
    config = {"sam3_url": arguments.point_sam_url.rstrip("/")}
    report = evaluate_scope(
        arguments.image_root, arguments.output,
        lambda image, points: _sam_request(
            config, "/segment_points", image, timeout_seconds=8.0, points=points))
    print(json.dumps({
        "output": str((arguments.output / "results.json").resolve()),
        "development": report["development"],
        "held_out": report["held_out"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
