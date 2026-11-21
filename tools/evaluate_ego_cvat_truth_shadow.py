#!/usr/bin/env python3
"""Compare the frozen ego prompt family with the local CVAT Task3 snapshot.

The CVAT annotations are read only.  They are a distinct, locally verified
snapshot but do not carry the v3 per-sample human receipt required by the
formal ego acceptance gate, so this evaluator reports diagnostic evidence
without approving training or production.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from source.agent.workflow import _sam_request
from tools.evaluate_ego_fixed_family import (
    MINIMUM_BASELINE_RETENTION,
    derive_minimum_family,
    evaluate_fixed_family_image,
)
from tools.evaluate_ego_unexposed_timepoints import select_even_unexposed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_label_mask(shapes: list[dict], label: str,
                      size: tuple[int, int]) -> np.ndarray:
    if (not isinstance(shapes, list) or not isinstance(label, str)
            or not label or len(size) != 2 or min(size) <= 0):
        raise ValueError("mask inputs are invalid")
    mask = Image.new("L", size, 0)
    for shape in shapes:
        if not isinstance(shape, dict) or shape.get("label") != label:
            continue
        parts = shape.get("parts")
        if parts == [] and not shape.get("points"):
            continue
        if not isinstance(parts, list) or not parts:
            parts = [{"points": shape.get("points"),
                      "holes": shape.get("holes", [])}]
        for part in parts:
            if not isinstance(part, dict):
                raise ValueError("mask part is invalid")
            points, holes = part.get("points"), part.get("holes", [])
            if (not isinstance(points, list) or len(points) < 3
                    or any(not isinstance(point, list) or len(point) != 2
                           for point in points)
                    or not isinstance(holes, list)):
                raise ValueError("mask geometry is invalid")
            component = Image.new("L", size, 0)
            draw = ImageDraw.Draw(component)
            draw.polygon([tuple(point) for point in points], fill=255)
            for hole in holes:
                if (not isinstance(hole, list) or len(hole) < 3
                        or any(not isinstance(point, list) or len(point) != 2
                               for point in hole)):
                    raise ValueError("mask hole is invalid")
                draw.polygon([tuple(point) for point in hole], fill=0)
            mask.paste(255, mask=component)
    return np.asarray(mask, dtype=np.uint8) > 0


def mask_metrics(candidate: np.ndarray, truth: np.ndarray,
                 person: np.ndarray, floor: np.ndarray) -> dict:
    shapes = {mask.shape for mask in (candidate, truth, person, floor)}
    if len(shapes) != 1 or any(mask.dtype != np.bool_ for mask in
                               (candidate, truth, person, floor)):
        raise ValueError("metric masks are incompatible")
    candidate_area, truth_area = int(candidate.sum()), int(truth.sum())
    if not truth_area:
        raise ValueError("ego truth is empty")
    intersection = int(np.logical_and(candidate, truth).sum())
    union = int(np.logical_or(candidate, truth).sum())

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator) / denominator if denominator else 0.0

    return {
        "candidate_area_pixels": candidate_area,
        "truth_area_pixels": truth_area,
        "iou": round(ratio(intersection, union), 6),
        "precision": round(ratio(intersection, candidate_area), 6),
        "recall": round(ratio(intersection, truth_area), 6),
        "person_overlap": round(ratio(
            int(np.logical_and(candidate, person).sum()), candidate_area), 6),
        "floor_overlap": round(ratio(
            int(np.logical_and(candidate, floor).sum()), candidate_area), 6),
    }


def _overlay(image_path: Path, truth: np.ndarray, candidate: np.ndarray,
             output: Path) -> None:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    base = np.asarray(image, dtype=np.uint8).copy()
    tint = base.astype(np.float32)
    tint[truth] = tint[truth] * 0.65 + np.array([20, 220, 120]) * 0.35
    tint[candidate] = tint[candidate] * 0.55 + np.array([255, 165, 35]) * 0.45
    output_image = Image.fromarray(np.clip(tint, 0, 255).astype(np.uint8))
    output_image.save(output, "JPEG", quality=90)


def _contact_sheet(output: Path, rows: list[dict], columns: int = 4) -> Path:
    cell_width, image_height, label_height = 320, 240, 32
    row_count = (len(rows) + columns - 1) // columns
    canvas = Image.new("RGB", (cell_width * columns,
                               (image_height + label_height) * row_count),
                       (15, 23, 31))
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    for item_index, row in enumerate(rows):
        column, item_row = item_index % columns, item_index // columns
        with Image.open(output / row["comparison_overlay"]) as opened:
            preview = opened.convert("RGB")
            preview.thumbnail((cell_width, image_height))
        left, top = column * cell_width, item_row * (image_height + label_height)
        canvas.paste(preview, (left + (cell_width - preview.width) // 2,
                               top + (image_height - preview.height) // 2))
        metrics = row["fixed_geometry_metrics"]
        label = (f"f{row['frame']:03d} gate={row['fixed_family']['candidate_available']} "
                 f"IoU={metrics['iou']:.2f} P={metrics['person_overlap']:.2f} "
                 f"F={metrics['floor_overlap']:.2f}")
        draw.rectangle((left, top + image_height, left + cell_width,
                        top + image_height + label_height), fill=(25, 35, 45))
        draw.text((left + 5, top + image_height + 8), label,
                  fill=(83, 210, 161) if row["fixed_family"]["candidate_available"]
                  else (244, 130, 90), font=font)
    path = output / "contact_sheet.jpg"
    canvas.save(path, "JPEG", quality=88)
    return path


def _summarize(rows: list[dict], key: str) -> dict:
    metrics = [row[key] for row in rows]
    return {
        "mean_iou": round(float(np.mean([item["iou"] for item in metrics])), 6),
        "median_iou": round(float(np.median([item["iou"] for item in metrics])), 6),
        "minimum_iou": round(float(min(item["iou"] for item in metrics)), 6),
        "mean_precision": round(float(np.mean([
            item["precision"] for item in metrics])), 6),
        "mean_recall": round(float(np.mean([
            item["recall"] for item in metrics])), 6),
        "maximum_person_overlap": round(float(max(
            item["person_overlap"] for item in metrics)), 6),
        "maximum_floor_overlap": round(float(max(
            item["floor_overlap"] for item in metrics)), 6),
        "sample_iou_at_least_0_75": sum(item["iou"] >= 0.75 for item in metrics),
    }


def failed_evaluation(error: Exception) -> tuple[dict, dict, dict, dict, list]:
    message = str(error)[:500]
    baseline = {"request_failed": True, "error": message}
    fixed = {
        "request_failed": True,
        "error": message,
        "prompt_satisfied": False,
        "negative_points_violated": 0,
        "baseline_retention_pass": False,
        "baseline_retention": 0.0,
        "additional_request_count": 0,
        "additional_elapsed_seconds": 0.0,
        "candidate_available": False,
    }
    empty_shape = {"id": 1, "label": "ego", "parts": [], "score": 0.0}
    return baseline, fixed, empty_shape, empty_shape.copy(), []


def evaluate_scope(discovery_path: Path, image_directory: Path,
                   annotation_path: Path, import_manifest_path: Path,
                   output: Path, request_points, *, sample_count: int = 24) -> dict:
    if output.exists():
        raise FileExistsError(output)
    discovery_bytes = discovery_path.read_bytes()
    family = derive_minimum_family(json.loads(discovery_bytes))
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    manifest = json.loads(import_manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("dataset") != "CVAT Task3"
            or manifest.get("frame_count") != 150
            or manifest.get("matches_reviewed_snapshot_content") is not True
            or manifest.get("original_image_hashes_verified") is not True
            or not isinstance(manifest.get("frames"), list)
            or len(manifest["frames"]) != manifest["frame_count"]
            or not isinstance(annotations, dict)
            or len(annotations) != manifest["frame_count"]):
        raise ValueError("CVAT snapshot contract is invalid")
    selected = select_even_unexposed(manifest["frame_count"], (), sample_count)
    output.mkdir(parents=True)
    rows = []
    for frame in selected:
        frame_meta = manifest["frames"][frame]
        if frame_meta.get("frame") != frame:
            raise ValueError("CVAT frame order is invalid")
        source = image_directory / frame_meta["file"]
        image_bytes = source.read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != frame_meta["sha256"]:
            raise ValueError(f"source image hash mismatch for frame {frame}")
        with Image.open(source) as opened:
            size = opened.size
        if list(size) != manifest["image_size"]:
            raise ValueError(f"source image dimensions mismatch for frame {frame}")
        shapes = annotations.get(str(frame))
        if not isinstance(shapes, list):
            raise ValueError(f"annotations missing for frame {frame}")
        ego = render_label_mask(shapes, "ego", size)
        person = render_label_mask(shapes, "person", size)
        floor = render_label_mask(shapes, "floor", size)
        try:
            baseline, fixed, baseline_shape, fixed_shape, attempts = (
                evaluate_fixed_family_image(image_bytes, family, request_points))
        except Exception as error:
            baseline, fixed, baseline_shape, fixed_shape, attempts = (
                failed_evaluation(error))
        baseline_mask = render_label_mask([baseline_shape | {"label": "ego"}],
                                          "ego", size)
        fixed_mask = render_label_mask([fixed_shape], "ego", size)
        accepted_mask = (fixed_mask if fixed["candidate_available"]
                         else np.zeros_like(fixed_mask))
        comparison = output / f"cvat-frame{frame:03d}-truth-fixed.jpg"
        _overlay(source, ego, fixed_mask, comparison)
        rows.append({
            "sample_id": f"cvat-task3-frame{frame:03d}",
            "frame": frame,
            "source_image": str(source.resolve()),
            "source_image_sha256": frame_meta["sha256"],
            "ego_annotation_count": sum(
                shape.get("label") == "ego" for shape in shapes),
            "baseline": baseline,
            "fixed_family": fixed,
            "attempts": attempts,
            "baseline_metrics": mask_metrics(
                baseline_mask, ego, person, floor),
            "fixed_geometry_metrics": mask_metrics(
                fixed_mask, ego, person, floor),
            "accepted_candidate_metrics": mask_metrics(
                accepted_mask, ego, person, floor),
            "comparison_overlay": comparison.name,
            "comparison_overlay_sha256": _sha256(comparison),
        })
    sheet = _contact_sheet(output, rows)
    report = {
        "schema_version": "labelagent.ego_cvat_truth_shadow.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "CVAT Task3 local reviewed snapshot",
        "task_id": manifest.get("task_id"),
        "job_id": manifest.get("job_id"),
        "distinct_dataset_identity": True,
        "cross_factory_generalization": False,
        "remote_cvat_current_state_checked": manifest.get(
            "remote_cvat_current_state_checked"),
        "matches_reviewed_snapshot_content": True,
        "cvat_source_counts": manifest.get("source_counts"),
        "formal_human_receipt_v3": False,
        "diagnostic_only": True,
        "candidate_only": True,
        "approved_for_training": False,
        "production_deployed": False,
        "formal_state_modified": False,
        "selected_frames": list(selected),
        "sample_count": len(rows),
        "fixed_family": [list(subset) for subset in family],
        "minimum_baseline_retention": MINIMUM_BASELINE_RETENTION,
        "candidate_available_count": sum(
            row["fixed_family"]["candidate_available"] for row in rows),
        "request_failure_count": sum(
            row["fixed_family"].get("request_failed") is True for row in rows),
        "negative_violation_count": sum(
            row["fixed_family"]["negative_points_violated"] for row in rows),
        "baseline": _summarize(rows, "baseline_metrics"),
        "fixed_geometry": _summarize(rows, "fixed_geometry_metrics"),
        "accepted_candidate": _summarize(rows, "accepted_candidate_metrics"),
        "fixed_geometry_iou_improved_count": sum(
            row["fixed_geometry_metrics"]["iou"]
            > row["baseline_metrics"]["iou"] for row in rows),
        "median_additional_elapsed_seconds": round(float(np.median([
            row["fixed_family"]["additional_elapsed_seconds"]
            for row in rows])), 6),
        "discovery_report_sha256": hashlib.sha256(discovery_bytes).hexdigest(),
        "annotation_snapshot_sha256": _sha256(annotation_path),
        "import_manifest_sha256": _sha256(import_manifest_path),
        "contact_sheet": sheet.name,
        "contact_sheet_sha256": _sha256(sheet),
        "samples": rows,
        "decision_boundary": (
            "This is an independent local dataset identity and pixel diagnostic, "
            "but not the hash-signed v3 human acceptance receipt. It cannot approve "
            "training, production, or cross-factory generalization."),
    }
    result = output / "results.json"
    result.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--image-directory", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--import-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--point-sam-url", default="http://127.0.0.1:28766")
    arguments = parser.parse_args()
    if not arguments.point_sam_url.startswith("http://127.0.0.1:"):
        raise ValueError("point SAM URL must be loopback")
    config = {"sam3_url": arguments.point_sam_url.rstrip("/")}
    report = evaluate_scope(
        arguments.discovery, arguments.image_directory, arguments.annotations,
        arguments.import_manifest, arguments.output,
        lambda image, points: _sam_request(
            config, "/segment_points", image, timeout_seconds=8.0, points=points),
        sample_count=arguments.sample_count)
    print(json.dumps({
        "output": str((arguments.output / "results.json").resolve()),
        "sample_count": report["sample_count"],
        "candidate_available_count": report["candidate_available_count"],
        "negative_violation_count": report["negative_violation_count"],
        "baseline": report["baseline"],
        "fixed_geometry": report["fixed_geometry"],
        "accepted_candidate": report["accepted_candidate"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
