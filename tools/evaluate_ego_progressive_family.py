#!/usr/bin/env python3
"""Evaluate a three-request camera1 family with bounded progressive fallback.

The fixed family is derived from development-only evidence.  When its union
fails prompt coverage or the 90% baseline-retention guard, remaining subsets
are tried in a fixed, data-independent order (larger subsets first) and stop as
soon as the guard passes.  This is local candidate evidence only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from source.agent.ego_shadow import materialize_camera_template
from source.agent.workflow import _sam_request, candidate_overlay
from tools.evaluate_ego_component_union import (
    MAXIMUM_AREA_RATIO,
    _decode_response,
    _find_image,
    _geometry,
    _prompt_metrics,
    _sha256,
)
from tools.evaluate_ego_fixed_family import (
    MINIMUM_BASELINE_RETENTION,
    derive_minimum_family,
)


def fallback_order(positive_count: int,
                   family: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    if type(positive_count) is not int or positive_count <= 1:
        raise ValueError("positive_count must be an integer greater than one")
    excluded = set(family) | {tuple(range(positive_count))}
    return tuple(
        subset
        for size in range(positive_count - 1, 0, -1)
        for subset in itertools.combinations(range(positive_count), size)
        if subset not in excluded
    )


def _union_result(masks: list[np.ndarray], positives: list[dict], negatives: list[dict],
                  baseline_mask: np.ndarray) -> tuple[dict, np.ndarray]:
    union = (np.logical_or.reduce(masks)
             if masks else np.zeros_like(baseline_mask, dtype=bool))
    if union.any():
        result = _prompt_metrics(union, positives, negatives) | _geometry(union)
    else:
        result = (_prompt_metrics(union, positives, negatives)
                  | {"area_pixels": 0, "area_ratio": 0.0, "bbox": None,
                     "touches_bottom": False})
    baseline_area = int(baseline_mask.sum())
    intersection = int(np.logical_and(union, baseline_mask).sum())
    union_area = int(np.logical_or(union, baseline_mask).sum())
    retention = intersection / baseline_area if baseline_area else 1.0
    result.update({
        "baseline_retention": round(retention, 6),
        "minimum_baseline_retention": MINIMUM_BASELINE_RETENTION,
        "baseline_retention_pass": retention >= MINIMUM_BASELINE_RETENTION,
        "iou_vs_baseline": round(intersection / union_area if union_area else 1.0, 6),
    })
    result["candidate_available"] = (
        result["prompt_satisfied"]
        and result["negative_points_violated"] == 0
        and result["area_ratio"] <= MAXIMUM_AREA_RATIO
        and result["baseline_retention_pass"])
    return result, union


def evaluate_progressive_image(image: bytes, family: tuple[tuple[int, ...], ...],
                               request_points) -> tuple[dict, dict, dict, dict, list[dict]]:
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

    started = time.monotonic()
    masks, parts, attempts = [], [], []

    def run_subset(subset: tuple[int, ...], phase: str) -> None:
        subset_points = [positives[index] for index in subset]
        row = {"requested_positive_indices": list(subset), "phase": phase}
        try:
            shape, mask, warnings = _decode_response(
                request_points(image, [*subset_points, *negatives]), width, height)
            requested_covered = all(bool(mask[positives[index]["y"],
                                               positives[index]["x"]])
                                    for index in subset)
            prompt = _prompt_metrics(mask, positives, negatives)
            row.update(_geometry(mask) | {
                "score": round(float(shape["score"]), 6),
                "sam_warning_count": len(warnings),
                "requested_points_covered": requested_covered,
                "covered_positive_indices": [
                    index for index, point in enumerate(positives)
                    if bool(mask[point["y"], point["x"]])],
                "negative_points_violated": prompt["negative_points_violated"],
            })
            row["attempt_available"] = (
                requested_covered and not warnings
                and prompt["negative_points_violated"] == 0
                and row["area_ratio"] <= MAXIMUM_AREA_RATIO)
            if row["attempt_available"]:
                masks.append(mask)
                parts.extend(shape["parts"])
        except Exception as error:
            row.update({"attempt_available": False, "error": str(error)[:500]})
        attempts.append(row)

    for subset in family:
        run_subset(subset, "fixed_family")
    result, _ = _union_result(masks, positives, negatives, baseline_mask)
    fixed_family_passed = result["candidate_available"]
    fallback_requests = 0
    if not fixed_family_passed:
        for subset in fallback_order(len(positives), family):
            run_subset(subset, "fallback")
            fallback_requests += 1
            result, _ = _union_result(masks, positives, negatives, baseline_mask)
            if result["candidate_available"]:
                break
    elapsed = time.monotonic() - started
    result.update({
        "fixed_family_request_count": len(family),
        "fallback_request_count": fallback_requests,
        "additional_request_count": len(family) + fallback_requests,
        "accepted_request_count": len(masks),
        "fixed_family_passed": fixed_family_passed,
        "fallback_used": fallback_requests > 0,
        "additional_elapsed_seconds": round(elapsed, 6),
    })
    shape = {
        "id": 2,
        "label": "ego",
        "parts": parts,
        "score": min((row.get("score", 0.0) for row in attempts
                      if row.get("attempt_available")), default=0.0),
    }
    return baseline, result, baseline_shape, shape, attempts


def _contact_sheet(output: Path, rows: list[dict]) -> Path:
    cell_width, image_height, label_height = 480, 384, 30
    canvas = Image.new("RGB", (cell_width * 2,
                               (image_height + label_height) * len(rows)),
                       (15, 23, 31))
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    for row_index, row in enumerate(rows):
        for column, key in enumerate(("baseline_overlay", "progressive_overlay")):
            with Image.open(output / row[key]) as image:
                preview = image.convert("RGB")
                preview.thumbnail((cell_width, image_height))
            left = column * cell_width
            top = row_index * (image_height + label_height)
            canvas.paste(preview, (left + (cell_width - preview.width) // 2,
                                   top + (image_height - preview.height) // 2))
            result = row["baseline" if column == 0 else "progressive"]
            if column == 0:
                label = f"{row['sample_id']} | baseline area={result['area_ratio']:.3f}"
                available = False
            else:
                label = (f"{row['sample_id']} | progressive={result['candidate_available']} "
                         f"requests={result['additional_request_count']} retain={result['baseline_retention']:.2f}")
                available = result["candidate_available"]
            draw.rectangle((left, top + image_height, left + cell_width,
                            top + image_height + label_height), fill=(25, 35, 45))
            draw.text((left + 6, top + image_height + 8), label,
                      fill=(83, 210, 161) if available else (244, 130, 90),
                      font=font)
    path = output / "contact_sheet.jpg"
    canvas.save(path, "JPEG", quality=88)
    return path


def evaluate_scope(discovery_path: Path, image_root: Path, output: Path,
                   request_points) -> dict:
    if output.exists():
        raise FileExistsError(output)
    discovery_bytes = discovery_path.read_bytes()
    discovery = json.loads(discovery_bytes)
    family = derive_minimum_family(discovery)
    development_frames = tuple(discovery["development_frames"])
    held_out_frames = tuple(discovery["held_out_frames"])
    if set(development_frames) & set(held_out_frames):
        raise ValueError("development and held-out frames must be disjoint")
    output.mkdir(parents=True)
    rows = []
    for split, frames in (("development", development_frames),
                          ("held_out", held_out_frames)):
        for frame in frames:
            source = _find_image(image_root, frame)
            image = source.read_bytes()
            baseline, result, baseline_shape, shape, attempts = evaluate_progressive_image(
                image, family, request_points)
            sample_id = f"camera1-frame{frame:03d}"
            baseline_overlay = output / f"{sample_id}-baseline.jpg"
            progressive_overlay = output / f"{sample_id}-progressive.jpg"
            baseline_overlay.write_bytes(candidate_overlay(image, [baseline_shape]))
            progressive_overlay.write_bytes(candidate_overlay(
                image, [shape] if shape["parts"] else []))
            rows.append({
                "sample_id": sample_id,
                "frame": frame,
                "split": split,
                "source_image": str(source.resolve()),
                "source_image_sha256": hashlib.sha256(image).hexdigest(),
                "baseline": baseline,
                "progressive": result,
                "attempts": attempts,
                "baseline_overlay": baseline_overlay.name,
                "baseline_overlay_sha256": _sha256(baseline_overlay),
                "progressive_overlay": progressive_overlay.name,
                "progressive_overlay_sha256": _sha256(progressive_overlay),
            })
    sheet = _contact_sheet(output, rows)

    def summarize(split: str) -> dict:
        selected = [row for row in rows if row["split"] == split]
        return {
            "sample_count": len(selected),
            "candidate_available_count": sum(
                row["progressive"]["candidate_available"] for row in selected),
            "fixed_family_pass_count": sum(
                row["progressive"]["fixed_family_passed"] for row in selected),
            "fallback_used_count": sum(
                row["progressive"]["fallback_used"] for row in selected),
            "negative_violation_count": sum(
                row["progressive"]["negative_points_violated"] for row in selected),
            "additional_request_count": sum(
                row["progressive"]["additional_request_count"] for row in selected),
            "median_additional_request_count": float(np.median([
                row["progressive"]["additional_request_count"] for row in selected])),
            "maximum_additional_request_count": max(
                row["progressive"]["additional_request_count"] for row in selected),
            "median_additional_elapsed_seconds": round(float(np.median([
                row["progressive"]["additional_elapsed_seconds"]
                for row in selected])), 6),
        }

    report = {
        "schema_version": "labelagent.ego_progressive_prompt_family_ab.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "production_deployed": False,
        "formal_state_modified": False,
        "camera": "camera1",
        "discovery_report_sha256": hashlib.sha256(discovery_bytes).hexdigest(),
        "family_derived_from": "development_only",
        "held_out_used_for_selection": False,
        "fixed_family": [list(subset) for subset in family],
        "fallback_order": [list(subset) for subset in fallback_order(4, family)],
        "fallback_order_policy": "larger_positive_subsets_first_then_lexicographic",
        "minimum_baseline_retention": MINIMUM_BASELINE_RETENTION,
        "color_dependent": False,
        "development": summarize("development"),
        "held_out": summarize("held_out"),
        "contact_sheet": sheet.name,
        "contact_sheet_sha256": _sha256(sheet),
        "samples": rows,
        "decision_boundary": (
            "Progressive prompt coverage and baseline retention are not human truth. "
            "Person/floor pixel gates remain required before production."),
    }
    result = output / "results.json"
    result.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-sam-url", default="http://127.0.0.1:28766")
    arguments = parser.parse_args()
    if not arguments.point_sam_url.startswith("http://127.0.0.1:"):
        raise ValueError("point SAM URL must be loopback")
    config = {"sam3_url": arguments.point_sam_url.rstrip("/")}
    report = evaluate_scope(
        arguments.discovery, arguments.image_root, arguments.output,
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
