#!/usr/bin/env python3
"""Derive and evaluate a fixed camera1 prompt family without held-out tuning.

The minimum family is derived only from development attempt metadata produced
by ``evaluate_ego_subset_cover.py``.  The resulting fixed subsets are then
called unchanged on development and held-out images.  This remains local,
candidate-only evidence and never writes workbench state or human truth.
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
    DEVELOPMENT_FRAMES,
    HELD_OUT_FRAMES,
    MAXIMUM_AREA_RATIO,
    _decode_response,
    _find_image,
    _geometry,
    _prompt_metrics,
    _sha256,
)


MINIMUM_BASELINE_RETENTION = 0.90


def derive_minimum_family(discovery: dict) -> tuple[tuple[int, ...], ...]:
    if (not isinstance(discovery, dict)
            or discovery.get("schema_version") != "labelagent.ego_subset_cover_ab.v1"
            or discovery.get("human_truth") is not False
            or discovery.get("candidate_only") is not True):
        raise ValueError("discovery report has an unsupported contract")
    development = [row for row in discovery.get("samples", [])
                   if isinstance(row, dict) and row.get("split") == "development"]
    if not development:
        raise ValueError("discovery report has no development samples")
    expected_frames = discovery.get("development_frames")
    if (not isinstance(expected_frames, list)
            or sorted(row.get("frame") for row in development) != sorted(expected_frames)):
        raise ValueError("development sample scope does not match the report")

    attempt_maps = []
    keys: set[tuple[int, ...]] = set()
    for row in development:
        mapping = {}
        for attempt in row.get("attempts", []):
            subset = attempt.get("requested_positive_indices")
            if (not isinstance(subset, list) or not subset
                    or any(type(value) is not int or value not in range(4)
                           for value in subset)
                    or subset != sorted(set(subset))):
                raise ValueError("discovery attempt contains an invalid subset")
            key = tuple(subset)
            if key in mapping:
                raise ValueError("discovery attempt subset is duplicated")
            covered = attempt.get("covered_positive_indices", [])
            available = attempt.get("attempt_available") is True
            if (not isinstance(covered, list)
                    or any(type(value) is not int or value not in range(4)
                           for value in covered)):
                raise ValueError("discovery attempt coverage is invalid")
            mapping[key] = set(covered) if available else set()
            keys.add(key)
        attempt_maps.append(mapping)

    ordered = sorted(keys, key=lambda item: (len(item), item))
    target = set(range(4))
    for size in range(1, len(ordered) + 1):
        for family in itertools.combinations(ordered, size):
            if all(set().union(*(mapping.get(key, set()) for key in family)) == target
                   for mapping in attempt_maps):
                return family
    raise ValueError("development samples have no complete fixed family")


def evaluate_fixed_family_image(image: bytes, family: tuple[tuple[int, ...], ...],
                                request_points) -> tuple[dict, dict, dict, dict, list[dict]]:
    with Image.open(io.BytesIO(image)) as decoded:
        width, height = decoded.size
    _, points = materialize_camera_template("camera1", width, height)
    positives = [point for point in points if point["label"] == 1]
    negatives = [point for point in points if point["label"] == 0]
    if (not isinstance(family, tuple) or not family
            or any(not subset or any(index not in range(len(positives)) for index in subset)
                   for subset in family)):
        raise ValueError("fixed family is invalid")

    baseline_started = time.monotonic()
    baseline_shape, baseline_mask, baseline_warnings = _decode_response(
        request_points(image, points), width, height)
    baseline_elapsed = time.monotonic() - baseline_started
    baseline = (_prompt_metrics(baseline_mask, positives, negatives)
                | _geometry(baseline_mask)
                | {"score": round(float(baseline_shape["score"]), 6),
                   "sam_warning_count": len(baseline_warnings),
                   "elapsed_seconds": round(baseline_elapsed, 6)})

    started = time.monotonic()
    rows, masks, parts = [], [], []
    for subset in family:
        subset_points = [positives[index] for index in subset]
        row = {"requested_positive_indices": list(subset)}
        try:
            shape, mask, warnings = _decode_response(
                request_points(image, [*subset_points, *negatives]), width, height)
            requested_covered = all(bool(mask[positives[index]["y"],
                                               positives[index]["x"]])
                                    for index in subset)
            covered = [index for index, point in enumerate(positives)
                       if bool(mask[point["y"], point["x"]])]
            prompt = _prompt_metrics(mask, positives, negatives)
            row.update(_geometry(mask) | {
                "score": round(float(shape["score"]), 6),
                "sam_warning_count": len(warnings),
                "requested_points_covered": requested_covered,
                "covered_positive_indices": covered,
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
        rows.append(row)
    elapsed = time.monotonic() - started
    union_mask = (np.logical_or.reduce(masks)
                  if masks else np.zeros((height, width), dtype=bool))
    if union_mask.any():
        result = _prompt_metrics(union_mask, positives, negatives) | _geometry(union_mask)
    else:
        result = (_prompt_metrics(union_mask, positives, negatives)
                  | {"area_pixels": 0, "area_ratio": 0.0, "bbox": None,
                     "touches_bottom": False})
    baseline_area = int(baseline_mask.sum())
    retained = (float(np.logical_and(union_mask, baseline_mask).sum()) / baseline_area
                if baseline_area else 1.0)
    union_area = int(union_mask.sum())
    iou = (float(np.logical_and(union_mask, baseline_mask).sum())
           / float(np.logical_or(union_mask, baseline_mask).sum())
           if union_area or baseline_area else 1.0)
    result.update({
        "additional_request_count": len(family),
        "baseline_request_count": 1,
        "evaluation_request_count": len(family) + 1,
        "accepted_request_count": len(masks),
        "additional_elapsed_seconds": round(elapsed, 6),
        "baseline_elapsed_seconds": round(baseline_elapsed, 6),
        "baseline_retention": round(retained, 6),
        "minimum_baseline_retention": MINIMUM_BASELINE_RETENTION,
        "baseline_retention_pass": retained >= MINIMUM_BASELINE_RETENTION,
        "iou_vs_baseline": round(iou, 6),
    })
    result["candidate_available"] = (
        result["prompt_satisfied"]
        and result["negative_points_violated"] == 0
        and result["area_ratio"] <= MAXIMUM_AREA_RATIO
        and result["baseline_retention_pass"])
    shape = {
        "id": 1,
        "label": "ego",
        "parts": parts,
        "score": min((row.get("score", 0.0) for row in rows
                      if row.get("attempt_available")), default=0.0),
    }
    return baseline, result, baseline_shape, shape, rows


def _contact_sheet(output: Path, rows: list[dict]) -> Path:
    cell_width, image_height, label_height = 480, 384, 30
    canvas = Image.new("RGB", (cell_width * 2,
                               (image_height + label_height) * len(rows)),
                       (15, 23, 31))
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    for row_index, row in enumerate(rows):
        for column, key in enumerate(("baseline_overlay", "fixed_family_overlay")):
            with Image.open(output / row[key]) as image:
                preview = image.convert("RGB")
                preview.thumbnail((cell_width, image_height))
            left = column * cell_width
            top = row_index * (image_height + label_height)
            canvas.paste(preview, (left + (cell_width - preview.width) // 2,
                                   top + (image_height - preview.height) // 2))
            result = row["baseline" if column == 0 else "fixed_family"]
            if column == 0:
                label = f"{row['sample_id']} | baseline area={result['area_ratio']:.3f}"
                available = False
            else:
                label = (f"{row['sample_id']} | fixed available={result['candidate_available']} "
                         f"retain={result['baseline_retention']:.2f}")
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
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
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
            baseline, result, baseline_shape, shape, attempts = evaluate_fixed_family_image(
                image, family, request_points)
            sample_id = f"camera1-frame{frame:03d}"
            baseline_overlay = output / f"{sample_id}-baseline.jpg"
            fixed_overlay = output / f"{sample_id}-fixed-family.jpg"
            baseline_overlay.write_bytes(candidate_overlay(image, [baseline_shape]))
            fixed_overlay.write_bytes(candidate_overlay(
                image, [shape] if shape["parts"] else []))
            rows.append({
                "sample_id": sample_id,
                "frame": frame,
                "split": split,
                "source_image": str(source.resolve()),
                "source_image_sha256": hashlib.sha256(image).hexdigest(),
                "baseline": baseline,
                "fixed_family": result,
                "attempts": attempts,
                "baseline_overlay": baseline_overlay.name,
                "baseline_overlay_sha256": _sha256(baseline_overlay),
                "fixed_family_overlay": fixed_overlay.name,
                "fixed_family_overlay_sha256": _sha256(fixed_overlay),
            })
    sheet = _contact_sheet(output, rows)

    def summarize(split: str) -> dict:
        selected = [row for row in rows if row["split"] == split]
        return {
            "sample_count": len(selected),
            "candidate_available_count": sum(
                row["fixed_family"]["candidate_available"] for row in selected),
            "negative_violation_count": sum(
                row["fixed_family"]["negative_points_violated"] for row in selected),
            "additional_request_count": sum(
                row["fixed_family"]["additional_request_count"] for row in selected),
            "baseline_evaluation_request_count": len(selected),
            "total_additional_elapsed_seconds": round(sum(
                row["fixed_family"]["additional_elapsed_seconds"]
                for row in selected), 6),
            "median_additional_elapsed_seconds": round(float(np.median([
                row["fixed_family"]["additional_elapsed_seconds"]
                for row in selected])), 6),
            "baseline_retention_failure_count": sum(
                not row["fixed_family"]["baseline_retention_pass"]
                for row in selected),
        }

    report = {
        "schema_version": "labelagent.ego_fixed_prompt_family_ab.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "human_truth": False,
        "human_confirmed": False,
        "approved_for_training": False,
        "production_deployed": False,
        "formal_state_modified": False,
        "camera": "camera1",
        "discovery_report": str(discovery_path.resolve()),
        "discovery_report_sha256": hashlib.sha256(discovery_bytes).hexdigest(),
        "family_derived_from": "development_only",
        "held_out_used_for_selection": False,
        "fixed_family": [list(subset) for subset in family],
        "requests_per_frame": len(family),
        "minimum_baseline_retention": MINIMUM_BASELINE_RETENTION,
        "color_dependent": False,
        "development": summarize("development"),
        "held_out": summarize("held_out"),
        "contact_sheet": sheet.name,
        "contact_sheet_sha256": _sha256(sheet),
        "samples": rows,
        "decision_boundary": (
            "Prompt coverage is not pixel truth. Human masks and person/floor overlap "
            "gates remain required before training or production."),
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
        "fixed_family": report["fixed_family"],
        "development": report["development"],
        "held_out": report["held_out"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
