#!/usr/bin/env python3
"""Read-only camera1 ego subset-cover A/B.

All non-empty positive-point subsets are evaluated with the same fixed negative
points.  Prompt-valid masks are combined by a deterministic minimum-cardinality
set cover.  Ties prefer agreement with the blocked baseline candidate and then
the higher minimum SAM score.  The strategy is candidate-only and never writes
workbench state or human truth.
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
    _iou,
    _prompt_metrics,
    _sha256,
)


def _cover_key(selection: tuple[dict, ...], union: np.ndarray,
               baseline_mask: np.ndarray) -> tuple:
    return (
        round(_iou(union, baseline_mask), 12),
        min(item["score"] for item in selection),
        -float(union.mean()),
        tuple(tuple(item["requested_positive_indices"]) for item in selection),
    )


def _select_cover(candidates: list[dict], positive_count: int,
                  baseline_mask: np.ndarray) -> tuple[tuple[dict, ...], np.ndarray]:
    target = set(range(positive_count))
    if not candidates:
        return (), np.zeros_like(baseline_mask)

    def coverage(selection) -> set[int]:
        result: set[int] = set()
        for item in selection:
            result.update(item["covered_positive_indices"])
        return result

    for size in range(1, min(positive_count, len(candidates)) + 1):
        full = [selection for selection in itertools.combinations(candidates, size)
                if coverage(selection) == target]
        if full:
            ranked = []
            for selection in full:
                union = np.logical_or.reduce([item["mask"] for item in selection])
                ranked.append((_cover_key(selection, union, baseline_mask),
                               selection, union))
            _, selection, union = max(ranked, key=lambda item: item[0])
            return selection, union

    maximum_coverage = max(len(coverage((item,))) for item in candidates)
    finalists = []
    for size in range(1, min(positive_count, len(candidates)) + 1):
        current = list(itertools.combinations(candidates, size))
        current_maximum = max((len(coverage(item)) for item in current), default=0)
        if current_maximum > maximum_coverage:
            maximum_coverage = current_maximum
            finalists = []
        if current_maximum == maximum_coverage:
            finalists.extend(item for item in current
                              if len(coverage(item)) == maximum_coverage)
        if finalists and maximum_coverage == positive_count:
            break
    minimum_size = min(len(item) for item in finalists)
    finalists = [item for item in finalists if len(item) == minimum_size]
    ranked = []
    for selection in finalists:
        union = np.logical_or.reduce([item["mask"] for item in selection])
        ranked.append((_cover_key(selection, union, baseline_mask), selection, union))
    _, selection, union = max(ranked, key=lambda item: item[0])
    return selection, union


def evaluate_subset_cover_image(image: bytes, request_points) -> tuple[dict, dict, dict, dict, list[dict]]:
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

    attempts, accepted = [], []
    for subset_size in range(1, len(positives) + 1):
        for subset in itertools.combinations(range(len(positives)), subset_size):
            subset_points = [positives[index] for index in subset]
            row = {"requested_positive_indices": list(subset)}
            try:
                shape, mask, warnings = _decode_response(
                    request_points(image, [*subset_points, *negatives]), width, height)
                all_metrics = _prompt_metrics(mask, positives, negatives)
                requested_covered = all(bool(mask[positives[index]["y"],
                                                   positives[index]["x"]])
                                        for index in subset)
                covered = [index for index, point in enumerate(positives)
                           if bool(mask[point["y"], point["x"]])]
                row.update(_geometry(mask) | {
                    "score": round(float(shape["score"]), 6),
                    "sam_warning_count": len(warnings),
                    "requested_points_covered": requested_covered,
                    "covered_positive_indices": covered,
                    "negative_points_violated": all_metrics["negative_points_violated"],
                })
                row["attempt_available"] = (
                    requested_covered and not warnings
                    and all_metrics["negative_points_violated"] == 0
                    and row["area_ratio"] <= MAXIMUM_AREA_RATIO)
                if row["attempt_available"]:
                    accepted.append(row | {"mask": mask, "parts": shape["parts"]})
            except Exception as error:
                row.update({"attempt_available": False, "error": str(error)[:500]})
            attempts.append(row)

    selection, union_mask = _select_cover(accepted, len(positives), baseline_mask)
    if union_mask.any():
        union = _prompt_metrics(union_mask, positives, negatives) | _geometry(union_mask)
    else:
        union = (_prompt_metrics(union_mask, positives, negatives)
                 | {"area_pixels": 0, "area_ratio": 0.0, "bbox": None,
                    "touches_bottom": False})
    union.update({
        "selected_attempt_count": len(selection),
        "selected_positive_subsets": [
            item["requested_positive_indices"] for item in selection],
        "accepted_attempt_count": len(accepted),
        "total_attempt_count": len(attempts),
        "iou_vs_baseline": round(_iou(union_mask, baseline_mask), 6),
    })
    union["candidate_available"] = (
        union["prompt_satisfied"]
        and union["negative_points_violated"] == 0
        and union["area_ratio"] <= MAXIMUM_AREA_RATIO)
    union_parts = [part for item in selection for part in item["parts"]]
    union_shape = {
        "id": 2,
        "label": "ego",
        "parts": union_parts,
        "score": min((item["score"] for item in selection), default=0.0),
    }
    public_attempts = [{key: value for key, value in row.items()
                        if key not in {"mask", "parts"}} for row in attempts]
    return baseline, union, baseline_shape, union_shape, public_attempts


def _contact_sheet(output: Path, rows: list[dict]) -> Path:
    cell_width, image_height, label_height = 480, 384, 30
    canvas = Image.new("RGB", (cell_width * 2,
                               (image_height + label_height) * len(rows)),
                       (15, 23, 31))
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    for row_index, row in enumerate(rows):
        for column, key in enumerate(("baseline_overlay", "subset_cover_overlay")):
            with Image.open(output / row[key]) as image:
                preview = image.convert("RGB")
                preview.thumbnail((cell_width, image_height))
            left = column * cell_width
            top = row_index * (image_height + label_height)
            canvas.paste(preview, (left + (cell_width - preview.width) // 2,
                                   top + (image_height - preview.height) // 2))
            result = row["baseline" if column == 0 else "subset_cover"]
            label = (f"{row['sample_id']} | {'baseline' if column == 0 else 'subset'} "
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
            baseline, subset_cover, baseline_shape, union_shape, attempts = (
                evaluate_subset_cover_image(image, request_points))
            sample_id = f"camera1-frame{frame:03d}"
            baseline_path = output / f"{sample_id}-baseline.jpg"
            subset_path = output / f"{sample_id}-subset-cover.jpg"
            baseline_path.write_bytes(candidate_overlay(image, [baseline_shape]))
            subset_path.write_bytes(candidate_overlay(
                image, [union_shape] if union_shape["parts"] else []))
            rows.append({
                "sample_id": sample_id,
                "frame": frame,
                "split": split,
                "source_image": str(source.resolve()),
                "source_image_sha256": hashlib.sha256(image).hexdigest(),
                "baseline": baseline,
                "subset_cover": subset_cover,
                "attempts": attempts,
                "baseline_overlay": baseline_path.name,
                "baseline_overlay_sha256": _sha256(baseline_path),
                "subset_cover_overlay": subset_path.name,
                "subset_cover_overlay_sha256": _sha256(subset_path),
            })
    sheet = _contact_sheet(output, rows)

    def summarize(split: str) -> dict:
        selected = [row for row in rows if row["split"] == split]
        return {
            "sample_count": len(selected),
            "baseline_available_count": sum(
                row["baseline"]["candidate_available"] for row in selected),
            "subset_cover_available_count": sum(
                row["subset_cover"]["candidate_available"] for row in selected),
            "mean_subset_cover_area_ratio": round(float(np.mean([
                row["subset_cover"]["area_ratio"] for row in selected])), 6),
            "negative_violation_count": sum(
                row["subset_cover"]["negative_points_violated"] for row in selected),
        }

    report = {
        "schema_version": "labelagent.ego_subset_cover_ab.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "human_truth": False,
        "human_confirmed": False,
        "approved_for_training": False,
        "production_deployed": False,
        "formal_state_modified": False,
        "camera": "camera1",
        "strategy": "all_positive_subsets_then_deterministic_minimum_set_cover",
        "color_dependent": False,
        "selection_order": [
            "full_positive_coverage", "fewest_masks", "highest_iou_vs_baseline",
            "highest_minimum_sam_score", "smallest_union_area"],
        "development_frames": list(development_frames),
        "held_out_frames": list(held_out_frames),
        "development": summarize("development"),
        "held_out": summarize("held_out"),
        "contact_sheet": sheet.name,
        "contact_sheet_sha256": _sha256(sheet),
        "samples": rows,
        "decision_boundary": (
            "The blocked baseline is used only as a tie-breaker, never as truth. "
            "Human pixel truth and person/floor overlap gates remain mandatory."),
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
