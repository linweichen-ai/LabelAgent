#!/usr/bin/env python3
"""Evaluate the frozen camera1 prompt family on unexposed timepoints.

Frames are selected deterministically and evenly from the same 152-frame
sequence after excluding every previously exposed frame.  This is a larger
same-sequence holdout, not a cross-dataset generalization claim.
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

from source.agent.workflow import _sam_request, candidate_overlay
from tools.evaluate_ego_fixed_family import (
    MINIMUM_BASELINE_RETENTION,
    derive_minimum_family,
    evaluate_fixed_family_image,
)


PREVIOUSLY_EXPOSED_FRAMES = (0, 19, 38, 57, 76, 95, 114, 133, 151)


def select_even_unexposed(total: int, excluded, count: int) -> tuple[int, ...]:
    if type(total) is not int or total <= 0 or type(count) is not int or count <= 0:
        raise ValueError("total and count must be positive integers")
    if (not isinstance(excluded, (list, tuple, set))
            or any(type(value) is not int or not 0 <= value < total for value in excluded)):
        raise ValueError("excluded frames are invalid")
    available = [index for index in range(total) if index not in set(excluded)]
    if len(available) < count:
        raise ValueError("not enough unexposed frames")
    if count == 1:
        return (available[len(available) // 2],)
    positions = [round(index * (len(available) - 1) / (count - 1))
                 for index in range(count)]
    selected = tuple(available[position] for position in positions)
    if len(selected) != len(set(selected)):
        raise ValueError("even selection produced duplicate frames")
    return selected


def _image_files(image_directory: Path) -> list[Path]:
    files = sorted(path for path in image_directory.iterdir()
                   if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise ValueError("camera image directory is empty")
    return files


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contact_sheet(output: Path, rows: list[dict], columns: int = 4) -> Path:
    cell_width, image_height, label_height = 320, 256, 28
    row_count = (len(rows) + columns - 1) // columns
    canvas = Image.new("RGB", (cell_width * columns,
                               (image_height + label_height) * row_count),
                       (15, 23, 31))
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    for item_index, row in enumerate(rows):
        column = item_index % columns
        item_row = item_index // columns
        with Image.open(output / row["fixed_family_overlay"]) as image:
            preview = image.convert("RGB")
            preview.thumbnail((cell_width, image_height))
        left = column * cell_width
        top = item_row * (image_height + label_height)
        canvas.paste(preview, (left + (cell_width - preview.width) // 2,
                               top + (image_height - preview.height) // 2))
        result = row["fixed_family"]
        label = (f"frame{row['frame']:03d} ok={result['candidate_available']} "
                 f"retain={result['baseline_retention']:.2f}")
        draw.rectangle((left, top + image_height, left + cell_width,
                        top + image_height + label_height), fill=(25, 35, 45))
        draw.text((left + 6, top + image_height + 7), label,
                  fill=(83, 210, 161) if result["candidate_available"]
                  else (244, 130, 90), font=font)
    path = output / "contact_sheet.jpg"
    canvas.save(path, "JPEG", quality=88)
    return path


def evaluate_scope(discovery_path: Path, image_directory: Path, output: Path,
                   request_points, *, sample_count: int = 24,
                   excluded=PREVIOUSLY_EXPOSED_FRAMES) -> dict:
    if output.exists():
        raise FileExistsError(output)
    files = _image_files(image_directory)
    selected = select_even_unexposed(len(files), excluded, sample_count)
    discovery_bytes = discovery_path.read_bytes()
    discovery = json.loads(discovery_bytes)
    family = derive_minimum_family(discovery)
    output.mkdir(parents=True)
    rows = []
    for frame in selected:
        source = files[frame]
        image = source.read_bytes()
        baseline, result, _baseline_shape, fixed_shape, attempts = (
            evaluate_fixed_family_image(image, family, request_points))
        overlay = output / f"camera1-frame{frame:03d}-fixed-family.jpg"
        overlay.write_bytes(candidate_overlay(
            image, [fixed_shape] if fixed_shape["parts"] else []))
        rows.append({
            "sample_id": f"camera1-frame{frame:03d}",
            "frame": frame,
            "source_image": str(source.resolve()),
            "source_image_sha256": hashlib.sha256(image).hexdigest(),
            "baseline": baseline,
            "fixed_family": result,
            "attempts": attempts,
            "fixed_family_overlay": overlay.name,
            "fixed_family_overlay_sha256": _sha256(overlay),
        })
    sheet = _contact_sheet(output, rows)
    report = {
        "schema_version": "labelagent.ego_unexposed_timepoints_ab.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "production_deployed": False,
        "formal_state_modified": False,
        "same_sequence_holdout": True,
        "independent_dataset": False,
        "cross_factory_generalization": False,
        "camera": "camera1",
        "source_image_count": len(files),
        "previously_exposed_frames": list(excluded),
        "selection_policy": "evenly_spaced_after_excluding_all_previously_exposed_frames",
        "selected_frames": list(selected),
        "fixed_family": [list(subset) for subset in family],
        "minimum_baseline_retention": MINIMUM_BASELINE_RETENTION,
        "sample_count": len(rows),
        "prompt_satisfied_count": sum(
            row["fixed_family"]["prompt_satisfied"] for row in rows),
        "candidate_available_count": sum(
            row["fixed_family"]["candidate_available"] for row in rows),
        "retention_failure_count": sum(
            not row["fixed_family"]["baseline_retention_pass"] for row in rows),
        "negative_violation_count": sum(
            row["fixed_family"]["negative_points_violated"] for row in rows),
        "median_baseline_retention": round(float(np.median([
            row["fixed_family"]["baseline_retention"] for row in rows])), 6),
        "median_additional_elapsed_seconds": round(float(np.median([
            row["fixed_family"]["additional_elapsed_seconds"] for row in rows])), 6),
        "additional_request_count": sum(
            row["fixed_family"]["additional_request_count"] for row in rows),
        "discovery_report_sha256": hashlib.sha256(discovery_bytes).hexdigest(),
        "contact_sheet": sheet.name,
        "contact_sheet_sha256": _sha256(sheet),
        "samples": rows,
        "decision_boundary": (
            "Unexposed timepoints from one sequence are not cross-dataset evidence. "
            "Human pixel truth remains required for quality claims."),
    }
    result = output / "results.json"
    result.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--image-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--point-sam-url", default="http://127.0.0.1:28766")
    arguments = parser.parse_args()
    if not arguments.point_sam_url.startswith("http://127.0.0.1:"):
        raise ValueError("point SAM URL must be loopback")
    config = {"sam3_url": arguments.point_sam_url.rstrip("/")}
    report = evaluate_scope(
        arguments.discovery, arguments.image_directory, arguments.output,
        lambda image, points: _sam_request(
            config, "/segment_points", image, timeout_seconds=8.0, points=points),
        sample_count=arguments.sample_count)
    print(json.dumps({
        "output": str((arguments.output / "results.json").resolve()),
        "selected_frames": report["selected_frames"],
        "sample_count": report["sample_count"],
        "prompt_satisfied_count": report["prompt_satisfied_count"],
        "candidate_available_count": report["candidate_available_count"],
        "retention_failure_count": report["retention_failure_count"],
        "negative_violation_count": report["negative_violation_count"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
