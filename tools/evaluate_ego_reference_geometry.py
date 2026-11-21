#!/usr/bin/env python3
"""Evaluate reference-truth-derived ego prompts on disjoint CVAT frames."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from source.agent.workflow import _sam_request
from tools.evaluate_ego_component_union import (
    MAXIMUM_AREA_RATIO,
    _decode_response,
    _geometry,
    _prompt_metrics,
)
from tools.evaluate_ego_cvat_truth_shadow import (
    _overlay,
    _sha256,
    _summarize,
    mask_metrics,
    render_label_mask,
)
from tools.evaluate_ego_unexposed_timepoints import select_even_unexposed


def connected_components(mask: np.ndarray, *, minimum_area: int = 1) -> list[np.ndarray]:
    if (not isinstance(mask, np.ndarray) or mask.ndim != 2
            or mask.dtype != np.bool_ or type(minimum_area) is not int
            or minimum_area <= 0):
        raise ValueError("component inputs are invalid")
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components = []
    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue
        queue = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        pixels = []
        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            for next_y, next_x in ((y - 1, x), (y + 1, x),
                                   (y, x - 1), (y, x + 1)):
                if (0 <= next_y < height and 0 <= next_x < width
                        and mask[next_y, next_x]
                        and not visited[next_y, next_x]):
                    visited[next_y, next_x] = True
                    queue.append((next_y, next_x))
        if len(pixels) >= minimum_area:
            components.append(np.asarray(pixels, dtype=np.int32))
    return sorted(components, key=lambda item: (-len(item), int(item[:, 1].min())))


def component_anchor(component: np.ndarray) -> tuple[int, int]:
    if (not isinstance(component, np.ndarray) or component.ndim != 2
            or component.shape[1:] != (2,) or len(component) == 0):
        raise ValueError("component is invalid")
    center = component.mean(axis=0)
    distances = np.square(component - center).sum(axis=1)
    y, x = component[int(np.argmin(distances))]
    return int(x), int(y)


def reference_prompt_contract(ego: np.ndarray, floor: np.ndarray,
                              *, negative_count: int = 3) -> dict:
    if (not isinstance(ego, np.ndarray) or not isinstance(floor, np.ndarray)
            or ego.shape != floor.shape or ego.dtype != np.bool_
            or floor.dtype != np.bool_ or not ego.any() or not floor.any()
            or type(negative_count) is not int or negative_count <= 0):
        raise ValueError("reference masks are invalid")
    components = connected_components(
        ego, minimum_area=max(16, round(int(ego.sum()) * 0.02)))
    if not components:
        raise ValueError("reference ego has no retained component")
    positives = [{"x": x, "y": y, "label": 1}
                 for x, y in map(component_anchor, components)]
    height, width = ego.shape
    available = np.argwhere(np.logical_and(floor, np.logical_not(ego)))
    if len(available) < negative_count:
        raise ValueError("reference floor has too few safe pixels")
    targets = np.linspace(0.2, 0.8, negative_count)
    negatives, used = [], set()
    for x_fraction in targets:
        target = np.array([height * 0.75, width * x_fraction])
        order = np.argsort(np.square(available - target).sum(axis=1))
        selected = None
        for index in order:
            y, x = map(int, available[int(index)])
            if (x, y) not in used:
                selected = (x, y)
                break
        if selected is None:
            raise ValueError("reference negative points are not unique")
        used.add(selected)
        negatives.append({"x": selected[0], "y": selected[1], "label": 0})
    return {
        "reference_component_count": len(components),
        "positive_points": positives,
        "negative_points": negatives,
        "points": [*positives, *negatives],
    }


def evaluate_reference_image(image: bytes, contract: dict,
                             request_points) -> tuple[dict, dict]:
    with Image.open(io.BytesIO(image)) as opened:
        width, height = opened.size
    positives = contract["positive_points"]
    negatives = contract["negative_points"]
    masks, parts, attempts = [], [], []
    started = time.monotonic()
    for index, positive in enumerate(positives):
        row = {"positive_index": index, "point": positive}
        try:
            shape, mask, warnings = _decode_response(
                request_points(image, [positive, *negatives]), width, height)
            requested_covered = bool(mask[positive["y"], positive["x"]])
            prompt = _prompt_metrics(mask, positives, negatives)
            row.update(_geometry(mask) | {
                "score": round(float(shape["score"]), 6),
                "sam_warning_count": len(warnings),
                "requested_point_covered": requested_covered,
                "negative_points_violated": prompt["negative_points_violated"],
            })
            row["attempt_available"] = (
                requested_covered and not warnings
                and row["negative_points_violated"] == 0
                and row["area_ratio"] <= MAXIMUM_AREA_RATIO)
            if row["attempt_available"]:
                masks.append(mask)
                parts.extend(shape["parts"])
        except Exception as error:
            row.update({"attempt_available": False, "error": str(error)[:500]})
        attempts.append(row)
    elapsed = time.monotonic() - started
    union = (np.logical_or.reduce(masks) if masks
             else np.zeros((height, width), dtype=bool))
    prompt = _prompt_metrics(union, positives, negatives)
    result = prompt | _geometry(union) | {
        "request_count": len(positives),
        "accepted_request_count": len(masks),
        "elapsed_seconds": round(elapsed, 6),
        "candidate_available": (
            len(masks) == len(positives)
            and prompt["prompt_satisfied"]
            and prompt["negative_points_violated"] == 0
            and _geometry(union)["area_ratio"] <= MAXIMUM_AREA_RATIO),
    }
    return {"result": result, "attempts": attempts,
            "shape": {"id": 1, "label": "ego", "parts": parts,
                      "score": min((row.get("score", 0.0) for row in attempts
                                    if row.get("attempt_available")), default=0.0)}}, union


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
        metrics = row["geometry_metrics"]
        label = (f"f{row['frame']:03d} gate={row['reference_geometry']['candidate_available']} "
                 f"IoU={metrics['iou']:.2f} P={metrics['person_overlap']:.2f} "
                 f"F={metrics['floor_overlap']:.2f}")
        draw.rectangle((left, top + image_height, left + cell_width,
                        top + image_height + label_height), fill=(25, 35, 45))
        draw.text((left + 5, top + image_height + 8), label,
                  fill=(83, 210, 161) if row["reference_geometry"]["candidate_available"]
                  else (244, 130, 90), font=font)
    path = output / "contact_sheet.jpg"
    canvas.save(path, "JPEG", quality=88)
    return path


def evaluate_scope(image_directory: Path, annotation_path: Path,
                   import_manifest_path: Path, output: Path, request_points,
                   *, reference_frame: int = 0, sample_count: int = 24) -> dict:
    if output.exists():
        raise FileExistsError(output)
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    manifest = json.loads(import_manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("dataset") != "CVAT Task3"
            or manifest.get("frame_count") != 150
            or len(manifest.get("frames", [])) != 150
            or not 0 <= reference_frame < 150):
        raise ValueError("CVAT snapshot contract is invalid")
    reference_shapes = annotations.get(str(reference_frame))
    size = tuple(manifest["image_size"])
    reference_ego = render_label_mask(reference_shapes, "ego", size)
    reference_floor = render_label_mask(reference_shapes, "floor", size)
    contract = reference_prompt_contract(reference_ego, reference_floor)
    selected = select_even_unexposed(150, (reference_frame,), sample_count)
    output.mkdir(parents=True)
    rows = []
    for frame in selected:
        frame_meta = manifest["frames"][frame]
        source = image_directory / frame_meta["file"]
        image_bytes = source.read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != frame_meta["sha256"]:
            raise ValueError(f"source image hash mismatch for frame {frame}")
        shapes = annotations.get(str(frame))
        ego = render_label_mask(shapes, "ego", size)
        person = render_label_mask(shapes, "person", size)
        floor = render_label_mask(shapes, "floor", size)
        evaluated, candidate = evaluate_reference_image(
            image_bytes, contract, request_points)
        accepted = (candidate if evaluated["result"]["candidate_available"]
                    else np.zeros_like(candidate))
        comparison = output / f"cvat-frame{frame:03d}-truth-reference.jpg"
        _overlay(source, ego, candidate, comparison)
        rows.append({
            "sample_id": f"cvat-task3-frame{frame:03d}",
            "frame": frame,
            "source_image_sha256": frame_meta["sha256"],
            "reference_geometry": evaluated["result"],
            "attempts": evaluated["attempts"],
            "geometry_metrics": mask_metrics(candidate, ego, person, floor),
            "accepted_candidate_metrics": mask_metrics(
                accepted, ego, person, floor),
            "comparison_overlay": comparison.name,
            "comparison_overlay_sha256": _sha256(comparison),
        })
    sheet = _contact_sheet(output, rows)
    available_rows = [row for row in rows
                      if row["reference_geometry"]["candidate_available"]]
    report = {
        "schema_version": "labelagent.ego_reference_geometry_shadow.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "CVAT Task3 local reviewed snapshot",
        "reference_frame": reference_frame,
        "reference_frame_excluded_from_evaluation": True,
        "reference_truth_source": "existing_workbench_cvat_polygon",
        "formal_human_receipt_v3": False,
        "diagnostic_only": True,
        "color_dependent": False,
        "candidate_only": True,
        "approved_for_training": False,
        "production_deployed": False,
        "formal_state_modified": False,
        "contract": contract,
        "selected_frames": list(selected),
        "sample_count": len(rows),
        "candidate_available_count": sum(
            row["reference_geometry"]["candidate_available"] for row in rows),
        "request_failure_count": sum(
            any("error" in attempt for attempt in row["attempts"])
            for row in rows),
        "negative_violation_count": sum(
            row["reference_geometry"]["negative_points_violated"] for row in rows),
        "geometry": _summarize(rows, "geometry_metrics"),
        "available_geometry": (_summarize(available_rows, "geometry_metrics")
                               if available_rows else None),
        "accepted_candidate": _summarize(rows, "accepted_candidate_metrics"),
        "median_elapsed_seconds": round(float(np.median([
            row["reference_geometry"]["elapsed_seconds"] for row in rows])), 6),
        "annotation_snapshot_sha256": _sha256(annotation_path),
        "import_manifest_sha256": _sha256(import_manifest_path),
        "contact_sheet": sheet.name,
        "contact_sheet_sha256": _sha256(sheet),
        "samples": rows,
        "decision_boundary": (
            "One CVAT reference frame derives prompts and is excluded from evaluation. "
            "The snapshot lacks the formal v3 human receipt, so results remain diagnostic."),
    }
    (output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-directory", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--import-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--point-sam-url", default="http://127.0.0.1:28766")
    arguments = parser.parse_args()
    if not arguments.point_sam_url.startswith("http://127.0.0.1:"):
        raise ValueError("point SAM URL must be loopback")
    config = {"sam3_url": arguments.point_sam_url.rstrip("/")}
    report = evaluate_scope(
        arguments.image_directory, arguments.annotations,
        arguments.import_manifest, arguments.output,
        lambda image, points: _sam_request(
            config, "/segment_points", image, timeout_seconds=8.0, points=points),
        reference_frame=arguments.reference_frame,
        sample_count=arguments.sample_count)
    print(json.dumps({
        "output": str((arguments.output / "results.json").resolve()),
        "reference_frame": report["reference_frame"],
        "contract": report["contract"],
        "sample_count": report["sample_count"],
        "candidate_available_count": report["candidate_available_count"],
        "request_failure_count": report["request_failure_count"],
        "geometry": report["geometry"],
        "available_geometry": report["available_geometry"],
        "accepted_candidate": report["accepted_candidate"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
