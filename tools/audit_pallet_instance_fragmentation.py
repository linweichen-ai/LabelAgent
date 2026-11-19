#!/usr/bin/env python3
"""Rank possible same-instance pallet fragments without merging candidates.

This is a review tool, not an automatic repair.  It uses only bounded geometric
signals to find a small candidate that is contained by or lies very close to a
larger pallet candidate in the same image.  Every result remains a hypothesis
requiring image/VLM or human review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _mapping(payload: dict) -> dict:
    nested = payload.get("candidates_by_topic")
    if isinstance(nested, dict):
        return nested
    if payload and all(str(key).isdigit() for key in payload):
        return {"/CVAT/Task3/images": payload}
    return payload


def _parts(shape: dict) -> list[dict]:
    parts = shape.get("parts")
    return parts if isinstance(parts, list) and parts else [shape]


def _bbox(shape: dict) -> tuple[float, float, float, float] | None:
    points = [point for part in _parts(shape)
              for point in (part.get("points") or [])]
    if len(points) < 3:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _polygon_area(points: list) -> float:
    if len(points) < 3:
        return 0.0
    return abs(sum(
        float(points[index][0]) * float(points[(index + 1) % len(points)][1])
        - float(points[(index + 1) % len(points)][0]) * float(points[index][1])
        for index in range(len(points))) / 2)


def _shape_area(shape: dict) -> float:
    area = 0.0
    for part in _parts(shape):
        area += _polygon_area(part.get("points") or [])
        area -= sum(_polygon_area(hole) for hole in (part.get("holes") or []))
    return max(area, 1.0)


def _pair_geometry(first: dict, second: dict) -> dict:
    a = first["bbox"]
    b = second["bbox"]
    width_a, height_a = a[2] - a[0] + 1, a[3] - a[1] + 1
    width_b, height_b = b[2] - b[0] + 1, b[3] - b[1] + 1
    overlap_x = max(0.0, min(a[2], b[2]) - max(a[0], b[0]) + 1)
    overlap_y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
    intersection = overlap_x * overlap_y
    smaller_bbox = min(width_a * height_a, width_b * height_b)
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return {
        "bbox_containment": intersection / max(smaller_bbox, 1.0),
        "bbox_gap_pixels": math.hypot(dx, dy),
        "horizontal_overlap": overlap_x / max(min(width_a, width_b), 1.0),
        "vertical_overlap": overlap_y / max(min(height_a, height_b), 1.0),
    }


def _local_mask(shape: dict, crop: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = crop
    mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
    for part in _parts(shape):
        points = np.rint(np.asarray(part.get("points") or [], dtype=np.float64)).astype(
            np.int32)
        if len(points) < 3:
            continue
        points[:, 0] -= left
        points[:, 1] -= top
        cv2.fillPoly(mask, [points], 1)
        for hole in part.get("holes") or []:
            hole_points = np.rint(np.asarray(hole, dtype=np.float64)).astype(np.int32)
            if len(hole_points) >= 3:
                hole_points[:, 0] -= left
                hole_points[:, 1] -= top
                cv2.fillPoly(mask, [hole_points], 0)
    return mask


def _mask_pair_geometry(first: dict, second: dict) -> dict:
    boxes = (first["bbox"], second["bbox"])
    left = math.floor(min(box[0] for box in boxes)) - 2
    top = math.floor(min(box[1] for box in boxes)) - 2
    right = math.ceil(max(box[2] for box in boxes)) + 3
    bottom = math.ceil(max(box[3] for box in boxes)) + 3
    crop = left, top, right, bottom
    first_mask = _local_mask(first["shape"], crop)
    second_mask = _local_mask(second["shape"], crop)
    first_area = int(np.count_nonzero(first_mask))
    second_area = int(np.count_nonzero(second_mask))
    intersection = int(np.count_nonzero(first_mask & second_mask))
    if intersection:
        gap = 0.0
    elif first_area and second_area:
        distance = cv2.distanceTransform(
            (first_mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
        gap = float(np.min(distance[second_mask != 0]))
    else:
        gap = float("inf")
    return {
        "mask_containment": intersection / max(min(first_area, second_area), 1),
        "mask_gap_pixels": gap,
        "mask_intersection_pixels": intersection,
        "first_mask_pixels": first_area,
        "second_mask_pixels": second_area,
    }


def _candidate_groups(payload: dict, audit_results: list[dict] | None = None
                      ) -> dict[tuple[str, int], list[dict]]:
    audit_by_key = {
        (str(row.get("topic")), int(row.get("frame", -1)),
         str(row.get("shape_id"))): row
        for row in (audit_results or [])
    }
    groups: dict[tuple[str, int], list[dict]] = {}
    for topic, frames in sorted(_mapping(payload).items()):
        for frame, shapes in frames.items():
            for shape in shapes:
                if str(shape.get("label", "")).casefold() != "pallet":
                    continue
                bbox = _bbox(shape)
                if bbox is None:
                    continue
                audit = audit_by_key.get((str(topic), int(frame), str(shape.get("id"))))
                if audit_results is not None and audit is None:
                    continue
                area = float((audit or {}).get("pixel_area") or _shape_area(shape))
                component_count = int(
                    (audit or {}).get("components_before") or len(_parts(shape)))
                width = bbox[2] - bbox[0] + 1
                height = bbox[3] - bbox[1] + 1
                groups.setdefault((str(topic), int(frame)), []).append({
                    "topic": str(topic), "frame": int(frame),
                    "shape_id": shape.get("id"), "shape": shape, "bbox": bbox,
                    "area": area, "width": width, "height": height,
                    "thinness": min(width, height) / max(width, height),
                    "component_count": component_count,
                })
    return groups


def rank_fragment_pairs(payload: dict, *, audit_results: list[dict] | None = None,
                        limit: int = 24,
                        conservative_only: bool = False
                        ) -> tuple[list[dict], int, int, int]:
    groups = _candidate_groups(payload, audit_results)
    rows = []
    considered = 0
    for (topic, frame), shapes in groups.items():
        for first, second in combinations(shapes, 2):
            considered += 1
            small, large = sorted((first, second), key=lambda row: row["area"])
            if small["component_count"] != 1 or large["component_count"] != 1:
                continue
            area_ratio = small["area"] / max(large["area"], 1.0)
            geometry = _pair_geometry(small, large)
            axis_overlap = max(
                geometry["horizontal_overlap"], geometry["vertical_overlap"])
            gap_limit = max(12.0, .08 * max(
                small["width"], small["height"], large["width"], large["height"]))
            contained = geometry["bbox_containment"] >= .4
            attached = (geometry["bbox_gap_pixels"] <= gap_limit
                        and axis_overlap >= .4 and area_ratio <= .55)
            if not contained and not attached:
                continue
            mask_geometry = _mask_pair_geometry(small, large)
            mask_contained = mask_geometry["mask_containment"] >= .4
            mask_attached = (mask_geometry["mask_gap_pixels"] <= gap_limit
                             and axis_overlap >= .4 and area_ratio <= .55)
            if not mask_contained and not mask_attached:
                continue
            proximity = max(
                0.0, 1.0 - mask_geometry["mask_gap_pixels"] / max(gap_limit, 1.0))
            score = (
                .45 * mask_geometry["mask_containment"]
                + .25 * proximity
                + .20 * (1.0 - min(area_ratio, 1.0))
                + .10 * (1.0 - min(small["thinness"] / .4, 1.0)))
            relation = ("possible_duplicate_or_nested_fragment"
                        if mask_geometry["mask_containment"] >= .65
                        else "possible_attached_fragment")
            smaller_mask_pixels = min(
                mask_geometry["first_mask_pixels"],
                mask_geometry["second_mask_pixels"])
            larger_mask_pixels = max(
                mask_geometry["first_mask_pixels"],
                mask_geometry["second_mask_pixels"])
            outside_pixels = max(
                0, smaller_mask_pixels - mask_geometry["mask_intersection_pixels"])
            union_expansion = outside_pixels / max(larger_mask_pixels, 1)
            conservative_duplicate_gate = (
                mask_geometry["mask_containment"] >= .85
                and area_ratio <= .65
                and union_expansion <= .03)
            rows.append({
                "topic": topic, "frame": frame,
                "small_id": small["shape_id"], "large_id": large["shape_id"],
                "small_area": int(round(small["area"])),
                "large_area": int(round(large["area"])),
                "area_ratio": round(area_ratio, 6),
                "small_thinness": round(small["thinness"], 6),
                **{key: round(value, 6) for key, value in geometry.items()},
                **{key: round(value, 6) for key, value in mask_geometry.items()},
                "gap_limit_pixels": round(gap_limit, 3),
                "fragment_hypothesis_score": round(score, 6),
                "relation_hypothesis": relation,
                "small_outside_large_pixels": outside_pixels,
                "union_expansion_fraction_of_large": round(union_expansion, 6),
                "conservative_duplicate_gate": conservative_duplicate_gate,
                "review_required": True,
                "automatic_merge_allowed": False,
                "_small": small,
                "_large": large,
            })
    rows.sort(key=lambda row: (
        row["fragment_hypothesis_score"], row["mask_containment"],
        -row["area_ratio"], row["topic"], -row["frame"],
        str(row["small_id"]), str(row["large_id"])), reverse=True)
    conservative_count = sum(row["conservative_duplicate_gate"] for row in rows)
    selected_rows = ([row for row in rows if row["conservative_duplicate_gate"]]
                     if conservative_only else rows)
    return selected_rows[:max(0, limit)], considered, len(rows), conservative_count


def rank_disjoint_pair_controls(
        payload: dict, *, audit_results: list[dict] | None = None,
        limit: int = 12) -> tuple[list[dict], int, int]:
    """Rank disjoint pairs for semantic-control review without assigning truth.

    Mask separation proves only that two candidates do not overlap.  It cannot
    establish that stacked boards, rails, or adjacent fragments belong to
    different physical pallet instances.  Callers must therefore review these
    rows before using any of them as negative controls.
    """
    groups = _candidate_groups(payload, audit_results)
    rows = []
    considered = 0
    for (topic, frame), shapes in groups.items():
        for first, second in combinations(shapes, 2):
            considered += 1
            if first["component_count"] != 1 or second["component_count"] != 1:
                continue
            small, large = sorted((first, second), key=lambda row: row["area"])
            area_ratio = small["area"] / max(large["area"], 1.0)
            if area_ratio < .5:
                continue
            geometry = _pair_geometry(small, large)
            axis_overlap = max(
                geometry["horizontal_overlap"], geometry["vertical_overlap"])
            gap_limit = max(40.0, .25 * max(
                small["width"], small["height"], large["width"], large["height"]))
            if (geometry["bbox_gap_pixels"] > gap_limit or axis_overlap < .3
                    or geometry["bbox_containment"] > .2):
                continue
            mask_geometry = _mask_pair_geometry(small, large)
            if (mask_geometry["mask_containment"] > .05
                    or mask_geometry["mask_gap_pixels"] < 3):
                continue
            proximity = max(
                0.0, 1.0 - mask_geometry["mask_gap_pixels"] / gap_limit)
            score = .6 * proximity + .4 * min(area_ratio, 1.0)
            rows.append({
                "topic": topic, "frame": frame,
                "small_id": small["shape_id"], "large_id": large["shape_id"],
                "small_area": int(round(small["area"])),
                "large_area": int(round(large["area"])),
                "area_ratio": round(area_ratio, 6),
                "small_thinness": round(small["thinness"], 6),
                **{key: round(value, 6) for key, value in geometry.items()},
                **{key: round(value, 6) for key, value in mask_geometry.items()},
                "gap_limit_pixels": round(gap_limit, 3),
                "fragment_hypothesis_score": round(score, 6),
                "relation_hypothesis": "disjoint_pair_control_candidate",
                "small_outside_large_pixels": mask_geometry["first_mask_pixels"],
                "union_expansion_fraction_of_large": round(
                    mask_geometry["first_mask_pixels"]
                    / max(mask_geometry["second_mask_pixels"], 1), 6),
                "conservative_duplicate_gate": False,
                "review_required": True,
                "automatic_merge_allowed": False,
                "_small": small,
                "_large": large,
            })
    rows.sort(key=lambda row: (
        row["fragment_hypothesis_score"], row["area_ratio"], row["topic"],
        -row["frame"], str(row["small_id"]), str(row["large_id"])), reverse=True)
    return rows[:max(0, limit)], considered, len(rows)


def _camera_name(topic: str) -> str:
    match = re.search(r"/(camera_\d+)(?:/|$)", topic)
    if not match:
        raise ValueError(f"cannot map topic to camera directory: {topic}")
    return match.group(1)


def _image_inventory(image_root: Path, topic: str) -> tuple[str, list[Path]]:
    try:
        camera = _camera_name(topic)
    except ValueError:
        camera = "flat"
    folder = image_root / camera if (image_root / camera).is_dir() else image_root
    files = sorted(
        path for pattern in ("*.png", "*.jpg", "*.jpeg")
        for path in folder.glob(pattern))
    if not files:
        raise ValueError(f"no supported images for {topic}: {folder}")
    return camera, files


def _draw_shape(layer: Image.Image, shape: dict, fill: tuple, outline: tuple) -> None:
    draw = ImageDraw.Draw(layer)
    for part in _parts(shape):
        points = [tuple(map(float, point)) for point in (part.get("points") or [])]
        if len(points) >= 3:
            draw.polygon(points, fill=fill, outline=outline, width=5)
        for hole in part.get("holes") or []:
            hole_points = [tuple(map(float, point)) for point in hole]
            if len(hole_points) >= 3:
                draw.polygon(hole_points, fill=(0, 0, 0, 0), outline=outline, width=3)


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, (19, 27, 34))
    panel.paste(fitted, ((size[0] - fitted.width) // 2,
                         (size[1] - fitted.height) // 2))
    return panel


def render_fragment_review(candidate_file: Path, image_root: Path,
                           audit_report: Path | None, output_dir: Path, *,
                           limit: int = 24,
                           conservative_only: bool = False,
                           separate_control: bool = False) -> dict:
    if conservative_only and separate_control:
        raise ValueError("conservative and separate-control selections are exclusive")
    source_before = candidate_file.read_bytes()
    payload = json.loads(source_before.decode("utf-8"))
    audit_payload = (json.loads(audit_report.read_text(encoding="utf-8"))
                     if audit_report is not None else {})
    audit_results = (audit_payload.get("results") or []) if audit_report else None
    if separate_control:
        rows, considered, hypothesis_count = rank_disjoint_pair_controls(
            payload, audit_results=audit_results, limit=limit)
        conservative_count = 0
    else:
        rows, considered, hypothesis_count, conservative_count = rank_fragment_pairs(
            payload, audit_results=audit_results, limit=limit,
            conservative_only=conservative_only)
    if not rows:
        raise ValueError("no bounded pallet fragment hypotheses found")

    inventories: dict[str, list[Path]] = {}
    panels = []
    public_rows = []
    font = ImageFont.load_default()
    for rank, row in enumerate(rows, start=1):
        inventory_key = row["topic"]
        if inventory_key not in inventories:
            camera, files = _image_inventory(image_root, row["topic"])
            inventories[inventory_key] = files
        else:
            camera = (_camera_name(row["topic"])
                      if "/camera_" in row["topic"] else "flat")
        files = inventories[inventory_key]
        if row["frame"] < 0 or row["frame"] >= len(files):
            raise ValueError(f"frame {row['frame']} outside {camera} image inventory")
        image_path = files[row["frame"]]
        image = Image.open(image_path).convert("RGB")
        boxes = (row["_small"]["bbox"], row["_large"]["bbox"])
        left = min(box[0] for box in boxes)
        top = min(box[1] for box in boxes)
        right = max(box[2] for box in boxes)
        bottom = max(box[3] for box in boxes)
        pad = max(28, round(max(right - left, bottom - top) * .2))
        crop = (max(0, int(left) - pad), max(0, int(top) - pad),
                min(image.width, int(right) + pad + 1),
                min(image.height, int(bottom) + pad + 1))
        original = _fit(image.crop(crop), (500, 300))
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        _draw_shape(layer, row["_large"]["shape"],
                    (38, 217, 133, 72), (38, 217, 133, 255))
        _draw_shape(layer, row["_small"]["shape"],
                    (255, 148, 32, 96), (255, 148, 32, 255))
        overlay = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
        overlay = _fit(overlay.crop(crop), (500, 300))
        panel = Image.new("RGB", (1000, 350), "white")
        panel.paste(original, (0, 50))
        panel.paste(overlay, (500, 50))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 6),
                  f"#{rank} {camera} frame {row['frame']}  small={row['small_id']}  "
                  f"large={row['large_id']}", fill="black", font=font)
        draw.text((8, 27),
                  f"score={row['fragment_hypothesis_score']:.3f}  "
                  f"mask contain={row['mask_containment']:.3f}  "
                  f"mask gap={row['mask_gap_pixels']:.1f}px  "
                  f"area ratio={row['area_ratio']:.3f}", fill="black", font=font)
        draw.text((510, 27), "green=larger candidate; orange=smaller candidate",
                  fill="black", font=font)
        panels.append(panel)
        public_rows.append({
            key: value for key, value in row.items() if not key.startswith("_")
        } | {"rank": rank, "image_path": str(image_path.resolve())})

    sheet = Image.new("RGB", (1000, sum(panel.height for panel in panels)), "white")
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_name = ("pallet-separate-instance-controls.jpg" if separate_control
                  else "pallet-conservative-duplicate-gate.jpg" if conservative_only
                  else "pallet-fragment-hypotheses.jpg")
    sheet_path = output_dir / sheet_name
    sheet.save(sheet_path, quality=90, optimize=True)
    if candidate_file.read_bytes() != source_before:
        raise RuntimeError("candidate source changed during read-only fragment review")
    report = {
        "schema_version": "labelagent.pallet_fragment_review.v1",
        "candidate_file": str(candidate_file.resolve()),
        "candidate_sha256": hashlib.sha256(source_before).hexdigest(),
        "audit_report": str(audit_report.resolve()) if audit_report else None,
        "image_root": str(image_root.resolve()),
        "pair_count_considered": considered,
        "fragment_hypothesis_count": hypothesis_count,
        "conservative_duplicate_gate_count": conservative_count,
        "selected_count": len(public_rows),
        "selected_conservative_duplicate_gate_count": sum(
            row["conservative_duplicate_gate"] for row in public_rows),
        "selection": ("disjoint_pair_control_candidates" if separate_control
                      else "bounded_same_frame_geometry_hypotheses"),
        "semantic_truth_assigned": False,
        "conservative_only": conservative_only,
        "separate_control": separate_control,
        "contact_sheet": str(sheet_path.resolve()),
        "contact_sheet_sha256": hashlib.sha256(sheet_path.read_bytes()).hexdigest(),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "automatic_merge_allowed": False,
        "rows": public_rows,
    }
    report_name = ("pallet-separate-instance-controls-review.json"
                   if separate_control else
                   "pallet-conservative-duplicate-gate-review.json"
                   if conservative_only else "pallet-fragment-review.json")
    report_path = output_dir / report_name
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path.resolve())
    report["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--conservative-only", action="store_true")
    parser.add_argument("--separate-control", action="store_true")
    args = parser.parse_args()
    report = render_fragment_review(
        args.candidates, args.image_root, args.audit_report, args.output,
        limit=args.limit, conservative_only=args.conservative_only,
        separate_control=args.separate_control)
    print(json.dumps({key: report[key] for key in (
        "pair_count_considered", "fragment_hypothesis_count",
        "conservative_duplicate_gate_count", "selected_count",
        "selected_conservative_duplicate_gate_count", "contact_sheet",
        "contact_sheet_sha256", "report_path", "report_sha256",
        "automatic_merge_allowed")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
