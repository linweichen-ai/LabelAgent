#!/usr/bin/env python3
"""Audit bounded pallet-mask cleanup across a stored candidate collection.

The input is the ``candidates_by_topic.json`` format used by LabelAgent's
dataset baselines.  The audit is read-only: it replays the current pallet
geometry cleanup on a deterministic size-stratified sample and writes only a
small evidence JSON when ``--output`` is supplied.
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np

from source.agent.edit_agent import _rasterize
from source.agent.workflow import (
    PALLET_COMPACT_TEXTURE_MINIMUM_CANDIDATE_AREA_RATIO,
    PALLET_IMAGE_BOUNDARY_MAXIMUM_COMPONENT_MERGES,
    _compact_pallet_texture_hole,
    refine_pallet_candidate_geometry,
)


def _bbox_area(shape: dict) -> int:
    points = [point for part in shape.get("parts", []) for point in part.get("points", [])]
    if not points:
        points = shape.get("points", [])
    if not points:
        return 0
    xs = [int(point[0]) for point in points]
    ys = [int(point[1]) for point in points]
    return (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)


def select_stratified(rows: list[dict], limit: int) -> list[dict]:
    """Select deterministic size quantiles without duplicating candidates."""
    ordered = sorted(rows, key=lambda row: (
        _bbox_area(row["shape"]), str(row["frame"]), str(row["shape"].get("id", ""))))
    if limit <= 0 or len(ordered) <= limit:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, num=limit, dtype=int).tolist()
    return [ordered[index] for index in dict.fromkeys(indices)]


def enclosed_holes(mask: np.ndarray) -> list[dict]:
    background = (mask == 0).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=8)
    border = set(np.unique(np.concatenate((
        components[0, :], components[-1, :], components[:, 0], components[:, -1]))).tolist())
    holes = []
    for component in range(1, count):
        if component in border:
            continue
        x, y, width, height, area = [int(value) for value in stats[component]]
        if area < 9:
            continue
        holes.append({
            "area": area,
            "bbox": [x, y, x + width - 1, y + height - 1],
            "width": width,
            "height": height,
            "span": max(width, height),
            "aspect_ratio": round(max(width, height) / max(min(width, height), 1), 6),
            "mask": components == component,
        })
    return holes


def _components(mask: np.ndarray) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return sum(int(stats[index, cv2.CC_STAT_AREA]) >= 9 for index in range(1, count))


def audit_candidate(row: dict, width: int, height: int, *, image_rgb=None,
                    image_path: str | None = None) -> dict:
    shape = copy.deepcopy(row["shape"])
    before = _rasterize(shape, width, height).astype(np.uint8)
    refined, adjustment = refine_pallet_candidate_geometry(
        shape, width, height, image_rgb=image_rgb)
    after = _rasterize(refined, width, height).astype(np.uint8)
    before_holes = enclosed_holes(before)
    after_holes = enclosed_holes(after)
    area = int(before.sum())
    def structural_hole(hole):
        micro_hole = hole["area"] <= 96 and hole["span"] <= 32
        texture_hole = hole["area"] <= 256 and hole["span"] <= 24
        compact_texture_hole = (
            area / max(before.size, 1)
            >= PALLET_COMPACT_TEXTURE_MINIMUM_CANDIDATE_AREA_RATIO
            and _compact_pallet_texture_hole(
                hole["area"], hole["width"], hole["height"]))
        return not (micro_hole or texture_hole or compact_texture_hole)

    structural = [hole for hole in before_holes if structural_hole(hole)]
    structural_retention = []
    for hole in structural:
        retained = int(np.count_nonzero(hole["mask"] & (after == 0)))
        structural_retention.append(round(retained / hole["area"], 6))
    changed = int(np.count_nonzero(before != after))
    components_before = _components(before)
    components_after = _components(after)
    component_merge_count = components_before - components_after
    boundary_notch_count = (
        adjustment.get("filled_image_boundary_notch_count", 0)
        if adjustment else 0)
    component_guard_passed = (
        components_after <= components_before
        and (component_merge_count == 0
             or (boundary_notch_count > 0
                 and component_merge_count
                 <= PALLET_IMAGE_BOUNDARY_MAXIMUM_COMPONENT_MERGES)))
    return {
        "topic": row["topic"],
        "frame": str(row["frame"]),
        "shape_id": shape.get("id"),
        "pixel_area": area,
        "candidate_area_ratio": round(area / max(before.size, 1), 6),
        "size_band": ("tiny_skipped" if area < 400 else
                      "small" if area < 5000 else
                      "medium" if area < 50000 else "large"),
        "changed_pixels": changed,
        "change_ratio": round(changed / max(area, 1), 6),
        "adjustment_applied": adjustment is not None,
        "boundary_simplification_applied": bool(
            adjustment and adjustment.get("boundary_simplification_applied")),
        "boundary_simplification_epsilon": (
            adjustment.get("boundary_simplification_epsilon")
            if adjustment else None),
        "boundary_vertices_before": (
            adjustment.get("boundary_vertices_before") if adjustment else None),
        "boundary_vertices_after": (
            adjustment.get("boundary_vertices_after") if adjustment else None),
        "boundary_vertex_reduction_ratio": (
            adjustment.get("boundary_vertex_reduction_ratio", 0.0)
            if adjustment else 0.0),
        "geometry_cleanup_version": (
            adjustment.get("geometry_cleanup_version") if adjustment else None),
        "image_guidance_used": image_rgb is not None,
        "image_path": image_path,
        "effective_maximum_change_ratio": (
            adjustment.get("effective_maximum_change_ratio", .025)
            if adjustment else .025),
        "filled_hole_count": (
            adjustment.get("filled_hole_count", 0) if adjustment else 0),
        "filled_compact_texture_hole_count": (
            adjustment.get("filled_compact_texture_hole_count", 0)
            if adjustment else 0),
        "components_before": components_before,
        "components_after": components_after,
        "component_merge_count": component_merge_count,
        "component_guard_passed": component_guard_passed,
        "filled_image_boundary_notch_count": boundary_notch_count,
        "holes_before": len(before_holes),
        "holes_after": len(after_holes),
        "structural_holes_before": len(structural),
        "minimum_structural_hole_retention": (
            min(structural_retention) if structural_retention else None),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
    }


def _camera_name_for_topic(topic: str) -> str:
    match = re.search(r"/(camera_\d+)(?:/|$)", str(topic))
    if not match:
        raise ValueError(f"cannot map topic to camera directory: {topic}")
    return match.group(1)


@functools.lru_cache(maxsize=8)
def _read_image_rgb(path: str) -> np.ndarray:
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _image_inventory(image_root: Path, topics: list[str]) -> dict[str, list[Path]]:
    inventory = {}
    for topic in topics:
        camera = _camera_name_for_topic(topic)
        files = sorted((image_root / camera).glob("*.png"))
        if not files:
            raise ValueError(f"no PNG images for {topic}: {image_root / camera}")
        inventory[topic] = files
    return inventory


def run_audit(source: Path, *, width: int, height: int, per_topic: int,
              image_root: Path | None = None) -> dict:
    stored = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(stored, dict):
        raise ValueError("expected topic-to-frame candidate mapping")
    payload = stored
    input_scope = "topic_candidate_mapping"
    if isinstance(stored.get("candidates_by_topic"), dict):
        payload = copy.deepcopy(stored["candidates_by_topic"])
        input_scope = "workbench_state.candidates_by_topic"
        active_topic = stored.get("topic")
        active_candidates = stored.get("candidates")
        if isinstance(active_topic, str) and isinstance(active_candidates, dict):
            payload[active_topic] = active_candidates
            input_scope = "workbench_state.current_candidates_overlaid_on_candidates_by_topic"
    selected = []
    topic_inventory = {}
    for topic, frames in sorted(payload.items()):
        rows = [{"topic": topic, "frame": frame, "shape": shape}
                for frame, shapes in frames.items()
                for shape in shapes
                if str(shape.get("label", "")).casefold() == "pallet"]
        topic_inventory[topic] = {
            "frames": len(frames),
            "pallet_candidates": len(rows),
        }
        selected.extend(select_stratified(rows, per_topic))
    images = (_image_inventory(image_root, sorted(payload))
              if image_root is not None else {})
    results = []
    for row in selected:
        image_rgb = None
        image_path = None
        if image_root is not None:
            try:
                frame_index = int(row["frame"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"frame is not an integer index: {row['frame']}") from error
            topic_images = images[row["topic"]]
            if frame_index < 0 or frame_index >= len(topic_images):
                raise ValueError(
                    f"frame {frame_index} outside image inventory for {row['topic']}")
            image_path = str(topic_images[frame_index].resolve())
            image_rgb = _read_image_rgb(image_path)
            if image_rgb.shape[:2] != (height, width):
                raise ValueError(
                    f"image dimensions {image_rgb.shape[1]}x{image_rgb.shape[0]} "
                    f"do not match requested {width}x{height}: {image_path}")
        results.append(audit_candidate(
            row, width, height, image_rgb=image_rgb, image_path=image_path))
    component_violations = [row for row in results
                            if not row["component_guard_passed"]]
    structural_violations = [row for row in results
                             if row["minimum_structural_hole_retention"] is not None
                             and row["minimum_structural_hole_retention"] < 0.75]
    change_violations = [
        row for row in results
        if row["change_ratio"] > row["effective_maximum_change_ratio"]]
    bands = {}
    for band in ("tiny_skipped", "small", "medium", "large"):
        subset = [row for row in results if row["size_band"] == band]
        bands[band] = {
            "count": len(subset),
            "adjusted": sum(row["adjustment_applied"] for row in subset),
            "maximum_change_ratio": max((row["change_ratio"] for row in subset), default=0.0),
        }
    return {
        "schema_version": "labelagent.pallet_geometry_cross_dataset_audit.v5",
        "input": str(source.resolve()),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "width": width,
        "height": height,
        "selection": "deterministic_size_stratified_per_topic",
        "input_scope": input_scope,
        "image_guidance": image_root is not None,
        "image_root": str(image_root.resolve()) if image_root is not None else None,
        "per_topic_limit": per_topic,
        "topic_inventory": topic_inventory,
        "sample_count": len(results),
        "size_bands": bands,
        "adjusted_count": sum(row["adjustment_applied"] for row in results),
        "v5_adjusted_count": sum(
            row["geometry_cleanup_version"] == 5 for row in results),
        "filled_hole_count": sum(row["filled_hole_count"] for row in results),
        "filled_compact_texture_hole_count": sum(
            row["filled_compact_texture_hole_count"] for row in results),
        "filled_image_boundary_notch_count": sum(
            row["filled_image_boundary_notch_count"] for row in results),
        "component_merge_count": sum(
            row["component_merge_count"] for row in results),
        "boundary_simplified_count": sum(
            row["boundary_simplification_applied"] for row in results),
        "median_boundary_vertex_reduction_ratio": (
            round(float(np.median([
                row["boundary_vertex_reduction_ratio"] for row in results
                if row["boundary_simplification_applied"]])), 6)
            if any(row["boundary_simplification_applied"] for row in results)
            else 0.0),
        "boundary_simplification_epsilon_counts": {
            str(epsilon): sum(
                row["boundary_simplification_epsilon"] == epsilon
                for row in results)
            for epsilon in (.75, .5)
        },
        "unchanged_tiny_count": sum(
            row["size_band"] == "tiny_skipped" and row["changed_pixels"] == 0
            for row in results),
        "holes_before": sum(row["holes_before"] for row in results),
        "holes_after": sum(row["holes_after"] for row in results),
        "component_count_violations": component_violations,
        "structural_hole_retention_violations": structural_violations,
        "change_budget_violations": change_violations,
        "gate_passed": not (component_violations or structural_violations or change_violations),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--per-topic", type=int, default=40)
    parser.add_argument(
        "--image-root", type=Path,
        help="Optional root containing camera_0/... PNG directories. When set, "
             "the audit replays the complete image-guided cleanup route.")
    args = parser.parse_args()
    report = run_audit(args.input, width=args.width, height=args.height,
                       per_topic=args.per_topic, image_root=args.image_root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "sample_count", "size_bands", "adjusted_count", "unchanged_tiny_count",
        "v5_adjusted_count", "filled_hole_count", "filled_compact_texture_hole_count",
        "filled_image_boundary_notch_count", "component_merge_count",
        "boundary_simplified_count", "median_boundary_vertex_reduction_ratio",
        "boundary_simplification_epsilon_counts", "holes_before", "holes_after",
        "gate_passed")}, ensure_ascii=False, indent=2))
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
