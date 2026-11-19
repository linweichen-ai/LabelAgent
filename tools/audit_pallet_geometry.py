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
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from source.agent.edit_agent import _rasterize
from source.agent.workflow import refine_pallet_candidate_geometry


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
            "span": max(width, height),
            "mask": components == component,
        })
    return holes


def _components(mask: np.ndarray) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return sum(int(stats[index, cv2.CC_STAT_AREA]) >= 9 for index in range(1, count))


def audit_candidate(row: dict, width: int, height: int) -> dict:
    shape = copy.deepcopy(row["shape"])
    before = _rasterize(shape, width, height).astype(np.uint8)
    refined, adjustment = refine_pallet_candidate_geometry(shape, width, height)
    after = _rasterize(refined, width, height).astype(np.uint8)
    before_holes = enclosed_holes(before)
    after_holes = enclosed_holes(after)
    structural = [hole for hole in before_holes
                  if hole["area"] > 256 or hole["span"] > 32]
    structural_retention = []
    for hole in structural:
        retained = int(np.count_nonzero(hole["mask"] & (after == 0)))
        structural_retention.append(round(retained / hole["area"], 6))
    area = int(before.sum())
    changed = int(np.count_nonzero(before != after))
    return {
        "topic": row["topic"],
        "frame": str(row["frame"]),
        "shape_id": shape.get("id"),
        "pixel_area": area,
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
        "components_before": _components(before),
        "components_after": _components(after),
        "holes_before": len(before_holes),
        "holes_after": len(after_holes),
        "structural_holes_before": len(structural),
        "minimum_structural_hole_retention": (
            min(structural_retention) if structural_retention else None),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
    }


def run_audit(source: Path, *, width: int, height: int, per_topic: int) -> dict:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected topic-to-frame candidate mapping")
    input_scope = "topic_candidate_mapping"
    if isinstance(payload.get("candidates_by_topic"), dict):
        payload = payload["candidates_by_topic"]
        input_scope = "workbench_state.candidates_by_topic"
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
    results = [audit_candidate(row, width, height) for row in selected]
    component_violations = [row for row in results
                            if row["components_before"] != row["components_after"]]
    structural_violations = [row for row in results
                             if row["minimum_structural_hole_retention"] is not None
                             and row["minimum_structural_hole_retention"] < 0.75]
    change_violations = [row for row in results if row["change_ratio"] > 0.025]
    bands = {}
    for band in ("tiny_skipped", "small", "medium", "large"):
        subset = [row for row in results if row["size_band"] == band]
        bands[band] = {
            "count": len(subset),
            "adjusted": sum(row["adjustment_applied"] for row in subset),
            "maximum_change_ratio": max((row["change_ratio"] for row in subset), default=0.0),
        }
    return {
        "schema_version": "labelagent.pallet_geometry_cross_dataset_audit.v1",
        "input": str(source.resolve()),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "width": width,
        "height": height,
        "selection": "deterministic_size_stratified_per_topic",
        "input_scope": input_scope,
        "per_topic_limit": per_topic,
        "topic_inventory": topic_inventory,
        "sample_count": len(results),
        "size_bands": bands,
        "adjusted_count": sum(row["adjustment_applied"] for row in results),
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
    args = parser.parse_args()
    report = run_audit(args.input, width=args.width, height=args.height,
                       per_topic=args.per_topic)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "sample_count", "size_bands", "adjusted_count", "unchanged_tiny_count",
        "boundary_simplified_count", "median_boundary_vertex_reduction_ratio",
        "boundary_simplification_epsilon_counts", "holes_before", "holes_after",
        "gate_passed")}, ensure_ascii=False, indent=2))
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
