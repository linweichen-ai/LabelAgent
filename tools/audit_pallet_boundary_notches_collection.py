#!/usr/bin/env python3
"""Read-only image-guided pallet cleanup audit across the live collection."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from source.agent.edit_agent import _rasterize
from source.agent.workflow import (
    PALLET_IMAGE_BOUNDARY_MAXIMUM_COMPONENT_MERGES,
    _pallet_geometry_vertex_count,
    _significant_component_count,
    refine_pallet_candidate_geometry,
)
from tools.audit_pallet_boundary_notches_live import (
    _frame_shapes,
    _get_image,
    _get_json,
    _minimum_structural_retention,
)


def audit_collection(base_url: str, *, start_frame: int = 0,
                     end_frame: int | None = None) -> dict:
    bootstrap = _get_json(base_url, "/api/bootstrap")
    scope = {
        "path": bootstrap["path"],
        "topic": bootstrap["topic"],
        "file_activation_generation": bootstrap["file_activation_generation"],
    }
    total_frame_count = int(bootstrap["catalog"][bootstrap["topic"]]["count"])
    end_frame = total_frame_count if end_frame is None else min(
        int(end_frame), total_frame_count)
    start_frame = max(0, int(start_frame))
    if start_frame >= end_frame:
        raise ValueError("start-frame must be smaller than end-frame")
    results = []
    frame_inventory = []
    for frame in range(start_frame, end_frame):
        state = _get_json(base_url, f"/api/frame_state/{frame}", scope)
        candidates = _frame_shapes(state.get("candidates", {}), frame)
        pallet_candidates = [
            shape for shape in candidates
            if str(shape.get("label", "")).casefold() == "pallet"
        ]
        frame_inventory.append({
            "frame": frame,
            "candidate_count": len(candidates),
            "pallet_candidate_count": len(pallet_candidates),
        })
        if not pallet_candidates:
            continue
        image_rgb = _get_image(base_url, frame, scope)
        height, width = image_rgb.shape[:2]
        for shape in pallet_candidates:
            before = _rasterize(shape, width, height).astype(np.uint8)
            refined, adjustment = refine_pallet_candidate_geometry(
                copy.deepcopy(shape), width, height, image_rgb=image_rgb)
            after = _rasterize(refined, width, height).astype(np.uint8)
            adjustment = adjustment or {}
            changed = int(np.count_nonzero(before != after))
            components_before = _significant_component_count(before)
            components_after = _significant_component_count(after)
            merge_count = components_before - components_after
            notch_count = int(adjustment.get(
                "filled_image_boundary_notch_count", 0))
            retention = _minimum_structural_retention(
                before, after, image_rgb)
            area = int(before.sum())
            vertices_before = _pallet_geometry_vertex_count(shape)
            vertices_after = _pallet_geometry_vertex_count(refined)
            gate_passed = (
                changed / max(area, 1) <= .06
                and retention >= .75
                and components_after <= components_before
                and (merge_count == 0
                     or (notch_count > 0
                         and merge_count
                         <= PALLET_IMAGE_BOUNDARY_MAXIMUM_COMPONENT_MERGES))
            )
            results.append({
                "frame": frame,
                "shape_id": shape.get("id"),
                "pixel_area": area,
                "geometry_cleanup_version": adjustment.get(
                    "geometry_cleanup_version"),
                "boundary_simplification_epsilon": adjustment.get(
                    "boundary_simplification_epsilon"),
                "boundary_simplification_scale_band": adjustment.get(
                    "boundary_simplification_scale_band"),
                "boundary_vertices_before": vertices_before,
                "boundary_vertices_after": vertices_after,
                "boundary_vertex_reduction_ratio": round(
                    1 - vertices_after / max(vertices_before, 1), 6),
                "changed_pixels": changed,
                "change_ratio": round(changed / max(area, 1), 6),
                "filled_image_boundary_notch_count": notch_count,
                "filled_image_boundary_notch_pixels": int(adjustment.get(
                    "filled_image_boundary_notch_pixels", 0)),
                "components_before": components_before,
                "components_after": components_after,
                "component_merge_count": merge_count,
                "minimum_structural_hole_retention": round(retention, 6),
                "gate_passed": gate_passed,
            })
    violations = [row for row in results if not row["gate_passed"]]
    adjusted = [row for row in results if row["changed_pixels"] > 0]
    report = {
        "schema_version": "labelagent.pallet_boundary_notch_collection_audit.v1",
        "workbench": base_url,
        "path": bootstrap["path"],
        "topic": bootstrap["topic"],
        "file_activation_generation": bootstrap["file_activation_generation"],
        "total_frame_count": total_frame_count,
        "frame_range": [start_frame, end_frame],
        "frame_count": end_frame - start_frame,
        "candidate_count": sum(row["candidate_count"] for row in frame_inventory),
        "pallet_candidate_count": len(results),
        "adjusted_count": len(adjusted),
        "v6_adjusted_count": sum(
            row["geometry_cleanup_version"] == 6 for row in adjusted),
        "filled_image_boundary_notch_count": sum(
            row["filled_image_boundary_notch_count"] for row in results),
        "component_merge_count": sum(
            row["component_merge_count"] for row in results),
        "boundary_vertices_before": sum(
            row["boundary_vertices_before"] for row in results),
        "boundary_vertices_after": sum(
            row["boundary_vertices_after"] for row in results),
        "boundary_vertex_reduction_ratio": round(
            1 - sum(row["boundary_vertices_after"] for row in results)
            / max(sum(row["boundary_vertices_before"] for row in results), 1), 6),
        "maximum_change_ratio": max(
            (row["change_ratio"] for row in results), default=0),
        "minimum_structural_hole_retention": min(
            (row["minimum_structural_hole_retention"] for row in results),
            default=1),
        "maximum_component_merge_count": max(
            (row["component_merge_count"] for row in results), default=0),
        "violation_count": len(violations),
        "gate_passed": not violations,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "top_changed": sorted(
            adjusted, key=lambda row: row["change_ratio"], reverse=True)[:30],
        "violations": violations,
        "frame_inventory": frame_inventory,
        "results": results,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_collection(
        args.base_url, start_frame=args.start_frame, end_frame=args.end_frame)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    summary = {key: report[key] for key in (
        "frame_count", "candidate_count", "pallet_candidate_count",
        "adjusted_count", "v6_adjusted_count",
        "filled_image_boundary_notch_count", "component_merge_count",
        "boundary_vertices_before", "boundary_vertices_after",
        "boundary_vertex_reduction_ratio",
        "maximum_change_ratio", "minimum_structural_hole_retention",
        "maximum_component_merge_count", "violation_count", "gate_passed",
    )}
    if args.output:
        summary["output_sha256"] = hashlib.sha256(
            args.output.read_bytes()).hexdigest()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
