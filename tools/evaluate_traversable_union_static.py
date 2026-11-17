"""Evaluate production floor-union-slope coverage on hash-bound dev8 images."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from source.agent.edit_agent import _rasterize
from source.agent.workflow import AnnotationAgent, _shape_summary, candidate_overlay
from tools.evaluate_slope_assistant_points import load_point_candidates
from tools.evaluate_slope_static_images import file_sha256, load_inputs


def summarize(rows: list[dict]) -> dict:
    reviewed = [row for row in rows if row["assistant_ramp_point"] is not None]
    return {
        "samples": len(rows),
        "assistant_reviewed_ramp_points": len(reviewed),
        "ramp_points_covered_by_traversable_union": sum(
            row["assistant_ramp_point_covered"] for row in reviewed),
        "ramp_points_covered_by_floor": sum(
            row["assistant_ramp_point_covered_by_floor"] for row in reviewed),
        "ramp_points_covered_by_slope": sum(
            row["assistant_ramp_point_covered_by_slope"] for row in reviewed),
        "images_with_floor_candidates": sum(row["floor_candidate_count"] > 0 for row in rows),
        "images_with_slope_candidates": sum(row["slope_candidate_count"] > 0 for row in rows),
        "elapsed_seconds": round(sum(row["elapsed_seconds"] for row in rows), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--sam-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.sam_url.startswith("http://127.0.0.1:"):
        raise ValueError("SAM URL must be a loopback endpoint")
    inputs, _ = load_inputs(args.dataset_dir, args.review, args.images_dir)
    pointed = load_point_candidates(args.points, inputs)
    positive_points = {
        row["sample_id"]: next(point for point in row["points"] if point["label"] == 1)
        for row in pointed
    }
    state = json.loads(args.state.read_text(encoding="utf-8"))
    config = dict(state.get("config") or {})
    config["sam3_url"] = args.sam_url.rstrip("/")
    args.output.mkdir(parents=True)
    rows = []
    for source in inputs:
        image = source["image_path"].read_bytes()
        events = []
        started = time.monotonic()
        agent = AnnotationAgent(config, image, {
            "description": "bounded dev8 traversable union evaluation",
            "prompts": [
                {"label": "pallet", "text": "pallet"},
                {"label": "floor", "text": "floor"},
                {"label": "slope", "text": "traversable surface"},
            ],
        }, ["pallet", "floor", "slope"], 1, on_progress=events.append)
        agent.generate()
        floors = [shape for shape in agent.shapes if shape.get("label") == "floor"]
        slopes = [shape for shape in agent.shapes if shape.get("label") == "slope"]
        union_mask = np.zeros((agent.height, agent.width), dtype=bool)
        floor_mask = np.zeros_like(union_mask)
        slope_mask = np.zeros_like(union_mask)
        for shape in floors:
            floor_mask |= _rasterize(shape, agent.width, agent.height).astype(bool)
        for shape in slopes:
            slope_mask |= _rasterize(shape, agent.width, agent.height).astype(bool)
        union_mask |= floor_mask | slope_mask
        point = positive_points.get(source["sample_id"])
        covered = (bool(union_mask[point["y"], point["x"]]) if point else None)
        covered_floor = (bool(floor_mask[point["y"], point["x"]]) if point else None)
        covered_slope = (bool(slope_mask[point["y"], point["x"]]) if point else None)
        union_shapes = [dict(shape, label="traversable_area")
                        for shape in [*floors, *slopes]]
        overlay = args.output / f"{source['sample_id']}-traversable-union.jpg"
        overlay.write_bytes(candidate_overlay(image, union_shapes))
        row = {
            "sample_id": source["sample_id"],
            "image_sha256": source["image_sha256"],
            "assistant_visible_slope_candidate": source[
                "assistant_visible_slope_candidate"],
            "assistant_ramp_point": point,
            "assistant_ramp_point_covered": covered,
            "assistant_ramp_point_covered_by_floor": covered_floor,
            "assistant_ramp_point_covered_by_slope": covered_slope,
            "floor_candidate_count": len(floors),
            "slope_candidate_count": len(slopes),
            "traversable_union_area_fraction": round(float(union_mask.mean()), 6),
            "floor_candidates": [_shape_summary(shape) for shape in floors],
            "slope_candidates": [_shape_summary(shape) for shape in slopes],
            "crop_fallback_used": any(
                event.get("operation") == "slope_overlap_crop_fallback"
                for event in events),
            "filter_reason_codes": [item.get("reason_code")
                                    for item in agent.candidate_filters],
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "overlay": overlay.name,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report = {
        "schema_version": "labelagent.traversable-union-static-eval.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "assistant_review_candidate_only",
        "split": "dev8", "source_sequence": "008",
        "human_truth": False, "human_confirmed": False,
        "candidate_only": True, "production_deployed": False,
        "business_mapping": {"operation": "union",
                             "source_labels": ["floor", "slope"],
                             "target_label": "traversable_area"},
        "state_sha256": file_sha256(args.state),
        "review_sha256": file_sha256(args.review),
        "points_sha256": file_sha256(args.points),
        "summary": summarize(rows), "rows": rows,
        "evidence_note": (
            "Coverage is measured only at assistant-selected visible ramp points. "
            "It is not human truth, full-mask IoU, obstacle safety, or deployment approval."),
    }
    (args.output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "summary": report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
