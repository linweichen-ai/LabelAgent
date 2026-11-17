"""Evaluate a geometry-prompted VLM point to SAM slope candidate cascade."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from source.agent.workflow import _shape_summary, candidate_overlay, chat, sam_request
from tools.evaluate_slope_static_images import file_sha256, load_inputs


POINT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "slope_visible": {"type": "boolean"},
        "point_x": {"type": "integer", "minimum": -1, "maximum": 639},
        "point_y": {"type": "integer", "minimum": -1, "maximum": 511},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["slope_visible", "point_x", "point_y", "confidence", "reason"],
}

POINT_INSTRUCTION = """Inspect this 640x512 factory image for a visibly inclined ramp or transition surface.
Judge by surface geometry and inclination, not by any specific color. A partially visible or occluded ramp counts only when visible pixels support an inclined surface. Ordinary level warehouse floor, pallets, robot decks, vehicle bodies, fork carriages, and horizontal beams are not ramps.
If a ramp is visible, return slope_visible=true and one integer pixel point well inside the visible ramp surface in the original image coordinates (x from 0 to 639, y from 0 to 511). The point must not lie on an edge, floor, cargo, vehicle, or container wall. If no ramp is visibly supported, return slope_visible=false and point_x=point_y=-1. Give one short visual reason. Return only the required JSON object."""


def validate_decision(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != set(POINT_SCHEMA["required"]):
        raise ValueError("VLM slope point decision has invalid fields")
    visible = value["slope_visible"]
    x, y, confidence, reason = (value["point_x"], value["point_y"],
                                value["confidence"], value["reason"])
    if (type(visible) is not bool or type(x) is not int or type(y) is not int
            or type(confidence) not in (int, float) or isinstance(confidence, bool)
            or not 0 <= confidence <= 1 or not isinstance(reason, str)
            or not reason.strip() or len(reason) > 300):
        raise ValueError("VLM slope point decision values are invalid")
    if visible and not (0 <= x < 640 and 0 <= y < 512):
        raise ValueError("visible slope requires an in-frame point")
    if not visible and (x != -1 or y != -1):
        raise ValueError("absent slope requires -1 coordinates")
    return {"slope_visible": visible, "point_x": x, "point_y": y,
            "confidence": float(confidence), "reason": reason.strip()}


def summarize(rows: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    for row in rows:
        expected = row["assistant_visible_slope_candidate"]
        predicted = row["vlm_decision"]["slope_visible"]
        tp += expected and predicted
        fp += (not expected) and predicted
        fn += expected and not predicted
        tn += (not expected) and not predicted
    return {
        "samples": len(rows), "true_positive_images": tp,
        "false_positive_images": fp, "false_negative_images": fn,
        "true_negative_images": tn,
        "precision_on_assistant_review": tp / (tp + fp) if tp + fp else None,
        "recall_on_assistant_review": tp / (tp + fn) if tp + fn else None,
        "sam_candidates_returned": sum(row["sam_candidate"] is not None for row in rows),
        "sam_failures": sum(row["sam_error"] is not None for row in rows),
        "vlm_elapsed_seconds": round(sum(row["vlm_elapsed_seconds"] for row in rows), 3),
        "sam_elapsed_seconds": round(sum(row["sam_elapsed_seconds"] for row in rows), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
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
    state = json.loads(args.state.read_text(encoding="utf-8"))
    config = dict(state.get("config") or {})
    config["sam3_url"] = args.sam_url.rstrip("/")
    for key in ("llm_fast_url", "llm_fast_model", "llm_fast_key"):
        config.pop(key, None)
    args.output.mkdir(parents=True)
    rows = []
    for source in inputs:
        image = source["image_path"].read_bytes()
        started = time.monotonic()
        decision = validate_decision(chat(
            config, POINT_INSTRUCTION, image, response_schema=POINT_SCHEMA))
        vlm_elapsed = time.monotonic() - started
        shape = None
        sam_error = None
        sam_elapsed = 0.0
        if decision["slope_visible"]:
            started = time.monotonic()
            try:
                result = sam_request(config, "/segment_points", image, points=[{
                    "x": decision["point_x"], "y": decision["point_y"], "label": 1}])
                shape = {"id": 1, "label": "slope",
                         "points": result.get("points") or [],
                         "holes": result.get("holes") or [],
                         "parts": result.get("parts") or [],
                         "score": result.get("score")}
            except Exception as error:
                sam_error = str(error)[:500]
            sam_elapsed = time.monotonic() - started
        overlay = args.output / f"{source['sample_id']}-vlm-point-sam.jpg"
        overlay.write_bytes(candidate_overlay(image, [shape] if shape else []))
        row = {
            "sample_id": source["sample_id"],
            "image_sha256": source["image_sha256"],
            "assistant_visible_slope_candidate": source[
                "assistant_visible_slope_candidate"],
            "assistant_view": source["assistant_view"],
            "vlm_decision": decision,
            "vlm_elapsed_seconds": round(vlm_elapsed, 4),
            "sam_candidate": (_shape_summary(shape) | {"score": shape.get("score")}
                              if shape else None),
            "sam_error": sam_error,
            "sam_elapsed_seconds": round(sam_elapsed, 4),
            "overlay": overlay.name,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report = {
        "schema_version": "labelagent.slope-vlm-point-cascade-eval.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "assistant_review_candidate_only",
        "split": "dev8", "source_sequence": "008",
        "human_truth": False, "human_confirmed": False,
        "candidate_only": True, "production_deployed": False,
        "color_dependent_prompt": False,
        "state_sha256": file_sha256(args.state),
        "review_sha256": file_sha256(args.review),
        "instruction": POINT_INSTRUCTION,
        "response_schema": POINT_SCHEMA,
        "summary": summarize(rows), "rows": rows,
        "evidence_note": (
            "VLM presence and point localization are compared with bounded assistant "
            "visual review. SAM overlays remain machine candidates, not human truth or IoU."),
    }
    (args.output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "summary": report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
