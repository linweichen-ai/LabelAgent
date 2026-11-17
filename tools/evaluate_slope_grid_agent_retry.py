"""Evaluate bounded VLM verification and one retry for grid-grounded slope masks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from source.agent.workflow import _shape_summary, candidate_overlay, chat, sam_request
from tools.atomic_progress import save_json
from tools.evaluate_slope_static_images import file_sha256, load_inputs
from tools.evaluate_slope_vlm_grid_cascade import (
    CELL_IDS, GRID_SCHEMA, cell_center, grid_overlay, validate_decision,
)


SIBLING_INSTRUCTION = """A synchronized camera in the same factory frame has a machine candidate for a ramp. This does not prove that a ramp is visible here. The image has an 8x8 grid, columns A-H and rows 1-8, with a central cross in each cell.
Inspect only this image for any visibly inclined ramp or transition surface. Judge by geometry and inclination, not by any specific color. Partial edge fragments count only when visible pixels support a ramp. Ordinary level floor, pallets, robot decks, vehicle bodies, fork carriages, and horizontal beams are not ramps.
If a ramp is visible and one central cross lies safely inside its visible pixels, return slope_visible=true and that cell. Otherwise return slope_visible=false and cell_id=NONE. Give one short visual reason and return only the required JSON object."""

VERIFY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "assessment": {"type": "string", "enum": [
            "accept_slope", "reject_not_slope", "uncertain"]},
        "alternate_cell_id": {"type": "string", "enum": ["NONE", *CELL_IDS]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["assessment", "alternate_cell_id", "confidence", "reason"],
}

VERIFY_INSTRUCTION = """The image shows a factory view with an 8x8 reference grid and a translucent red machine-candidate mask. Decide whether the red mask follows visible pixels of an inclined ramp or transition surface. Judge the underlying surface by geometry and inclination, not by its color. Ordinary level floor, pallets, robot decks, vehicle bodies, fork carriages, and horizontal beams are not ramps.
Use assessment=accept_slope only when the red mask mainly follows a visible ramp. If it masks another object or ordinary floor, use reject_not_slope. If evidence is insufficient, use uncertain. For a rejection, if another cell's central cross clearly lies inside a visible ramp, return that cell as alternate_cell_id; otherwise return NONE. For accept_slope or uncertain, alternate_cell_id must be NONE. Give one short visual reason and return only the required JSON object."""


def validate_verification(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != set(VERIFY_SCHEMA["required"]):
        raise ValueError("candidate verification has invalid fields")
    assessment = value["assessment"]
    cell_id = value["alternate_cell_id"]
    confidence, reason = value["confidence"], value["reason"]
    if (assessment not in VERIFY_SCHEMA["properties"]["assessment"]["enum"]
            or cell_id not in ["NONE", *CELL_IDS]
            or type(confidence) not in (int, float) or isinstance(confidence, bool)
            or not 0 <= confidence <= 1 or not isinstance(reason, str)
            or not reason.strip() or len(reason) > 300):
        raise ValueError("candidate verification values are invalid")
    if assessment != "reject_not_slope" and cell_id != "NONE":
        raise ValueError("only a rejected candidate may propose an alternate cell")
    return {"assessment": assessment, "alternate_cell_id": cell_id,
            "confidence": float(confidence), "reason": reason.strip()}


def frame_group(sample_id: str) -> str:
    marker = "_camera_"
    if marker not in sample_id:
        raise ValueError("sample id lacks camera suffix")
    return sample_id.split(marker, 1)[0]


def summarize(rows: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    for row in rows:
        expected = row["assistant_visible_slope_candidate"]
        predicted = row["final_candidate_accepted"]
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
        "accepted_candidates": sum(row["final_candidate_accepted"] for row in rows),
        "retry_masks_generated": sum(row["retry_mask_generated"] for row in rows),
        "vlm_elapsed_seconds": round(sum(row["vlm_elapsed_seconds"] for row in rows), 3),
        "sam_elapsed_seconds": round(sum(row["sam_elapsed_seconds"] for row in rows), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--source-result", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--sam-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.sam_url.startswith("http://127.0.0.1:"):
        raise ValueError("SAM URL must be a loopback endpoint")
    inputs, _ = load_inputs(args.dataset_dir, args.review, args.images_dir)
    source_report = json.loads(args.source_result.read_text(encoding="utf-8"))
    if (source_report.get("schema_version") !=
            "labelagent.slope-vlm-grid-cascade-eval.v1"
            or source_report.get("human_truth") is not False
            or source_report.get("candidate_only") is not True
            or len(source_report.get("rows") or []) != 8):
        raise ValueError("source grid result boundary differs")
    source_rows = {row["sample_id"]: row for row in source_report["rows"]}
    if set(source_rows) != {row["sample_id"] for row in inputs}:
        raise ValueError("source grid result samples differ")
    state = json.loads(args.state.read_text(encoding="utf-8"))
    config = dict(state.get("config") or {})
    config["sam3_url"] = args.sam_url.rstrip("/")
    for key in ("llm_fast_url", "llm_fast_model", "llm_fast_key"):
        config.pop(key, None)
    args.output.mkdir(parents=True)
    (args.output / "rows").mkdir()
    initial_positive_groups = {
        frame_group(sample_id) for sample_id, row in source_rows.items()
        if row["vlm_decision"]["slope_visible"]
    }
    rows = []
    for source in inputs:
        image = source["image_path"].read_bytes()
        initial = source_rows[source["sample_id"]]
        initial_positive = initial["vlm_decision"]["slope_visible"]
        candidate_shape = None
        candidate_overlay_bytes = None
        discovery = None
        first_verification = None
        retry_verification = None
        retry_mask_generated = False
        vlm_elapsed = sam_elapsed = 0.0
        if initial_positive:
            candidate_overlay_bytes = (args.source_result.parent / initial["overlay"]).read_bytes()
        elif frame_group(source["sample_id"]) in initial_positive_groups:
            started = time.monotonic()
            discovery = validate_decision(chat(
                config, SIBLING_INSTRUCTION, grid_overlay(image),
                response_schema=GRID_SCHEMA))
            vlm_elapsed += time.monotonic() - started
            if discovery["slope_visible"]:
                point = cell_center(discovery["cell_id"])
                started = time.monotonic()
                result = sam_request(config, "/segment_points", image,
                                     points=[{"x": point[0], "y": point[1], "label": 1}])
                sam_elapsed += time.monotonic() - started
                candidate_shape = {"id": 1, "label": "slope",
                                   "points": result.get("points") or [],
                                   "holes": result.get("holes") or [],
                                   "parts": result.get("parts") or [],
                                   "score": result.get("score")}
                candidate_overlay_bytes = candidate_overlay(image, [candidate_shape])
        if candidate_overlay_bytes is not None:
            started = time.monotonic()
            first_verification = validate_verification(chat(
                config, VERIFY_INSTRUCTION, grid_overlay(candidate_overlay_bytes),
                response_schema=VERIFY_SCHEMA))
            vlm_elapsed += time.monotonic() - started
            alternate = first_verification["alternate_cell_id"]
            if (first_verification["assessment"] == "reject_not_slope"
                    and alternate != "NONE"):
                point = cell_center(alternate)
                started = time.monotonic()
                result = sam_request(config, "/segment_points", image,
                                     points=[{"x": point[0], "y": point[1], "label": 1}])
                sam_elapsed += time.monotonic() - started
                candidate_shape = {"id": 1, "label": "slope",
                                   "points": result.get("points") or [],
                                   "holes": result.get("holes") or [],
                                   "parts": result.get("parts") or [],
                                   "score": result.get("score")}
                candidate_overlay_bytes = candidate_overlay(image, [candidate_shape])
                retry_mask_generated = True
                started = time.monotonic()
                retry_verification = validate_verification(chat(
                    config, VERIFY_INSTRUCTION, grid_overlay(candidate_overlay_bytes),
                    response_schema=VERIFY_SCHEMA))
                vlm_elapsed += time.monotonic() - started
        final_verification = retry_verification or first_verification
        accepted = bool(final_verification and
                        final_verification["assessment"] == "accept_slope")
        overlay_name = None
        if candidate_overlay_bytes is not None:
            overlay_name = f"{source['sample_id']}-final.jpg"
            (args.output / overlay_name).write_bytes(candidate_overlay_bytes)
        row = {
            "schema_version": "labelagent.slope-grid-agent-retry-row.v1",
            "sample_id": source["sample_id"],
            "image_sha256": source["image_sha256"],
            "assistant_visible_slope_candidate": source[
                "assistant_visible_slope_candidate"],
            "assistant_view": source["assistant_view"],
            "initial_grid_decision": initial["vlm_decision"],
            "sibling_probe": discovery,
            "first_verification": first_verification,
            "retry_verification": retry_verification,
            "retry_mask_generated": retry_mask_generated,
            "final_candidate_accepted": accepted,
            "final_candidate": (_shape_summary(candidate_shape) | {
                "score": candidate_shape.get("score")} if candidate_shape else
                initial.get("sam_candidate") if accepted else None),
            "vlm_elapsed_seconds": round(vlm_elapsed, 4),
            "sam_elapsed_seconds": round(sam_elapsed, 4),
            "overlay": overlay_name,
        }
        rows.append(row)
        save_json(args.output / "rows" / f"{source['sample_id']}.json", row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report = {
        "schema_version": "labelagent.slope-grid-agent-retry-eval.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "assistant_review_candidate_only",
        "split": "dev8", "source_sequence": "008",
        "human_truth": False, "human_confirmed": False,
        "candidate_only": True, "production_deployed": False,
        "color_dependent_prompt": False,
        "maximum_mask_retries": 1,
        "state_sha256": file_sha256(args.state),
        "review_sha256": file_sha256(args.review),
        "source_result_sha256": file_sha256(args.source_result),
        "sibling_instruction_sha256": hashlib.sha256(
            SIBLING_INSTRUCTION.encode("utf-8")).hexdigest(),
        "verification_instruction_sha256": hashlib.sha256(
            VERIFY_INSTRUCTION.encode("utf-8")).hexdigest(),
        "summary": summarize(rows), "rows": rows,
        "evidence_note": (
            "All VLM decisions and masks remain machine candidates compared with "
            "bounded assistant review, not human truth or pixel IoU."),
    }
    save_json(args.output / "results.json", report)
    print(json.dumps({"output": str(args.output.resolve()),
                      "summary": report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
