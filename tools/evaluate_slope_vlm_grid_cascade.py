"""Evaluate grid-grounded VLM localization followed by SAM point segmentation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import string
import sys
import time

from PIL import Image, ImageDraw, ImageFont

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from source.agent.workflow import _shape_summary, candidate_overlay, chat, sam_request
from tools.atomic_progress import save_json
from tools.evaluate_slope_static_images import file_sha256, load_inputs


GRID_SIZE = 8
CELL_IDS = [f"{column}{row}" for row in range(1, GRID_SIZE + 1)
            for column in string.ascii_uppercase[:GRID_SIZE]]
GRID_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "slope_visible": {"type": "boolean"},
        "cell_id": {"type": "string", "enum": ["NONE", *CELL_IDS]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["slope_visible", "cell_id", "confidence", "reason"],
}
GRID_INSTRUCTION = """The image has an 8x8 reference grid. Columns are A-H from left to right and rows are 1-8 from top to bottom. Each cell contains a small central cross.
Inspect the factory image for a visibly inclined ramp or transition surface. Judge by surface geometry and inclination, not by any specific color. Ordinary level floor, pallets, robot decks, vehicle bodies, fork carriages, and horizontal beams are not ramps.
If a ramp is visible, choose exactly one cell whose central cross lies clearly inside visible ramp pixels, away from boundaries and occluders. Return slope_visible=true and that cell_id. If no central cross lies safely inside a visibly supported ramp, return slope_visible=false and cell_id=NONE. Give one short visual reason. Return only the required JSON object."""

DUAL_IMAGE_INSTRUCTION = """The two images show the exact same 640x512 factory view. Image 1 is the unobstructed original. Image 2 adds an 8x8 reference grid: columns A-H from left to right and rows 1-8 from top to bottom, with a small central cross in each cell.
First inspect Image 1 for a visibly inclined ramp or transition surface. Judge by surface geometry and inclination, not by any specific color. Ordinary level floor, pallets, robot decks, vehicle bodies, fork carriages, and horizontal beams are not ramps. Then use Image 2 only to locate that same visible surface.
If a ramp is visible, choose exactly one cell whose central cross lies clearly inside visible ramp pixels in both images, away from boundaries and occluders. Return slope_visible=true and that cell_id. If no central cross lies safely inside a visibly supported ramp, return slope_visible=false and cell_id=NONE. Give one short visual reason. Return only the required JSON object."""


def cell_center(cell_id: str, width: int = 640, height: int = 512) -> tuple[int, int]:
    if cell_id not in CELL_IDS:
        raise ValueError("invalid grid cell")
    column = ord(cell_id[0]) - ord("A")
    row = int(cell_id[1:]) - 1
    return (round((column + 0.5) * width / GRID_SIZE),
            round((row + 0.5) * height / GRID_SIZE))


def grid_overlay(image: bytes) -> bytes:
    source = Image.open(io.BytesIO(image)).convert("RGB")
    if source.size != (640, 512):
        raise ValueError("grid cascade expects exact 640x512 images")
    overlay = Image.new("RGBA", source.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    for index in range(1, GRID_SIZE):
        x = round(index * source.width / GRID_SIZE)
        y = round(index * source.height / GRID_SIZE)
        draw.line((x, 0, x, source.height), fill=(255, 255, 255, 135), width=1)
        draw.line((0, y, source.width, y), fill=(255, 255, 255, 135), width=1)
    for cell_id in CELL_IDS:
        x, y = cell_center(cell_id, source.width, source.height)
        draw.line((x - 4, y, x + 4, y), fill=(255, 255, 255, 230), width=1)
        draw.line((x, y - 4, x, y + 4), fill=(255, 255, 255, 230), width=1)
        box = draw.textbbox((0, 0), cell_id, font=font)
        text_width, text_height = box[2] - box[0], box[3] - box[1]
        label_x, label_y = x - text_width // 2, y - text_height - 5
        draw.rectangle((label_x - 1, label_y - 1, label_x + text_width,
                        label_y + text_height), fill=(0, 0, 0, 150))
        draw.text((label_x, label_y), cell_id, fill=(255, 255, 255, 255), font=font)
    marked = Image.alpha_composite(source.convert("RGBA"), overlay).convert("RGB")
    output = io.BytesIO()
    marked.save(output, "JPEG", quality=95)
    return output.getvalue()


def validate_decision(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != set(GRID_SCHEMA["required"]):
        raise ValueError("VLM grid decision has invalid fields")
    visible, cell_id = value["slope_visible"], value["cell_id"]
    confidence, reason = value["confidence"], value["reason"]
    if (type(visible) is not bool or cell_id not in ["NONE", *CELL_IDS]
            or type(confidence) not in (int, float) or isinstance(confidence, bool)
            or not 0 <= confidence <= 1 or not isinstance(reason, str)
            or not reason.strip() or len(reason) > 300):
        raise ValueError("VLM grid decision values are invalid")
    if visible != (cell_id != "NONE"):
        raise ValueError("VLM grid presence and cell disagree")
    return {"slope_visible": visible, "cell_id": cell_id,
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


def json_sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def load_completed_row(path: Path, source: dict, bindings: dict,
                       output: Path) -> dict:
    row = json.loads(path.read_text(encoding="utf-8"))
    if (row.get("status") != "succeeded"
            or row.get("sample_id") != source["sample_id"]
            or row.get("image_sha256") != source["image_sha256"]
            or row.get("bindings") != bindings
            or not (output / row.get("grid_image", "")).is_file()
            or not (output / row.get("overlay", "")).is_file()):
        raise ValueError("completed grid row binding or output differs")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--sam-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dual-image", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    if not args.sam_url.startswith("http://127.0.0.1:"):
        raise ValueError("SAM URL must be a loopback endpoint")
    inputs, _ = load_inputs(args.dataset_dir, args.review, args.images_dir)
    state = json.loads(args.state.read_text(encoding="utf-8"))
    config = dict(state.get("config") or {})
    config["sam3_url"] = args.sam_url.rstrip("/")
    for key in ("llm_fast_url", "llm_fast_model", "llm_fast_key"):
        config.pop(key, None)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    (args.output / "rows").mkdir(exist_ok=args.resume)
    (args.output / "failures").mkdir(exist_ok=args.resume)
    instruction = DUAL_IMAGE_INSTRUCTION if args.dual_image else GRID_INSTRUCTION
    bindings = {
        "state_sha256": file_sha256(args.state),
        "review_sha256": file_sha256(args.review),
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "response_schema_sha256": json_sha256(GRID_SCHEMA),
        "dual_image_grounding": args.dual_image,
    }
    rows = []
    failures = []
    resumed_rows = 0
    for source in inputs:
        row_path = args.output / "rows" / f"{source['sample_id']}.json"
        if args.resume and row_path.exists():
            row = load_completed_row(row_path, source, bindings, args.output)
            rows.append(row)
            resumed_rows += 1
            print(json.dumps({"sample_id": row["sample_id"], "resumed": True,
                              "progress": f"{len(rows)}/{len(inputs)}"},
                             ensure_ascii=False), flush=True)
            continue
        image = source["image_path"].read_bytes()
        marked = grid_overlay(image)
        marked_path = args.output / f"{source['sample_id']}-grid.jpg"
        marked_path.write_bytes(marked)
        started = time.monotonic()
        try:
            decision = validate_decision(chat(
                config, instruction, image if args.dual_image else marked,
                extra_images=(marked,) if args.dual_image else (),
                response_schema=GRID_SCHEMA))
        except Exception as error:
            failure = {
                "schema_version": "labelagent.slope-vlm-grid-cascade-failure.v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "failed_retryable", "sample_id": source["sample_id"],
                "image_sha256": source["image_sha256"], "bindings": bindings,
                "error_type": type(error).__name__, "error": str(error)[:1000],
                "elapsed_seconds": round(time.monotonic() - started, 4),
            }
            failures.append(failure)
            failure_path = args.output / "failures" / (
                f"{source['sample_id']}-{len(list((args.output / 'failures').glob(source['sample_id'] + '-*.json'))) + 1:03d}.json")
            save_json(failure_path, failure)
            print(json.dumps(failure, ensure_ascii=False), flush=True)
            continue
        vlm_elapsed = time.monotonic() - started
        point = cell_center(decision["cell_id"]) if decision["slope_visible"] else None
        shape = None
        sam_error = None
        sam_elapsed = 0.0
        if point:
            started = time.monotonic()
            try:
                result = sam_request(config, "/segment_points", image,
                                     points=[{"x": point[0], "y": point[1], "label": 1}])
                shape = {"id": 1, "label": "slope",
                         "points": result.get("points") or [],
                         "holes": result.get("holes") or [],
                         "parts": result.get("parts") or [],
                         "score": result.get("score")}
            except Exception as error:
                sam_error = str(error)[:500]
            sam_elapsed = time.monotonic() - started
        overlay = args.output / f"{source['sample_id']}-grid-sam.jpg"
        overlay.write_bytes(candidate_overlay(image, [shape] if shape else []))
        row = {
            "schema_version": "labelagent.slope-vlm-grid-cascade-row.v1",
            "status": "succeeded", "bindings": bindings,
            "sample_id": source["sample_id"],
            "image_sha256": source["image_sha256"],
            "grid_image_sha256": file_sha256(marked_path),
            "assistant_visible_slope_candidate": source[
                "assistant_visible_slope_candidate"],
            "assistant_view": source["assistant_view"],
            "vlm_decision": decision,
            "mapped_point": list(point) if point else None,
            "vlm_elapsed_seconds": round(vlm_elapsed, 4),
            "sam_candidate": (_shape_summary(shape) | {"score": shape.get("score")}
                              if shape else None),
            "sam_error": sam_error,
            "sam_elapsed_seconds": round(sam_elapsed, 4),
            "grid_image": marked_path.name, "overlay": overlay.name,
        }
        rows.append(row)
        save_json(row_path, row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    base_summary = summarize(rows)
    base_summary.update({"expected_samples": len(inputs),
                         "completed_samples": len(rows),
                         "failed_samples": len(failures),
                         "resumed_rows": resumed_rows})
    report = {
        "schema_version": "labelagent.slope-vlm-grid-cascade-eval.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "assistant_review_candidate_only",
        "split": "dev8", "source_sequence": "008",
        "human_truth": False, "human_confirmed": False,
        "candidate_only": True, "production_deployed": False,
        "color_dependent_prompt": False,
        "dual_image_grounding": args.dual_image,
        "grid_size": GRID_SIZE,
        "bindings": bindings,
        "instruction": (DUAL_IMAGE_INSTRUCTION if args.dual_image else GRID_INSTRUCTION),
        "response_schema": GRID_SCHEMA,
        "summary": base_summary, "rows": rows, "failures": failures,
        "evidence_note": (
            "Grid-grounded VLM localization is compared with bounded assistant visual "
            "review. SAM overlays remain machine candidates, not human truth or IoU."),
    }
    save_json(args.output / "results.json", report)
    print(json.dumps({"output": str(args.output.resolve()),
                      "summary": report["summary"]}, ensure_ascii=False))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
