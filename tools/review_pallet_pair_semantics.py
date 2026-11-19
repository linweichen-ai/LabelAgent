#!/usr/bin/env python3
"""Ask the configured VLM whether two pallet masks belong to one instance.

The output is assistant review evidence only.  It never edits candidates,
creates human truth, approves training data, or authorizes an automatic merge.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

from source.agent.workflow import _chat
from tools.audit_pallet_instance_fragmentation import (
    _draw_shape,
    _mapping,
    _parts,
)


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "same_physical_instance": {
            "type": "string", "enum": ["yes", "no", "uncertain"]},
        "smaller_is_redundant_fragment": {"type": "boolean"},
        "safe_to_consolidate": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "minLength": 1, "maxLength": 400},
    },
    "required": [
        "same_physical_instance", "smaller_is_redundant_fragment",
        "safe_to_consolidate", "confidence", "reason"],
    "additionalProperties": False,
}


PROMPT = """You are reviewing two machine-generated pallet masks from one image.
The first image is the unmarked crop. The second image overlays the larger mask
in green and the smaller mask in orange. Decide from visible physical object
boundaries, not color or geometric overlap alone.

Questions:
1. Do both masks refer to the same single physical pallet instance?
2. Is the orange mask only a redundant fragment/duplicate of the green pallet?
3. Would consolidating the orange candidate into the green candidate preserve
all visible pallet instances? Answer safe_to_consolidate=false whenever another
pallet, a person/cargo occlusion, disconnected regions, blur, or ambiguity could
make consolidation lose an object.

Return only the requested JSON. This is an assistant review, not human truth."""


def _jpeg(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.convert("RGB").save(output, "JPEG", quality=92)
    return output.getvalue()


def _review_images(image_path: Path, small: dict, large: dict) -> tuple[bytes, bytes]:
    image = Image.open(image_path).convert("RGB")
    points = [point for shape in (small, large) for part in _parts(shape)
              for point in (part.get("points") or [])]
    left = min(float(point[0]) for point in points)
    top = min(float(point[1]) for point in points)
    right = max(float(point[0]) for point in points)
    bottom = max(float(point[1]) for point in points)
    pad = max(28, round(max(right - left, bottom - top) * .28))
    crop = (max(0, int(left) - pad), max(0, int(top) - pad),
            min(image.width, int(right) + pad + 1),
            min(image.height, int(bottom) + pad + 1))
    original = image.crop(crop)
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    _draw_shape(layer, large, (38, 217, 133, 72), (38, 217, 133, 255))
    _draw_shape(layer, small, (255, 148, 32, 96), (255, 148, 32, 255))
    overlay = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB").crop(crop)
    for value in (original, overlay):
        value.thumbnail((768, 768), Image.Resampling.LANCZOS)
    return _jpeg(original), _jpeg(overlay)


def _shape_lookup(payload: dict, topic: str, frame: int, shape_id) -> dict:
    mapping = _mapping(payload)
    frames = mapping.get(topic)
    if not isinstance(frames, dict):
        raise ValueError(f"topic missing from candidate payload: {topic}")
    shapes = frames.get(str(frame), frames.get(frame))
    if not isinstance(shapes, list):
        raise ValueError(f"frame missing from candidate payload: {topic} frame {frame}")
    matches = [shape for shape in shapes if str(shape.get("id")) == str(shape_id)]
    if len(matches) != 1:
        raise ValueError(f"candidate ID is not unique: {shape_id}")
    return matches[0]


def _validate_decision(decision: dict) -> dict:
    if not isinstance(decision, dict):
        raise ValueError("VLM decision must be an object")
    if decision.get("same_physical_instance") not in {"yes", "no", "uncertain"}:
        raise ValueError("invalid same_physical_instance")
    if any(type(decision.get(key)) is not bool for key in (
            "smaller_is_redundant_fragment", "safe_to_consolidate")):
        raise ValueError("VLM boolean decision fields are invalid")
    confidence = decision.get("confidence")
    if (type(confidence) not in (int, float) or isinstance(confidence, bool)
            or not 0 <= confidence <= 1):
        raise ValueError("VLM confidence is invalid")
    reason = decision.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 400:
        raise ValueError("VLM reason is invalid")
    if (decision["safe_to_consolidate"]
            and (decision["same_physical_instance"] != "yes"
                 or not decision["smaller_is_redundant_fragment"]
                 or confidence < .9)):
        raise ValueError("unsafe semantic decision combination")
    return decision


def review_pairs(candidate_file: Path, pair_report: Path, output_dir: Path,
                 ranks: list[int], config: dict, *, chat=_chat) -> dict:
    candidate_before = candidate_file.read_bytes()
    payload = json.loads(candidate_before.decode("utf-8"))
    pair_bytes = pair_report.read_bytes()
    pair_payload = json.loads(pair_bytes.decode("utf-8"))
    by_rank = {int(row["rank"]): row for row in pair_payload.get("rows") or []}
    if not ranks or len(set(ranks)) != len(ranks) or any(rank not in by_rank for rank in ranks):
        raise ValueError("ranks must be unique and present in the pair review")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "review-images"
    image_dir.mkdir(exist_ok=True)
    results = []
    for rank in ranks:
        row = by_rank[rank]
        small = _shape_lookup(
            payload, row["topic"], int(row["frame"]), row["small_id"])
        large = _shape_lookup(
            payload, row["topic"], int(row["frame"]), row["large_id"])
        original, overlay = _review_images(Path(row["image_path"]), small, large)
        original_path = image_dir / f"rank-{rank:03d}-original.jpg"
        overlay_path = image_dir / f"rank-{rank:03d}-overlay.jpg"
        original_path.write_bytes(original)
        overlay_path.write_bytes(overlay)
        trace = {}
        decision = _validate_decision(chat(
            config, PROMPT, original, extra_images=(overlay,), raw=trace,
            response_schema=RESPONSE_SCHEMA))
        results.append({
            "rank": rank, "topic": row["topic"], "frame": row["frame"],
            "small_id": row["small_id"], "large_id": row["large_id"],
            "geometry_gate": row["conservative_duplicate_gate"],
            "decision": decision, "trace": trace,
            "original_sha256": hashlib.sha256(original).hexdigest(),
            "overlay_sha256": hashlib.sha256(overlay).hexdigest(),
        })
    if candidate_file.read_bytes() != candidate_before:
        raise RuntimeError("candidate source changed during semantic review")
    report = {
        "schema_version": "labelagent.pallet_pair_semantic_review.v1",
        "candidate_file": str(candidate_file.resolve()),
        "candidate_sha256": hashlib.sha256(candidate_before).hexdigest(),
        "pair_report": str(pair_report.resolve()),
        "pair_report_sha256": hashlib.sha256(pair_bytes).hexdigest(),
        "ranks": ranks,
        "review_count": len(results),
        "same_instance_yes_count": sum(
            row["decision"]["same_physical_instance"] == "yes" for row in results),
        "semantic_safe_count": sum(
            row["decision"]["safe_to_consolidate"] for row in results),
        "assistant_review_only": True,
        "automatic_merge_allowed": False,
        "human_truth": False,
        "approved_for_training": False,
        "results": results,
    }
    report_path = output_dir / "pallet-pair-semantic-review.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path.resolve())
    report["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--pair-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", required=True,
                        help="Comma-separated one-based ranks from the pair report")
    parser.add_argument("--url", default="http://127.0.0.1:28868/v1")
    parser.add_argument("--model", default="labelagent-vlm8b")
    args = parser.parse_args()
    ranks = [int(value) for value in args.ranks.split(",") if value.strip()]
    report = review_pairs(
        args.candidates, args.pair_report, args.output, ranks,
        {"llm_url": args.url, "llm_model": args.model, "llm_key": "",
         "llm_prompt_profile": "compact", "llm_max_tokens": 384,
         "llm_busy_retries": 5})
    print(json.dumps({key: report[key] for key in (
        "review_count", "same_instance_yes_count", "semantic_safe_count",
        "report_path", "report_sha256", "automatic_merge_allowed")},
        ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
