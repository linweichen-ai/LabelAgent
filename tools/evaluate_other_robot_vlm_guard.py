"""Evaluate a high-precision 8B other_robot review guard on shadow candidates."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request


PROMPT = (
    "Act as a high-precision verification gate for the project label other_robot. "
    "Search the entire image for ANY physically separate robot before classifying. "
    "When a second image is provided, it is a magnified candidate ROI from the same frame; "
    "use the first image for context and the second for target detail. "
    "The camera-carrying ego vehicle may be visible in the foreground at the same time; "
    "do not choose ego_robot_part when another independent robot is also visible. "
    "Classify the evidence into exactly one scene_class, giving priority to "
    "separate_unmanned_robot whenever a separate robot is visibly supported. "
    "separate_unmanned_robot means a physically separate autonomous-looking mobile robot "
    "or industrial robot. ego_robot_part means only the camera-carrying vehicle, its forks, "
    "carriage, chassis, or reflection. conventional_manned_forklift means a normal operator "
    "forklift or pallet truck; do not infer autonomy merely from forks, a mast, or a chassis. "
    "no_robot_evidence means pallets, cargo, people, floor, or tools only. Use uncertain when "
    "identity or autonomy cannot be supported visually. other_robot_evidence must be present "
    "only for separate_unmanned_robot, absent for ego_robot_part, "
    "conventional_manned_forklift, or no_robot_evidence, and uncertain otherwise."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "scene_class": {"type": "string", "enum": [
            "separate_unmanned_robot", "ego_robot_part",
            "conventional_manned_forklift", "no_robot_evidence", "uncertain"]},
        "other_robot_evidence": {"type": "string",
                                 "enum": ["present", "absent", "uncertain"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "minLength": 1, "maxLength": 160},
    },
    "required": ["scene_class", "other_robot_evidence", "confidence", "reason"],
    "additionalProperties": False,
}


def expected_evidence(scene_class: str) -> str:
    if scene_class == "separate_unmanned_robot":
        return "present"
    if scene_class in {"ego_robot_part", "conventional_manned_forklift",
                       "no_robot_evidence"}:
        return "absent"
    if scene_class == "uncertain":
        return "uncertain"
    raise ValueError("guard scene_class is invalid")


def validate_output(value: dict) -> dict:
    required = set(SCHEMA["required"])
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("guard output keys differ")
    if value["scene_class"] not in SCHEMA["properties"]["scene_class"]["enum"]:
        raise ValueError("guard scene_class differs")
    if value["other_robot_evidence"] != expected_evidence(value["scene_class"]):
        raise ValueError("guard evidence contradicts scene_class")
    if not isinstance(value["confidence"], (int, float)) or not 0 <= value["confidence"] <= 1:
        raise ValueError("guard confidence differs")
    if not isinstance(value["reason"], str) or not value["reason"]:
        raise ValueError("guard reason differs")
    return value


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def crop_data_url(path: Path, roi: list[int] | None) -> str | None:
    if roi is None:
        return None
    if (not isinstance(roi, list) or len(roi) != 4
            or not all(isinstance(value, int) for value in roi)):
        raise ValueError("assistant ROI must be four integers")
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        left, top, right, bottom = roi
        if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
            raise ValueError("assistant ROI is outside the image")
        crop = image.crop((left, top, right, bottom))
    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def request_json(url: str, body: dict | None = None, timeout: int = 300) -> dict:
    request = urllib.request.Request(
        url, data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18878/v1")
    parser.add_argument("--model", default="labelagent-vlm8b")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    images = manifest.get("images")
    if (manifest.get("schema_version") != "labelagent.qwen3vl-shadow-manifest.v1"
            or manifest.get("human_truth") is not False
            or manifest.get("candidate_only") is not True
            or manifest.get("production_deployed") is not False
            or not isinstance(images, list) or not images):
        raise ValueError("guard shadow manifest boundary differs")
    health_url = args.base.rsplit("/v1", 1)[0] + "/health"
    health_before = request_json(health_url)
    if (health_before.get("status") != "ok"
            or health_before.get("model_loaded") is not True
            or health_before.get("exact_requests_inflight") != 0):
        raise RuntimeError("formal VLM is not idle and ready")
    rows = []
    for item in images:
        path = Path(item["image_path"])
        if file_sha256(path) != item["image_sha256"]:
            raise ValueError(f"guard image hash differs: {item['sample_id']}")
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        content = [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url":
                f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()}},
        ]
        roi = item.get("assistant_candidate_roi_xyxy")
        roi_url = crop_data_url(path, roi)
        if roi_url:
            content.append({"type": "image_url", "image_url": {"url": roi_url}})
        payload = {
            "model": args.model, "temperature": 0, "max_tokens": 160,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "other_robot_guard", "schema": SCHEMA, "strict": True}},
        }
        started = time.perf_counter()
        raw = request_json(args.base.rstrip("/") + "/chat/completions", payload)
        raw_value = json.loads(raw["choices"][0]["message"]["content"])
        validation_error = None
        try:
            value = validate_output(raw_value)
            guard_evidence = value["other_robot_evidence"]
            status = "guard_candidate_generated"
        except ValueError as error:
            value = raw_value
            guard_evidence = "uncertain"
            status = "guard_logical_conflict_abstained"
            validation_error = str(error)
        review = item.get("assistant_review_candidate")
        rows.append({
            "sample_id": item["sample_id"], "source_ref": item.get("source_ref"),
            "status": status, "human_truth": False,
            "candidate_only": True, "production_deployed": False,
            "assistant_review_candidate": review,
            "assistant_candidate_agreement": (
                guard_evidence == review["expected_evidence"]
                if isinstance(review, dict) else None),
            "guard": value, "effective_guard_evidence": guard_evidence,
            "assistant_candidate_roi_xyxy": roi,
            "validation_error": validation_error,
            "wall_seconds": round(time.perf_counter() - started, 3),
            "usage": raw.get("usage"), "local_vlm": raw.get("local_vlm"),
        })
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    health_after = request_json(health_url)
    reviewed = [row for row in rows if row["assistant_review_candidate"]]
    result = {
        "schema_version": "labelagent.other-robot-vlm-guard-shadow.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "evaluated", "human_truth": False, "candidate_only": True,
        "production_deployed": False, "accuracy_computed": False,
        "manifest_sha256": file_sha256(args.manifest),
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "model": args.model,
        "assistant_candidate_review": {
            "reviewed": len(reviewed),
            "agreements": sum(row["assistant_candidate_agreement"] is True for row in reviewed),
            "disagreements": sum(row["assistant_candidate_agreement"] is False for row in reviewed),
            "is_accuracy": False,
        },
        "logical_conflict_abstentions": sum(
            row["status"] == "guard_logical_conflict_abstained" for row in rows),
        "rows": rows,
        "health_before": {"status": health_before.get("status"),
                          "exact_requests_inflight": health_before.get("exact_requests_inflight")},
        "health_after": {"status": health_after.get("status"),
                         "exact_requests_inflight": health_after.get("exact_requests_inflight")},
        "boundary": "Assistant-reviewed shadow agreement is triage, not human accuracy.",
    }
    args.output.mkdir(parents=True)
    (args.output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
