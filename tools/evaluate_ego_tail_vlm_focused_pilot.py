#!/usr/bin/env python3
"""Run a bounded VLM pilot with a square, target-focused review montage."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time

from PIL import Image, ImageDraw, ImageFont

try:
    from tools.evaluate_ego_tail_vlm_guard import (
        SCHEMA,
        file_sha256,
        request_json,
        select_camera1_tail,
        validate_vlm_decision,
    )
except ModuleNotFoundError:  # Direct execution adds tools/, not the repo root.
    from evaluate_ego_tail_vlm_guard import (  # type: ignore[no-redef]
        SCHEMA,
        file_sha256,
        request_json,
        select_camera1_tail,
        validate_vlm_decision,
    )


SOURCE_SIZE = (1240, 360)
PANEL_BOXES = {
    "context_overlay": (375, 76, 630, 331),
    "raw_micro": (655, 72, 915, 332),
    "overlay_micro": (955, 72, 1215, 332),
}

PROMPT = """You are independently reviewing one tiny machine-generated fill on an ego-vehicle segmentation mask.

The square review montage keeps the target large. Top left is a 96-pixel context overlay. Top right is a raw 32-pixel micro crop without segmentation color. Bottom left is the same micro crop with the proposed tiny fill highlighted in magenta and an outside yellow locator. Judge the physical evidence in the raw and context panels, not the overlay color.

Choose likely_fill_ok only when the highlighted tiny region clearly lies on the same solid ego-vehicle surface and only closes a segmentation pinhole. Choose likely_reject when it crosses a real mechanical opening, slot, gap, background, or object boundary. Choose uncertain whenever the target is dark, blurred, tiny, occluded, or physically ambiguous. Be conservative: unclear evidence is uncertain, never likely_fill_ok.

This is assistant shadow review only. Return the requested JSON."""


def _fit_square(image: Image.Image, side: int = 350) -> Image.Image:
    copy = image.convert("RGB")
    copy = copy.resize((side, side), Image.Resampling.NEAREST)
    return copy


def extract_review_panels(image: Image.Image) -> dict[str, Image.Image]:
    """Extract the exact rendered panels and fail closed on layout drift."""
    if image.size != SOURCE_SIZE:
        raise ValueError(f"unexpected component review size: {image.size}")
    return {name: image.crop(box).convert("RGB")
            for name, box in PANEL_BOXES.items()}


def build_focused_montage(image_path: Path) -> bytes:
    """Create a square 2x2 montage so micro panels survive server resizing."""
    with Image.open(image_path) as image:
        panels = extract_review_panels(image)
    canvas = Image.new("RGB", (768, 768), (9, 15, 22))
    placements = {
        "context_overlay": (16, 48),
        "raw_micro": (402, 48),
        "overlay_micro": (16, 418),
    }
    labels = {
        "context_overlay": "1 CONTEXT OVERLAY",
        "raw_micro": "2 RAW MICRO - PHYSICAL EVIDENCE",
        "overlay_micro": "3 OVERLAY MICRO - TARGET LOCATOR",
    }
    for name, position in placements.items():
        canvas.paste(_fit_square(panels[name]), position)
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    draw.text((16, 14), "FOCUSED EGO MICRO-HOLE REVIEW",
              fill=(245, 245, 248), font=font)
    draw.text((16, 34), labels["context_overlay"],
              fill=(255, 220, 40), font=font)
    draw.text((402, 34), labels["raw_micro"],
              fill=(255, 220, 40), font=font)
    draw.text((16, 404), labels["overlay_micro"],
              fill=(255, 220, 40), font=font)
    draw.multiline_text(
        (402, 430),
        "DECISION RULE\n\n"
        "Use raw + context evidence.\n"
        "Magenta only locates the target.\n\n"
        "UNCLEAR => uncertain\n"
        "REAL GAP => likely_reject\n"
        "CLEAR SOLID PINHOLE => likely_fill_ok",
        fill=(220, 228, 236), font=font, spacing=9)
    payload = io.BytesIO()
    canvas.save(payload, "PNG", optimize=True)
    return payload.getvalue()


def select_pilot_rows(policy: dict, component_review: dict,
                      component_ids: list[str]) -> list[dict]:
    if not component_ids or len(component_ids) != len(set(component_ids)):
        raise ValueError("pilot component ids must be unique and non-empty")
    tail = {row["component_id"]: row
            for row in select_camera1_tail(policy, component_review)}
    missing = [component_id for component_id in component_ids
               if component_id not in tail]
    if missing:
        raise ValueError(f"pilot component is outside safe tail: {missing}")
    return [tail[component_id] for component_id in component_ids]


def evaluate(policy_path: Path, component_review_path: Path, output: Path,
             focused_image_dir: Path, base: str, model: str,
             component_ids: list[str]) -> dict:
    if output.exists() or focused_image_dir.exists():
        raise FileExistsError("pilot output already exists")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    component_review = json.loads(
        component_review_path.read_text(encoding="utf-8"))
    selected = select_pilot_rows(policy, component_review, component_ids)
    health_url = base.rsplit("/v1", 1)[0].rstrip("/") + "/health"
    health_before = request_json(health_url)
    if (health_before.get("status") != "ok"
            or health_before.get("model_loaded") is not True
            or health_before.get("exact_requests_inflight") != 0):
        raise RuntimeError("VLM is not idle and ready")

    temp_dir = focused_image_dir.with_name(
        f".{focused_image_dir.name}.tmp-{os.getpid()}")
    if temp_dir.exists():
        raise FileExistsError(temp_dir)
    temp_dir.mkdir(parents=True)
    rows = []
    try:
        for item in selected:
            source_path = component_review_path.parent / item["review_image"]
            if file_sha256(source_path) != item["review_image_sha256"]:
                raise ValueError(
                    f"review image hash mismatch: {item['component_id']}")
            focused_bytes = build_focused_montage(source_path)
            focused_name = f"{item['component_id']}-focused.png"
            focused_path = temp_dir / focused_name
            focused_path.write_bytes(focused_bytes)
            focused_sha = hashlib.sha256(focused_bytes).hexdigest()
            payload = {
                "model": model,
                "temperature": 0,
                "max_tokens": 180,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url":
                        "data:image/png;base64," + base64.b64encode(
                            focused_bytes).decode("ascii")}},
                ]}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "ego_tail_focused_guard", "schema": SCHEMA,
                    "strict": True}},
            }
            started = time.perf_counter()
            raw = request_json(base.rstrip("/") + "/chat/completions", payload)
            raw_value = json.loads(raw["choices"][0]["message"]["content"])
            validation_error = None
            try:
                value = validate_vlm_decision(raw_value)
                effective_decision = value["decision"]
                status = "vlm_shadow_candidate_generated"
            except ValueError as error:
                value = raw_value
                effective_decision = "uncertain"
                status = "vlm_logical_conflict_abstained"
                validation_error = str(error)
            rows.append({
                **item,
                "focused_image": focused_name,
                "focused_image_sha256": focused_sha,
                "status": status,
                "vlm": value,
                "effective_vlm_decision": effective_decision,
                "vlm_supports_deferral": effective_decision == "likely_fill_ok",
                "validation_error": validation_error,
                "wall_seconds": round(time.perf_counter() - started, 3),
                "usage": raw.get("usage"),
                "local_vlm": raw.get("local_vlm"),
            })
            print(json.dumps({
                "component_id": item["component_id"],
                "effective_vlm_decision": effective_decision,
                "wall_seconds": rows[-1]["wall_seconds"],
            }, ensure_ascii=False), flush=True)
        health_after = request_json(health_url)
        packet = {
            "schema_version": "labelagent.ego_tail_vlm_focused_pilot.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_only": True,
            "assistant_review_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "production_deployed": False,
            "default_enabled": False,
            "automatic_decisions_enabled": False,
            "pilot_only": True,
            "policy_sha256": file_sha256(policy_path),
            "component_review_sha256": file_sha256(component_review_path),
            "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
            "model": model,
            "review_count": len(rows),
            "vlm_likely_fill_ok": sum(
                row["effective_vlm_decision"] == "likely_fill_ok"
                for row in rows),
            "vlm_uncertain": sum(
                row["effective_vlm_decision"] == "uncertain" for row in rows),
            "vlm_likely_reject": sum(
                row["effective_vlm_decision"] == "likely_reject"
                for row in rows),
            "logical_conflict_abstentions": sum(
                row["status"] == "vlm_logical_conflict_abstained"
                for row in rows),
            "rows": rows,
            "health_before": {
                "status": health_before.get("status"),
                "exact_requests_inflight": health_before.get(
                    "exact_requests_inflight"),
            },
            "health_after": {
                "status": health_after.get("status"),
                "exact_requests_inflight": health_after.get(
                    "exact_requests_inflight"),
            },
            "boundary": (
                "This bounded focused-input result is assistant shadow triage, "
                "not truth, training approval, automatic acceptance, or a "
                "generalization claim."),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
        try:
            temporary.write_text(
                json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            readback = json.loads(temporary.read_text(encoding="utf-8"))
            if readback["review_count"] != len(rows):
                raise ValueError("focused pilot readback mismatch")
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        temp_dir.replace(focused_image_dir)
        return packet
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--component-review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--focused-image-dir", required=True, type=Path)
    parser.add_argument("--component-id", required=True, action="append")
    parser.add_argument("--base", default="http://127.0.0.1:28868/v1")
    parser.add_argument("--model", default="labelagent-vlm8b")
    arguments = parser.parse_args()
    packet = evaluate(
        arguments.policy, arguments.component_review, arguments.output,
        arguments.focused_image_dir, arguments.base, arguments.model,
        arguments.component_id)
    print(json.dumps({key: packet[key] for key in (
        "review_count", "vlm_likely_fill_ok", "vlm_uncertain",
        "vlm_likely_reject", "logical_conflict_abstentions")},
        ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
