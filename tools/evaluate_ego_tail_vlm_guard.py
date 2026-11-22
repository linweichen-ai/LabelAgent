#!/usr/bin/env python3
"""Run a conservative VLM shadow review over the fully covered camera1 tail."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


PROMPT = """You are independently reviewing one tiny machine-generated fill on an ego-vehicle segmentation mask.
The review image contains a full-frame location view, a context crop, a raw close-up, and an overlay close-up. Judge the physical image evidence, not the overlay color.

Choose likely_fill_ok only when the highlighted tiny region clearly lies on the same solid ego-vehicle surface and only closes a segmentation pinhole. Choose likely_reject when it crosses a real mechanical opening, slot, gap, background, or object boundary. Choose uncertain whenever the target is dark, blurred, tiny, occluded, or physically ambiguous. Be conservative: unclear evidence is uncertain, never likely_fill_ok.

This is assistant shadow review only. Return the requested JSON."""


SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": [
            "likely_fill_ok", "uncertain", "likely_reject"]},
        "visual_evidence_clear": {"type": "boolean"},
        "same_solid_surface": {"type": "boolean"},
        "crosses_real_opening_or_background": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "minLength": 1, "maxLength": 200},
    },
    "required": [
        "decision", "visual_evidence_clear", "same_solid_surface",
        "crosses_real_opening_or_background", "confidence", "reason"],
    "additionalProperties": False,
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_json(url: str, body: dict | None = None, timeout: int = 300) -> dict:
    request = urllib.request.Request(
        url, data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def validate_vlm_decision(value: dict) -> dict:
    required = set(SCHEMA["required"])
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("VLM output keys differ")
    if value["decision"] not in SCHEMA["properties"]["decision"]["enum"]:
        raise ValueError("VLM decision differs")
    for field in ("visual_evidence_clear", "same_solid_surface",
                  "crosses_real_opening_or_background"):
        if type(value[field]) is not bool:
            raise ValueError(f"VLM boolean differs: {field}")
    confidence = value["confidence"]
    if (type(confidence) not in (int, float) or isinstance(confidence, bool)
            or not 0 <= confidence <= 1):
        raise ValueError("VLM confidence differs")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("VLM reason differs")
    if value["decision"] == "likely_fill_ok" and (
            not value["visual_evidence_clear"]
            or not value["same_solid_surface"]
            or value["crosses_real_opening_or_background"]
            or confidence < 0.85):
        raise ValueError("unsafe likely_fill_ok combination")
    if (value["decision"] == "likely_reject"
            and not value["crosses_real_opening_or_background"]):
        raise ValueError("unsupported likely_reject combination")
    return value


def select_camera1_tail(policy: dict, component_review: dict) -> list[dict]:
    for name, packet in (("policy", policy), ("component review", component_review)):
        if (packet.get("candidate_only") is not True
                or packet.get("human_truth") is not False
                or packet.get("approved_for_training") is not False
                or packet.get("production_deployed") is not False
                or packet.get("default_enabled") is not False):
            raise ValueError(f"{name} safety flags are inconsistent")
    if (policy.get("automatic_decisions_enabled") is not False
            or policy.get("policy", {}).get(
                "camera1_bottom_tail_evidence_coverage") != "complete"):
        raise ValueError("policy tail coverage is not safe")
    defer_after = policy.get("policy", {}).get("camera1_defer_after_rank")
    if not isinstance(defer_after, int) or defer_after <= 0:
        raise ValueError("policy camera1 tail boundary is invalid")
    review_rows = component_review.get("rows")
    if not isinstance(review_rows, list) or not review_rows:
        raise ValueError("component review rows are missing")
    review_by_id = {row.get("component_id"): row for row in review_rows}
    if None in review_by_id or len(review_by_id) != len(review_rows):
        raise ValueError("component review identities are invalid")
    selected = []
    for row in policy.get("rows", []):
        if (row.get("camera") == "camera1"
                and int(row.get("camera_review_priority_rank", 0)) > defer_after):
            source = review_by_id.get(row.get("component_id"))
            if source is None or source.get("review_image_sha256") is None:
                raise ValueError(f"component review binding missing: {row.get('component_id')}")
            selected.append({
                "component_id": row["component_id"],
                "camera_rank": row["camera_review_priority_rank"],
                "policy_queue": row["manual_review_queue"],
                "policy_reason": row["policy_reason"],
                "review_image": source["review_image"],
                "review_image_sha256": source["review_image_sha256"],
            })
    selected.sort(key=lambda row: (row["camera_rank"], row["component_id"]))
    expected = policy.get("counts", {}).get("camera1_bottom_total")
    if expected != len(selected) or not selected:
        raise ValueError("camera1 tail count mismatch")
    return selected


def evaluate(policy_path: Path, component_review_path: Path, output: Path,
             base: str, model: str) -> dict:
    if output.exists():
        raise FileExistsError(output)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    component_review = json.loads(
        component_review_path.read_text(encoding="utf-8"))
    selected = select_camera1_tail(policy, component_review)
    health_url = base.rsplit("/v1", 1)[0].rstrip("/") + "/health"
    health_before = request_json(health_url)
    if (health_before.get("status") != "ok"
            or health_before.get("model_loaded") is not True
            or health_before.get("exact_requests_inflight") != 0):
        raise RuntimeError("VLM is not idle and ready")
    rows = []
    for item in selected:
        image_path = component_review_path.parent / item["review_image"]
        if file_sha256(image_path) != item["review_image_sha256"]:
            raise ValueError(f"review image hash mismatch: {item['component_id']}")
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        payload = {
            "model": model,
            "temperature": 0,
            "max_tokens": 180,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url":
                    f"data:{mime};base64," + base64.b64encode(
                        image_path.read_bytes()).decode("ascii")}},
            ]}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "ego_tail_guard", "schema": SCHEMA, "strict": True}},
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
            "policy_queue": item["policy_queue"],
            "effective_vlm_decision": effective_decision,
            "wall_seconds": rows[-1]["wall_seconds"],
        }, ensure_ascii=False), flush=True)
    health_after = request_json(health_url)
    packet = {
        "schema_version": "labelagent.ego_tail_vlm_guard_shadow.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "assistant_review_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "production_deployed": False,
        "default_enabled": False,
        "automatic_decisions_enabled": False,
        "policy_sha256": file_sha256(policy_path),
        "component_review_sha256": file_sha256(component_review_path),
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "model": model,
        "review_count": len(rows),
        "vlm_likely_fill_ok": sum(
            row["effective_vlm_decision"] == "likely_fill_ok" for row in rows),
        "vlm_uncertain": sum(
            row["effective_vlm_decision"] == "uncertain" for row in rows),
        "vlm_likely_reject": sum(
            row["effective_vlm_decision"] == "likely_reject" for row in rows),
        "policy_deferred_supported": sum(
            row["policy_queue"] == "deferred_manual_review"
            and row["vlm_supports_deferral"] for row in rows),
        "logical_conflict_abstentions": sum(
            row["status"] == "vlm_logical_conflict_abstained" for row in rows),
        "rows": rows,
        "health_before": {
            "status": health_before.get("status"),
            "exact_requests_inflight": health_before.get("exact_requests_inflight"),
        },
        "health_after": {
            "status": health_after.get("status"),
            "exact_requests_inflight": health_after.get("exact_requests_inflight"),
        },
        "boundary": (
            "VLM agreement is assistant shadow triage, not human accuracy, "
            "truth, training approval, or automatic acceptance."),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        temporary.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        readback = json.loads(temporary.read_text(encoding="utf-8"))
        if readback["review_count"] != len(rows):
            raise ValueError("VLM guard readback mismatch")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--component-review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base", default="http://127.0.0.1:28868/v1")
    parser.add_argument("--model", default="labelagent-vlm8b")
    arguments = parser.parse_args()
    packet = evaluate(arguments.policy, arguments.component_review,
                      arguments.output, arguments.base, arguments.model)
    print(json.dumps({key: packet[key] for key in (
        "review_count", "vlm_likely_fill_ok", "vlm_uncertain",
        "vlm_likely_reject", "policy_deferred_supported",
        "logical_conflict_abstentions")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
