"""Evaluate one color-independent other_robot fallback through production gates.

This is a candidate-only, read-only experiment. It regenerates SAM proposals for
``robotic pallet truck``, runs the existing label-blind 8B candidate audit, and
records geometry warnings. It never edits workbench state, truth, training
approval, production prompts, or model weights.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from PIL import Image

from source.agent.candidate_class_audit import audit_high_risk_candidates
from source.agent.workflow import (AnnotationAgent, VISIBLE_LABEL_DEFINITIONS,
                                   _chat, _chat_batch, _shape_summary,
                                   candidate_overlay, sam_text_multi)


FALLBACK_PROMPT = "robotic pallet truck"
SUPPORTED_FALLBACK_PROMPTS = (
    FALLBACK_PROMPT, "autonomous pallet truck", "floor cleaning robot")


def sam_prompts(fallback_prompt: str) -> dict[str, str]:
    if fallback_prompt not in SUPPORTED_FALLBACK_PROMPTS:
        raise ValueError("unsupported fallback prompt")
    return {"baseline": "robot", "fallback": fallback_prompt}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _polygon_area(points) -> float:
    return abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))) / 2.0)


def candidate_geometry(candidate: dict, width: int, height: int) -> dict:
    """Return content-free geometry and a warning, never a rejection decision."""
    parts = candidate.get("parts") or [candidate]
    rings = [part.get("points") or [] for part in parts]
    points = [point for ring in rings for point in ring]
    if (not points or width <= 0 or height <= 0
            or any(len(point) != 2 or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in point) for point in points)):
        raise ValueError("candidate geometry is invalid")
    left = min(point[0] for point in points)
    top = min(point[1] for point in points)
    right = max(point[0] for point in points)
    bottom = max(point[1] for point in points)
    foreground_area = 0.0
    for part in parts:
        foreground_area += _polygon_area(part.get("points") or [])
        foreground_area -= sum(
            _polygon_area(hole) for hole in part.get("holes") or [])
    bbox_area = max(0.0, right - left + 1) * max(0.0, bottom - top + 1)
    image_area = width * height
    touches_bottom = bottom >= height - 2
    width_fraction = (right - left + 1) / width
    bbox_area_fraction = bbox_area / image_area
    foreground_area_fraction = max(0.0, foreground_area) / image_area
    # This is deliberately only a review warning. The observed false proposal
    # spans most of the image width and touches the camera-host edge, while the
    # three visually plausible robots do neither. More truth is required before
    # this can become a production filter.
    bottom_large_ego_risk = (
        touches_bottom and width_fraction >= 0.50
        and bbox_area_fraction >= 0.08)
    return {
        "bbox_xyxy": [round(value, 3) for value in (left, top, right, bottom)],
        "touches_bottom": touches_bottom,
        "width_fraction": round(width_fraction, 6),
        "bbox_area_fraction": round(bbox_area_fraction, 6),
        "foreground_area_fraction": round(foreground_area_fraction, 6),
        "bottom_large_ego_risk": bottom_large_ego_risk,
        "warning_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--sam-url", required=True)
    parser.add_argument("--llm-url")
    parser.add_argument("--fallback-prompt", choices=SUPPORTED_FALLBACK_PROMPTS,
                        default=FALLBACK_PROMPT)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.sam_url.startswith("http://127.0.0.1:"):
        raise ValueError("SAM URL must be a loopback endpoint")
    if args.llm_url and not args.llm_url.startswith("http://127.0.0.1:"):
        raise ValueError("LLM URL must be a loopback endpoint")
    if any(not image.is_file() for image in args.images):
        raise FileNotFoundError("Every image must exist")

    state_bytes = args.state.read_bytes()
    state = json.loads(state_bytes.decode("utf-8"))
    config = dict(state.get("config") or {})
    config["sam3_url"] = args.sam_url.rstrip("/")
    if args.llm_url:
        config["llm_url"] = args.llm_url.rstrip("/")
    labels = list(state.get("labels") or [])
    if "other_robot" not in labels:
        raise ValueError("state label registry lacks other_robot")
    if not config.get("llm_url") or not config.get("llm_model"):
        raise ValueError("state lacks Local VLM settings")

    args.output_dir.mkdir(parents=True)
    rows = []
    for index, image_path in enumerate(args.images):
        image_bytes = image_path.read_bytes()
        with Image.open(image_path) as opened:
            width, height = opened.size
        started = time.perf_counter()
        outputs, sam_diagnostics = sam_text_multi(
            image_bytes, sam_prompts(args.fallback_prompt), config)
        baseline_agent = AnnotationAgent(
            config, image_bytes, {"prompts": []}, labels, 1)
        baseline_candidates = baseline_agent.candidates(
            outputs["baseline"]["results"], "other_robot")
        fallback_agent = AnnotationAgent(
            config, image_bytes, {"prompts": []}, labels, 1)
        candidates = fallback_agent.candidates(
            outputs["fallback"]["results"], "other_robot")
        sam_elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        audit_started = time.perf_counter()
        audits, review_queue = audit_high_risk_candidates(
            config, image_bytes, candidates, labels,
            VISIBLE_LABEL_DEFINITIONS, _chat,
            batch_chat=_chat_batch)
        audit_elapsed_ms = round((time.perf_counter() - audit_started) * 1000, 3)
        audits_by_id = {item["candidate_id"]: item for item in audits}
        overlay = args.output_dir / f"{index:02d}-{image_path.stem}-fallback.jpg"
        overlay.write_bytes(candidate_overlay(image_bytes, candidates))
        candidate_rows = []
        for candidate in candidates:
            audit = audits_by_id[candidate["id"]]
            candidate_rows.append({
                "candidate": _shape_summary(candidate),
                "geometry": candidate_geometry(candidate, width, height),
                "audit": {
                    "candidate_sha256": audit["candidate_sha256"],
                    "audit_sha256": audit["audit_sha256"],
                    "route": audit["route"],
                    "minimum_confidence": audit["minimum_confidence"],
                    "execution_mode": audit["execution_mode"],
                    "protocol_retry_count": audit["protocol_retry_count"],
                    "protocol_repair": audit["protocol_repair"],
                    "decision": audit["decision"],
                    "candidate_mutated": audit["candidate_mutated"],
                    "human_truth": audit["human_truth"],
                    "approved_for_training": audit["approved_for_training"],
                    "approved_for_deployment": audit["approved_for_deployment"],
                },
            })
        row = {
            "image": str(image_path.resolve()),
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "image_size": [width, height],
            "sam_elapsed_ms": sam_elapsed_ms,
            "audit_elapsed_ms": audit_elapsed_ms,
            "baseline_candidate_count": len(baseline_candidates),
            "candidate_count": len(candidates),
            "preserved_count": sum(
                item["audit"]["route"] == "candidate_preserved"
                for item in candidate_rows),
            "human_review_count": len(review_queue),
            "bottom_large_ego_risk_count": sum(
                item["geometry"]["bottom_large_ego_risk"]
                for item in candidate_rows),
            "sam_diagnostics": sam_diagnostics,
            "candidates": candidate_rows,
            "overlay": overlay.name,
            "overlay_sha256": file_sha256(overlay),
        }
        rows.append(row)
        print(json.dumps({key: row[key] for key in (
            "image", "candidate_count", "preserved_count",
            "baseline_candidate_count", "human_review_count",
            "bottom_large_ego_risk_count",
            "sam_elapsed_ms", "audit_elapsed_ms")}, ensure_ascii=False), flush=True)

    report = {
        "schema_version": "labelagent.other_robot_prompt_fallback_audit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "assistant_review_candidate_only",
        "fallback_prompt": args.fallback_prompt,
        "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "approved_for_deployment": False,
        "production_prompt_modified": False,
        "model_modified": False,
        "rows": rows,
    }
    result_path = args.output_dir / "results.json"
    result_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"output": str(result_path.resolve()),
                      "sha256": file_sha256(result_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
