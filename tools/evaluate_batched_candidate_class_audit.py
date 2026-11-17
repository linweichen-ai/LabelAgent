"""Evaluate one strict multi-candidate class-audit call against persisted audits.

This is an isolated experiment: it reads an existing candidate state and source
image, calls the configured VLM once, and writes only the requested receipt.
It never mutates candidates, truth, references, or training approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from agent.candidate_class_audit import (  # noqa: E402
    HIGH_RISK_LABELS,
    blind_candidate_class_schema,
    candidate_fingerprint,
    prepare_candidate_evidence,
    validate_blind_candidate_class_decision,
)
from agent.workflow import VISIBLE_LABEL_DEFINITIONS, chat  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--candidate-ids", nargs="+", type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--llm-url")
    parser.add_argument("--llm-model")
    return parser.parse_args()


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def batch_schema(candidate_ids: list[int], labels: list[str]) -> dict:
    properties = {str(candidate_id): blind_candidate_class_schema(labels)
                  for candidate_id in candidate_ids}
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "object",
                "properties": properties,
                "required": [str(candidate_id) for candidate_id in candidate_ids],
                "additionalProperties": False,
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def batch_prompt(targets: list[dict], labels: list[str]) -> str:
    pairs = [{"candidate_id": item["id"],
              "uncolored_image": 2 * index + 1,
              "cyan_mask_image": 2 * index + 2}
             for index, item in enumerate(targets)]
    definitions = {label: VISIBLE_LABEL_DEFINITIONS[label] for label in labels
                   if isinstance(VISIBLE_LABEL_DEFINITIONS.get(label), str)}
    return (
        "Independently classify every highlighted machine candidate. Images are supplied "
        "in exact pairs listed below: the first image in each pair is an uncolored local "
        "crop, and the second is the identical crop with only that candidate mask highlighted "
        "in cyan; holes are excluded. Cyan is not an object color. No current candidate class "
        "is provided. Return exactly one decision for every listed candidate ID, keyed by its "
        "decimal ID in decisions. Judge each pair independently and never transfer visual cues "
        "between IDs. Use known_label only for one visually supported allowed class. For tiny, "
        "blurred, occluded, mixed, multiple-object, or weak evidence use uncertain with "
        "observed_label=null. If no allowed class fits use conflict with observed_label=null. "
        "For person require visible human anatomy inside cyan. For ego require visible host "
        "robot body, chassis, or forks inside cyan; viewpoint alone never proves ego. For "
        "other_robot require a distinct robot or mobile platform, separate from ego and people. "
        "A mask mixing a person with vehicle or robot structure is uncertain. This is "
        "candidate-only evidence, never human truth. Return only the strict JSON object.\nInput: "
        + json.dumps({"image_pairs": pairs, "allowed_labels": labels,
                      "class_definitions": definitions}, ensure_ascii=False,
                     separators=(",", ":"))
    )


def semantic_core(decision: dict, current_label: str) -> dict:
    matches = (decision["classification"] == "known_label"
               and decision["observed_label"] == current_label
               and decision["confidence"] >= 0.90)
    return {
        "classification": decision["classification"],
        "observed_label": decision["observed_label"],
        "candidate_supported": decision["candidate_supported"],
        "reason_code": decision["reason_code"],
        "route": "candidate_preserved" if matches else "human_review_required",
    }


def main() -> int:
    args = parse_args()
    state_path = args.state.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    image_bytes = image_path.read_bytes()
    frame_key = str(args.frame)
    candidates = list((state.get("candidates") or {}).get(frame_key) or [])
    selected_ids = set(args.candidate_ids or [])
    targets = [candidate for candidate in candidates
               if candidate.get("label") in HIGH_RISK_LABELS
               and (not selected_ids or candidate.get("id") in selected_ids)]
    if not 2 <= len(targets) <= 6:
        raise ValueError("The experiment requires between two and six high-risk candidates")
    if selected_ids != {item["id"] for item in targets} and selected_ids:
        raise ValueError("One or more requested candidate IDs are missing or not high risk")
    labels = list(state.get("labels") or [])
    config = dict(state.get("config") or {})
    if args.llm_url:
        config["llm_url"] = args.llm_url
    if args.llm_model:
        config["llm_model"] = args.llm_model
    config["llm_fast_url"] = ""
    config["llm_fast_model"] = ""
    config["llm_max_tokens"] = max(int(config.get("llm_max_tokens") or 768),
                                   384 * len(targets))

    persisted = {item.get("candidate_id"): item for item in
                 ((state.get("agent_runs") or {}).get(frame_key, {})
                  .get("candidate_class_audits") or [])}
    if any(item["id"] not in persisted or not isinstance(
            persisted[item["id"]].get("decision"), dict) for item in targets):
        raise ValueError("Persisted production audits are required for every target")

    evidence = [prepare_candidate_evidence(image_bytes, candidate)
                for candidate in targets]
    images = [payload for item in evidence
              for payload in (item["original"], item["marked"])]
    schema = batch_schema([item["id"] for item in targets], labels)
    prompt = batch_prompt(targets, labels)
    started = time.perf_counter()
    response = chat(config, prompt, images[0], extra_images=tuple(images[1:]),
                    response_schema=schema)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    decisions = response.get("decisions") if isinstance(response, dict) else None
    if not isinstance(decisions, dict) or set(decisions) != {
            str(item["id"]) for item in targets}:
        raise ValueError("Batched audit did not return every exact candidate ID")

    rows = []
    for candidate, item_evidence in zip(targets, evidence):
        candidate_id = candidate["id"]
        raw_batch_decision = decisions[str(candidate_id)]
        batch_protocol_error = None
        try:
            batch_decision = validate_blind_candidate_class_decision(
                raw_batch_decision, labels)
        except Exception as error:
            batch_decision = None
            batch_protocol_error = str(error)
        baseline_decision = validate_blind_candidate_class_decision(
            persisted[candidate_id]["decision"], labels)
        batch_core = (semantic_core(batch_decision, candidate["label"])
                      if batch_decision is not None else None)
        baseline_core = semantic_core(baseline_decision, candidate["label"])
        rows.append({
            "candidate_id": candidate_id,
            "candidate_sha256": candidate_fingerprint(candidate),
            "bound_current_label": candidate["label"],
            "evidence": {
                "crop_box_xyxy": item_evidence["crop_box_xyxy"],
                "original_sha256": item_evidence["original_sha256"],
                "marked_sha256": item_evidence["marked_sha256"],
                "current_label_shown_to_model": False,
            },
            "baseline_core": baseline_core,
            "batch_core": batch_core,
            "semantic_core_match": (batch_decision is not None
                                     and baseline_core == batch_core),
            "batch_protocol_valid": batch_decision is not None,
            "batch_protocol_error": batch_protocol_error,
            "raw_batch_decision": raw_batch_decision,
            "baseline_confidence": baseline_decision["confidence"],
            "batch_confidence": (batch_decision["confidence"]
                                 if batch_decision is not None else None),
            "batch_visible_cues": (batch_decision["visible_cues"]
                                   if batch_decision is not None else None),
            "batch_reason": (batch_decision["reason"]
                             if batch_decision is not None else None),
        })

    result = {
        "schema_version": "labelagent.batched-candidate-class-audit-experiment.v1",
        "state": str(state_path),
        "state_sha256": sha256(state_path.read_bytes()),
        "image": str(image_path),
        "image_sha256": sha256(image_bytes),
        "frame": args.frame,
        "candidate_count": len(targets),
        "candidate_ids": [item["id"] for item in targets],
        "image_count": len(images),
        "elapsed_ms": elapsed_ms,
        "prompt_sha256": sha256(prompt.encode("utf-8")),
        "schema_sha256": sha256(json.dumps(
            schema, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8")),
        "rows": rows,
        "semantic_core_matches": sum(row["semantic_core_match"] for row in rows),
        "all_semantic_cores_match": all(row["semantic_core_match"] for row in rows),
        "protocol_valid_count": sum(row["batch_protocol_valid"] for row in rows),
        "all_protocol_valid": all(row["batch_protocol_valid"] for row in rows),
        "candidate_only": True,
        "candidate_mutated": False,
        "human_truth": False,
        "approved_for_training": False,
        "approved_for_deployment": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
