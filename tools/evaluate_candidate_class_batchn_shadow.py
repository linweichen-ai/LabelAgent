#!/usr/bin/env python3
"""Shadow-evaluate one native VLM batch against persisted accepted audits.

The evaluator is read-only with respect to the workbench.  It sends one batch
request, writes one local receipt, and never changes candidates, truth,
reference frames, training approval, configuration, or model weights.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.agent import candidate_class_audit as audit  # noqa: E402
from source.agent.workflow import VISIBLE_LABEL_DEFINITIONS  # noqa: E402
from tools.evaluate_candidate_class_batch2_live import (  # noqa: E402
    body_for,
    canonical_sha256,
    fetch_frame,
    frame_candidates,
    parse_output,
    request_json,
    semantic_core,
    sha256_bytes,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--candidate-ids", nargs="+", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tokens", default=385, type=int)
    args = parser.parse_args()
    if not 2 <= len(args.candidate_ids) <= 8:
        raise ValueError("shadow batch size must be between 2 and 8")
    if len(set(args.candidate_ids)) != len(args.candidate_ids):
        raise ValueError("candidate IDs must be unique")
    parsed_base = urllib.parse.urlsplit(args.base_url)
    if parsed_base.scheme != "http" or parsed_base.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("workbench endpoint must be loopback")
    output = args.output.resolve()
    allowed_roots = ((ROOT / "artifacts").resolve(), (ROOT / "docs" / "evidence").resolve())
    if not any(output.is_relative_to(root) for root in allowed_roots):
        raise ValueError("output must be inside artifacts or docs/evidence")
    if output.exists():
        raise FileExistsError(output)
    state_path = args.state.resolve()
    state_before = state_path.read_bytes()
    state = json.loads(state_before)
    config = dict(state.get("config") or {})
    labels = list(state.get("labels") or [])
    topic = state.get("topic")
    source_path = state.get("path") or state.get("video_path")
    if not config.get("llm_url") or not config.get("llm_model") or not labels:
        raise ValueError("state lacks model settings or labels")
    model_url = urllib.parse.urlsplit(config["llm_url"])
    if model_url.scheme != "http" or model_url.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("VLM endpoint must be loopback")

    candidates = frame_candidates(state, topic, args.frame)
    by_id = {item.get("id"): item for item in candidates}
    targets = [by_id.get(identifier) for identifier in args.candidate_ids]
    if any(item is None for item in targets):
        raise ValueError("one or more candidate IDs are missing")
    if not all(audit.candidate_requires_class_audit(item) for item in targets):
        raise ValueError("all candidates must belong to the established audit gate")
    run = (state.get("agent_runs") or {}).get(str(args.frame)) or {}
    accepted = {item.get("candidate_id"): item
                for item in run.get("candidate_class_audits") or []}
    if not all(identifier in accepted for identifier in args.candidate_ids):
        raise ValueError("persisted accepted audit is missing")

    image_bytes = fetch_frame(
        args.base_url, args.frame, topic=topic, path=source_path
    )
    schema = audit.blind_candidate_class_schema(labels)
    prepared = []
    for candidate in targets:
        evidence = audit.prepare_candidate_evidence(image_bytes, candidate)
        prompt = audit.build_blind_candidate_class_prompt(
            candidate["id"], labels, VISIBLE_LABEL_DEFINITIONS,
            isolated_only=str(candidate.get("label", "")).casefold() == "pallet",
        )
        primary, extra = audit.candidate_audit_images(candidate, evidence)
        prepared.append((candidate, prompt, (primary, *extra)))
    bodies = [body_for(config, prompt, images, schema, args.max_tokens)
              for _, prompt, images in prepared]
    try:
        payload, wall_seconds = request_json(
            config["llm_url"].rstrip("/") + "/chat/batch_completions",
            {"requests": bodies}, config.get("llm_key", ""))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        if state_path.read_bytes() != state_before:
            raise RuntimeError("workbench state changed during rejected shadow evaluation")
        result = {
            "schema_version": "labelagent.candidate-class-batchn-shadow.v1",
            "frame": args.frame,
            "topic": topic,
            "batch_size": len(bodies),
            "max_tokens": args.max_tokens,
            "endpoint_accepted": False,
            "http_status": error.code,
            "error": body[:1000],
            "candidate_ids": args.candidate_ids,
            "candidate_artifact_sha256": canonical_sha256(candidates),
            "source_image_sha256": sha256_bytes(image_bytes),
            "state_sha256": sha256_bytes(state_before),
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "model_inference_started": False,
            "production_adopted": False,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        return 0
    outputs = payload.get("results")
    if not isinstance(outputs, list) or len(outputs) != len(prepared):
        raise RuntimeError("batch endpoint returned a mismatched result count")

    rows = []
    for (candidate, _, _), output_value in zip(prepared, outputs):
        raw, decision, repair, error = parse_output(output_value, labels)
        if decision is None:
            current = None
        else:
            current = semantic_core(decision, candidate["label"])
        prior_decision = accepted[candidate["id"]].get("decision") or {}
        prior = semantic_core(prior_decision, candidate["label"])
        rows.append({
            "candidate_id": candidate["id"],
            "bound_current_label": candidate["label"],
            "candidate_sha256": audit.candidate_fingerprint(candidate),
            "persisted_execution_mode": accepted[candidate["id"]].get("execution_mode"),
            "persisted_semantic_core": prior,
            "shadow_semantic_core": current,
            "semantic_core_match": current == prior,
            "route_match": current is not None and current["route"] == prior["route"],
            "shadow_confidence": decision.get("confidence") if decision else None,
            "shadow_protocol_valid": decision is not None,
            "shadow_protocol_repair": repair,
            "shadow_protocol_error": error,
            "shadow_raw_decision": raw,
            "cache_hit": bool((output_value.get("local_vlm") or {}).get("cache_hit")),
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
        })
    if state_path.read_bytes() != state_before:
        raise RuntimeError("workbench state changed during read-only shadow evaluation")
    result = {
        "schema_version": "labelagent.candidate-class-batchn-shadow.v1",
        "frame": args.frame,
        "topic": topic,
        "batch_size": len(rows),
        "max_tokens": args.max_tokens,
        "wall_seconds": wall_seconds,
        "local_vlm": payload.get("local_vlm"),
        "all_uncached": all(not item["cache_hit"] for item in rows),
        "protocol_valid": all(item["shadow_protocol_valid"] for item in rows),
        "semantic_core_match": all(item["semantic_core_match"] for item in rows),
        "route_match": all(item["route_match"] for item in rows),
        "candidate_artifact_sha256": canonical_sha256(candidates),
        "source_image_sha256": sha256_bytes(image_bytes),
        "state_sha256": sha256_bytes(state_before),
        "rows": rows,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "production_adopted": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
