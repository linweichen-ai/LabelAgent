"""Compare two uncached single-candidate VLM budgets without changing state."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--candidate-ids", nargs="+")
    parser.add_argument("--baseline-max-tokens", default=768, type=int)
    parser.add_argument("--compact-max-tokens", default=384, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    state_path = args.state.resolve()
    output_path = args.output.resolve()
    if not output_path.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("output must be inside this repository's artifacts directory")
    if output_path.exists():
        raise FileExistsError(output_path)
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    for value in (args.baseline_max_tokens, args.compact_max_tokens):
        if not 1 <= value <= 4096:
            raise ValueError("token budgets must be between 1 and 4096")
    if args.compact_max_tokens >= args.baseline_max_tokens:
        raise ValueError("compact budget must be smaller than baseline budget")

    state_before = state_path.read_bytes()
    state = json.loads(state_before)
    config = dict(state.get("config") or {})
    labels = list(state.get("labels") or [])
    topic = state.get("topic")
    source_path = state.get("path") or state.get("video_path")
    if not isinstance(topic, str) or not isinstance(source_path, str):
        raise ValueError("state lacks the selected dataset path or topic")
    if not config.get("llm_url") or not config.get("llm_model") or not labels:
        raise ValueError("state lacks model settings or labels")
    parsed = urllib.parse.urlsplit(config["llm_url"])
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("the evaluator only permits a loopback VLM endpoint")

    candidates = frame_candidates(state, topic, args.frame)
    requested_ids = set(args.candidate_ids or [])
    targets = [
        candidate for candidate in candidates
        if audit.candidate_requires_class_audit(candidate)
        and (not requested_ids or str(candidate.get("id")) in requested_ids)
    ]
    if not targets:
        raise ValueError("no current audit target was selected")
    if requested_ids and {str(item.get("id")) for item in targets} != requested_ids:
        raise ValueError("one or more requested candidate IDs are missing")

    image_bytes = fetch_frame(args.base_url, args.frame, topic=topic, path=source_path)
    schema = audit.blind_candidate_class_schema(labels)
    endpoint = config["llm_url"].rstrip("/") + "/chat/completions"
    api_key = config.get("llm_key", "")
    rows = []
    totals = {"baseline": 0.0, "compact": 0.0}
    for index, candidate in enumerate(targets):
        evidence = audit.prepare_candidate_evidence(image_bytes, candidate)
        prompt = audit.build_blind_candidate_class_prompt(
            candidate["id"], labels, VISIBLE_LABEL_DEFINITIONS,
            isolated_only=str(candidate.get("label", "")).casefold() == "pallet",
        )
        primary, extra = audit.candidate_audit_images(candidate, evidence)
        images = (primary, *extra)
        budgets = [
            ("baseline", args.baseline_max_tokens),
            ("compact", args.compact_max_tokens),
        ]
        if index % 2:
            budgets.reverse()
        outputs = {}
        for name, budget in budgets:
            payload, elapsed = request_json(
                endpoint, body_for(config, prompt, images, schema, budget), api_key
            )
            raw, decision, repair, error = parse_output(payload, labels)
            core = semantic_core(decision, candidate["label"]) if decision else None
            outputs[name] = {
                "max_tokens": budget,
                "wall_seconds": elapsed,
                "cache_hit": bool((payload.get("local_vlm") or {}).get("cache_hit")),
                "model_seconds": (payload.get("local_vlm") or {}).get("seconds"),
                "finish_reason": payload["choices"][0].get("finish_reason"),
                "protocol_valid": decision is not None,
                "protocol_repair": repair,
                "protocol_error": error,
                "semantic_core": core,
                "confidence": decision.get("confidence") if decision else None,
                "raw_decision": raw,
            }
            totals[name] += elapsed
        rows.append({
            "candidate_id": candidate["id"],
            "bound_current_label": candidate["label"],
            "candidate_sha256": audit.candidate_fingerprint(candidate),
            "evidence_isolated_sha256": evidence["isolated_sha256"],
            **outputs,
            "semantic_core_match": (
                outputs["baseline"]["semantic_core"]
                == outputs["compact"]["semantic_core"]
            ),
            "route_match": (
                outputs["baseline"]["semantic_core"] is not None
                and outputs["compact"]["semantic_core"] is not None
                and outputs["baseline"]["semantic_core"]["route"]
                == outputs["compact"]["semantic_core"]["route"]
            ),
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
        })

    all_uncached = all(
        not row[name]["cache_hit"] for row in rows
        for name in ("baseline", "compact")
    )
    all_valid = all(
        row[name]["protocol_valid"] for row in rows
        for name in ("baseline", "compact")
    )
    all_stopped = all(
        row[name]["finish_reason"] == "stop" for row in rows
        for name in ("baseline", "compact")
    )
    semantic_match = all(row["semantic_core_match"] for row in rows)
    route_match = all(row["route_match"] for row in rows)
    gain = totals["baseline"] - totals["compact"]
    result = {
        "schema_version": "labelagent.candidate-class-token-budget-live-evaluation.v1",
        "frame": args.frame,
        "topic": topic,
        "state_sha256": sha256_bytes(state_before),
        "candidate_artifact_sha256": canonical_sha256(candidates),
        "source_image_sha256": sha256_bytes(image_bytes),
        "candidate_ids": [item["id"] for item in targets],
        "baseline_max_tokens": args.baseline_max_tokens,
        "compact_max_tokens": args.compact_max_tokens,
        "rows": rows,
        "summary": {
            "candidate_count": len(rows),
            "baseline_wall_seconds": round(totals["baseline"], 3),
            "compact_wall_seconds": round(totals["compact"], 3),
            "latency_gain_seconds": round(gain, 3),
            "latency_reduction_fraction": (
                round(gain / totals["baseline"], 4) if totals["baseline"] else None
            ),
            "all_uncached": all_uncached,
            "all_protocol_valid": all_valid,
            "all_finish_reason_stop": all_stopped,
            "all_semantic_cores_match": semantic_match,
            "all_routes_match": route_match,
            "safe_to_consider_compact_budget": (
                all_uncached and all_valid and all_stopped
                and semantic_match and route_match and gain > 0
            ),
        },
        "inputs_unchanged": {
            "state": state_path.read_bytes() == state_before,
            "candidates": canonical_sha256(
                frame_candidates(json.loads(state_path.read_bytes()), topic, args.frame)
            ) == canonical_sha256(candidates),
        },
        "formal_workbench_writes": False,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "approved_for_deployment": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
