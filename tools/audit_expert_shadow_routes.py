"""Audit persisted machine candidates with the upload-free expert router."""
import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.agent.expert_routing import evaluate_expert_shadow_route


def audit_state(state, *, image_size):
    if not isinstance(state, dict):
        raise ValueError("state must be an object")
    runs = state.get("agent_runs") or {}
    candidates = state.get("candidates") or {}
    if not isinstance(runs, dict) or not isinstance(candidates, dict):
        raise ValueError("state candidate ledgers are invalid")
    config = copy.deepcopy(state.get("config") or {})
    config["llm_expert_shadow_routing_enabled"] = True
    priority_frames = set(config.get("llm_expert_shadow_priority_frames") or [])
    rows = []
    skipped_test = 0
    for frame_key in sorted(runs, key=lambda value: int(value)):
        run = runs[frame_key]
        if not isinstance(run, dict):
            continue
        if str(run.get("dataset_split") or "").casefold() in {"test", "test24"}:
            skipped_test += 1
            continue
        frame_candidates = candidates.get(frame_key) or []
        decision = evaluate_expert_shadow_route(
            config,
            discovery=run.get("visible_label_discovery"),
            agent=run,
            review_queue=run.get("review_queue"),
            class_audits=run.get("candidate_class_audits"),
            candidate_labels=[item.get("label") for item in frame_candidates
                              if isinstance(item, dict)],
            candidates=frame_candidates,
            image_size=image_size,
            manual_priority=int(frame_key) in priority_frames,
            local_model_latency_ms=(run.get("expert_shadow_routing") or {})
            .get("timing", {}).get("local_model_latency_ms"),
            sam_result_reused=True,
        )
        rows.append({
            "frame": int(frame_key),
            "route": decision["route"],
            "recommended": decision["recommended"],
            "difficulty_score": decision["difficulty_score"],
            "reason_codes": [item["code"] for item in decision["reasons"]],
            "signals": decision["signals"],
            "candidate_count": len(frame_candidates),
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
        })
    reason_counts = Counter(
        code for row in rows for code in row["reason_codes"]
        if code != "priority_class_involved")
    recommended = sum(row["recommended"] for row in rows)
    return {
        "schema_version": "labelagent.expert-shadow-routing-audit.v1",
        "status": "completed",
        "frames_audited": len(rows),
        "frames_recommended_for_external_expert": recommended,
        "frames_local_result_sufficient": len(rows) - recommended,
        "external_call_ratio": (round(recommended / len(rows), 4) if rows else None),
        "reason_counts": dict(sorted(reason_counts.items())),
        "test_frames_skipped": skipped_test,
        "dispatch_performed": False,
        "external_upload_performed": False,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "frames": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--image-width", type=int, required=True)
    parser.add_argument("--image-height", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.image_width <= 0 or args.image_height <= 0:
        raise ValueError("image dimensions must be positive")
    state = json.loads(args.state.read_text(encoding="utf-8"))
    result = audit_state(state, image_size=(args.image_width, args.image_height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "frames"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
