"""Freeze a path-free expert shadow queue from local Qwen suggestions."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


SCHEMA_VERSION = "labelagent.mixed-hard-expert-shadow-queue.v1"
SOURCE_SCHEMA = "labelagent.local-qwen-mixed-hard-suggestions.v1"


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def build_queue(source_bytes, *, expert_latency_ms=21070.0,
                estimated_cost_per_call=None, selection_mode="recommended"):
    if not isinstance(source_bytes, bytes):
        raise ValueError("source suggestion bytes are required")
    payload = json.loads(source_bytes.decode("utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if (payload.get("schema_version") != SOURCE_SCHEMA
            or payload.get("status") != "completed"
            or payload.get("samples_total") != 34
            or payload.get("samples_passed") != 34
            or payload.get("samples_failed") != 0
            or not isinstance(rows, list) or len(rows) != 34
            or len({row.get("sample_id") for row in rows}) != 34):
        raise ValueError("source suggestions are incomplete or have the wrong schema")
    for key, expected in (("candidate_only", True), ("human_truth", False),
                          ("approved_for_training", False),
                          ("test24_used", False)):
        if payload.get(key) is not expected:
            raise ValueError("source suggestion safety boundary differs")
    if type(expert_latency_ms) not in (int, float) or expert_latency_ms <= 0:
        raise ValueError("expert latency estimate must be positive")
    if (estimated_cost_per_call is not None
            and (type(estimated_cost_per_call) not in (int, float)
                 or estimated_cost_per_call < 0)):
        raise ValueError("estimated cost per call must be non-negative")
    if selection_mode not in {"recommended", "all"}:
        raise ValueError("selection mode must be recommended or all")

    selected, strata, signals = [], Counter(), Counter()
    for row in rows:
        if (row.get("status") != "passed"
                or row.get("candidate_only") is not True
                or row.get("human_truth") is not False
                or row.get("approved_for_training") is not False
                or row.get("test24_used") is not False):
            raise ValueError("source row safety boundary differs")
        route = row.get("expert_shadow_route")
        selection = route.get("selection") if isinstance(route, dict) else None
        if (not isinstance(route, dict)
                or route.get("dispatch_performed") is not False
                or route.get("external_upload_performed") is not False
                or not isinstance(selection, dict)):
            raise ValueError("source route is missing its no-dispatch attestation")
        if selection_mode == "recommended" and route.get("recommended") is not True:
            continue
        sample_id = row.get("sample_id")
        image_sha = row.get("copied_image_sha256")
        cohort = row.get("cohort")
        hard_signals = selection.get("hard_signal_codes") or []
        if (not isinstance(sample_id, str) or not sample_id
                or not isinstance(image_sha, str) or len(image_sha) != 64
                or not isinstance(cohort, str) or not cohort
                or not isinstance(hard_signals, list)
                or any(not isinstance(value, str) for value in hard_signals)):
            raise ValueError("selected row identity is invalid")
        strata[cohort] += 1
        signals.update(hard_signals)
        selected.append({
            "sample_id": sample_id,
            "cohort": cohort,
            "source_image_sha256": image_sha,
            "difficulty_score": route.get("difficulty_score"),
            "hard_signal_codes": sorted(set(hard_signals)),
            "local_model_latency_ms": row.get("latency_ms"),
            "external_upload_authorized": False,
            "external_dispatch_performed": False,
            "human_reference_available": False,
        })
    selected.sort(key=lambda item: (item["cohort"], item["sample_id"]))
    calls = len(selected)
    cost = (round(calls * estimated_cost_per_call, 8)
            if estimated_cost_per_call is not None else None)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_suggestions_sha256": _sha256(source_bytes),
        "sample_count": calls,
        "source_sample_count": len(rows),
        "external_call_ratio": round(calls / len(rows), 6),
        "strata": dict(sorted(strata.items())),
        "hard_signal_counts": dict(sorted(signals.items())),
        "estimated_expert_latency_ms_per_call": round(float(expert_latency_ms), 3),
        "estimated_sequential_expert_latency_ms": round(calls * expert_latency_ms, 3),
        "estimated_cost_per_call": estimated_cost_per_call,
        "estimated_total_cost": cost,
        "cost_status": "estimated" if cost is not None else "unknown_requires_billing_rate",
        "external_upload_authorized": False,
        "ready_for_shadow_execution": False,
        "ready_for_quality_ab": False,
        "images_copied": False,
        "image_paths_included": False,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "test24_used": False,
        "samples": selected,
        **({"selection_mode": "all"} if selection_mode == "all" else {}),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expert-latency-ms", type=float, default=21070.0)
    parser.add_argument("--estimated-cost-per-call", type=float)
    parser.add_argument("--selection-mode", choices=("recommended", "all"),
                        default="recommended")
    args = parser.parse_args(argv)
    result = build_queue(
        args.source.read_bytes(), expert_latency_ms=args.expert_latency_ms,
        estimated_cost_per_call=args.estimated_cost_per_call,
        selection_mode=args.selection_mode)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in (
        "sample_count", "source_sample_count", "external_call_ratio",
        "estimated_sequential_expert_latency_ms", "cost_status",
        "external_upload_authorized", "ready_for_shadow_execution",
        "candidate_only", "human_truth", "approved_for_training",
        "test24_used")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
