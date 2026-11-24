#!/usr/bin/env python3
"""Evaluate a baseline-Qwen then focused-Qwen shadow cascade without dispatch."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.audit_second_assistant_slope_triage import (
    audit_second_assistant_triage,
)
from tools.run_local_qwen_mixed_hard_suggestions import (
    CONTAINER_TRANSITION_V3_INSTRUCTION,
    SCHEMA_VERSION,
    sha256_bytes,
)


NEGATIVE_COHORTS = frozenset({
    "batchb_flat_negative_candidate",
    "slope_hard_negative_candidate",
    "slope_negative_candidate",
})
V3_PROFILE = "mixed-hard-container-transition-v3"
V3_INSTRUCTION_SHA256 = sha256_bytes(
    CONTAINER_TRANSITION_V3_INSTRUCTION.encode("utf-8"))
CONTEXT_PATTERN = re.compile(r"\b(container|doorway|entrance|threshold)\b", re.I)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_suggestions(payload: dict[str, Any], *, focused: bool) -> dict[str, Any]:
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if (payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("status") != "completed"
            or payload.get("samples_total") != 34
            or payload.get("samples_passed") != 34
            or payload.get("samples_failed") != 0
            or not isinstance(rows, list) or len(rows) != 34
            or len({row.get("sample_id") for row in rows}) != 34):
        raise ValueError("Qwen suggestion packet is incomplete")
    for key, expected in {
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "test24_used": False,
    }.items():
        if payload.get(key) != expected:
            raise ValueError(f"unsafe Qwen packet boundary: {key}")
    if focused and (payload.get("prompt_profile") != V3_PROFILE
                    or payload.get("instruction_sha256") != V3_INSTRUCTION_SHA256):
        raise ValueError("focused Qwen packet uses another prompt")
    result = {}
    for row in rows:
        if (row.get("status") != "passed"
                or row.get("candidate_only") is not True
                or row.get("human_truth") is not False
                or row.get("approved_for_training") is not False
                or row.get("test24_used") is not False):
            raise ValueError("unsafe Qwen row boundary")
        route = row.get("expert_shadow_route")
        if (not isinstance(route, dict)
                or route.get("dispatch_performed") is not False
                or route.get("external_upload_performed") is not False):
            raise ValueError("Qwen row lacks no-dispatch routing attestation")
        result[row["sample_id"]] = row
    return result


def evaluate_prompt_cascade(
        packet_dir: Path | str,
        baseline_path: Path | str,
        focused_path: Path | str,
        triage_path: Path | str,
        *, expert_latency_ms: float = 21070.0) -> dict[str, Any]:
    packet_dir, baseline_path, focused_path, triage_path = map(
        Path, (packet_dir, baseline_path, focused_path, triage_path))
    if (type(expert_latency_ms) not in (int, float)
            or isinstance(expert_latency_ms, bool) or expert_latency_ms <= 0):
        raise ValueError("expert latency estimate must be positive")
    triage_audit = audit_second_assistant_triage(
        packet_dir, baseline_path, triage_path)
    baseline_payload, focused_payload, triage = map(
        _read_json, (baseline_path, focused_path, triage_path))
    baseline = _validate_suggestions(baseline_payload, focused=False)
    focused = _validate_suggestions(focused_payload, focused=True)
    if (baseline_payload.get("manifest_sha256")
            != focused_payload.get("manifest_sha256")
            or baseline_payload.get("receipt_sha256")
            != focused_payload.get("receipt_sha256")):
        raise ValueError("Qwen packets use different review inputs")
    if set(baseline) != set(focused):
        raise ValueError("Qwen packet sample identities differ")
    for sample_id in baseline:
        if (baseline[sample_id].get("copied_image_sha256")
                != focused[sample_id].get("copied_image_sha256")):
            raise ValueError(f"Qwen image binding differs: {sample_id}")

    priority = {row["sample_id"]: row for row in triage["rows"]}
    baseline_external = 0
    focused_second_pass = 0
    focused_local_resolved = 0
    focused_external_added = 0
    negative_external = 0
    priority_covered = 0
    uncovered = []
    focused_local_ms = 0.0
    baseline_local_ms = 0.0
    route_reasons = Counter()
    context_second_pass = 0
    context_local_resolved = 0
    context_external_added = 0
    context_negative_external = 0
    context_priority_covered = 0
    context_local_ms = 0.0
    context_uncovered = []
    for sample_id, base in baseline.items():
        focus = focused[sample_id]
        base_route = base["expert_shadow_route"].get("recommended") is True
        baseline_external += int(base_route)
        baseline_local_ms += float(base.get("latency_ms") or 0)
        second_pass = not base_route
        if second_pass:
            focused_second_pass += 1
            focused_local_ms += float(focus.get("latency_ms") or 0)
        suggestion = focus.get("suggestion") or {}
        assessments = suggestion.get("assessments") or {}
        slope = assessments.get("slope")
        geometry = suggestion.get("slope_geometry")
        difficult = (slope == "uncertain"
                     or geometry in {"partial_edge_slope", "occluded_slope"})
        # A positive label at a cropped or occluded edge is still a hard case.
        # Do not count it as both locally resolved and externally escalated.
        local_resolved = second_pass and slope == "present" and not difficult
        extra_external = second_pass and difficult
        focused_local_resolved += int(local_resolved)
        focused_external_added += int(extra_external)
        external = base_route or extra_external
        context_focus = (second_pass and bool(CONTEXT_PATTERN.search(
            str((base.get("suggestion") or {}).get("review_note") or ""))))
        context_second_pass += int(context_focus)
        if context_focus:
            context_local_ms += float(focus.get("latency_ms") or 0)
        context_resolved = context_focus and slope == "present" and not difficult
        context_extra_external = context_focus and difficult
        context_local_resolved += int(context_resolved)
        context_external_added += int(context_extra_external)
        context_external = base_route or context_extra_external
        if external:
            for reason in base["expert_shadow_route"].get("reasons") or []:
                if isinstance(reason, dict) and reason.get("code"):
                    route_reasons[reason["code"]] += 1
        if base.get("cohort") in NEGATIVE_COHORTS and external:
            negative_external += 1
        if base.get("cohort") in NEGATIVE_COHORTS and context_external:
            context_negative_external += 1
        if sample_id in priority:
            covered = external or local_resolved
            priority_covered += int(covered)
            if not covered:
                row = priority[sample_id]
                uncovered.append({
                    "sample_id": sample_id,
                    "source_image_sha256": row["source_image_sha256"],
                    "assistant_confidence": row["assistant_confidence"],
                    "local_qwen_slope": row["local_qwen_slope"],
                    "focused_qwen_slope": slope,
                    "focused_slope_geometry": geometry,
                    "difficulty_flags": sorted(set(
                        row.get("difficulty_flags") or [])),
                    "proposal_only": True,
                    "external_upload_authorized": False,
                    "human_reference_available": False,
                })
            context_covered = context_external or context_resolved
            context_priority_covered += int(context_covered)
            if not context_covered:
                row = priority[sample_id]
                context_uncovered.append({
                    "sample_id": sample_id,
                    "source_image_sha256": row["source_image_sha256"],
                    "assistant_confidence": row["assistant_confidence"],
                    "difficulty_flags": sorted(set(
                        row.get("difficulty_flags") or [])),
                    "proposal_only": True,
                    "external_upload_authorized": False,
                    "human_reference_available": False,
                })
    uncovered.sort(key=lambda row: row["sample_id"])
    context_uncovered.sort(key=lambda row: row["sample_id"])
    total_external = baseline_external + focused_external_added
    baseline_estimated_ms = baseline_local_ms + baseline_external * expert_latency_ms
    cascade_estimated_ms = (baseline_local_ms + focused_local_ms
                            + total_external * expert_latency_ms)
    context_external_count = baseline_external + context_external_added
    context_estimated_ms = (baseline_local_ms + context_local_ms
                            + context_external_count * expert_latency_ms)
    return {
        "schema_version": "labelagent.local-qwen-prompt-cascade-evaluation.v1",
        "status": "shadow_candidate_human_confirmation_pending",
        "baseline_sha256": _sha256(baseline_path),
        "focused_sha256": _sha256(focused_path),
        "assistant_triage_sha256": _sha256(triage_path),
        "review_manifest_sha256": triage_audit["review_manifest_sha256"],
        "source_sample_count": len(baseline),
        "baseline_external_count": baseline_external,
        "baseline_external_ratio": round(baseline_external / len(baseline), 6),
        "focused_second_pass_count": focused_second_pass,
        "focused_local_slope_resolved_count": focused_local_resolved,
        "focused_external_added_count": focused_external_added,
        "cascade_external_count": total_external,
        "cascade_external_ratio": round(total_external / len(baseline), 6),
        "priority_disagreement_count": len(priority),
        "priority_covered_count": priority_covered,
        "priority_uncovered_count": len(uncovered),
        "priority_coverage_ratio": round(priority_covered / len(priority), 6),
        "negative_control_count": sum(
            row.get("cohort") in NEGATIVE_COHORTS for row in baseline.values()),
        "negative_external_count": negative_external,
        "route_reason_counts": dict(sorted(route_reasons.items())),
        "uncovered_review_proposal": uncovered,
        "latency": {
            "baseline_local_observed_ms": round(baseline_local_ms, 3),
            "focused_second_pass_observed_ms": round(focused_local_ms, 3),
            "expert_estimate_ms_per_call": round(float(expert_latency_ms), 3),
            "baseline_sequential_estimate_ms": round(baseline_estimated_ms, 3),
            "cascade_sequential_estimate_ms": round(cascade_estimated_ms, 3),
            "cascade_minus_baseline_ms": round(
                cascade_estimated_ms - baseline_estimated_ms, 3),
        },
        "semantic_context_cascade": {
            "context_terms": ["container", "doorway", "entrance", "threshold"],
            "focused_second_pass_count": context_second_pass,
            "focused_local_slope_resolved_count": context_local_resolved,
            "focused_external_added_count": context_external_added,
            "cascade_external_count": context_external_count,
            "cascade_external_ratio": round(
                context_external_count / len(baseline), 6),
            "priority_covered_count": context_priority_covered,
            "priority_uncovered_count": len(context_uncovered),
            "priority_coverage_ratio": round(
                context_priority_covered / len(priority), 6),
            "negative_external_count": context_negative_external,
            "uncovered_review_proposal": context_uncovered,
            "focused_second_pass_observed_ms": round(context_local_ms, 3),
            "sequential_estimate_ms": round(context_estimated_ms, 3),
            "minus_baseline_ms": round(
                context_estimated_ms - baseline_estimated_ms, 3),
        },
        "dispatch_performed": False,
        "external_upload_performed": False,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "training_authorized": False,
        "automatic_approval": False,
        "test24_used": False,
        "production_deployed": False,
        "human_review_required": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--focused", required=True, type=Path)
    parser.add_argument("--triage", required=True, type=Path)
    parser.add_argument("--expert-latency-ms", type=float, default=21070.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate_prompt_cascade(
            args.packet, args.baseline, args.focused, args.triage,
            expert_latency_ms=args.expert_latency_ms)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise ValueError("output already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
