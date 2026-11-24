"""Evaluate a 512-first, 640-on-uncertainty expert cascade offline.

This tool never calls a model.  It reuses the frozen local, 512x410 and
640x512 shadow packets so the cascade can be assessed without another upload.
"""
import argparse
import json
import math
from pathlib import Path

from tools.build_mixed_hard_adjudication_workspace import (
    read_json, validate_packet, validate_submission,
)
from tools.evaluate_expert_resolution_ab import _validate_packet_binding
from tools.evaluate_mixed_hard_three_arm import (
    _arm_metrics, _prediction_image_sha, _validate_prediction_packet,
    sha256_file,
)


SCHEMA_VERSION = "labelagent.expert-resolution-cascade-evaluation.v1"
CRITICAL_LABELS = (
    "slope", "floor", "车辆", "other_robot", "ego", "pallet", "cargo_floor",
)


def escalation_reasons(local_result, resized_result, *, min_confidence=0.85):
    """Return stable, explainable reasons for requesting the source image."""
    if (not isinstance(min_confidence, (int, float))
            or isinstance(min_confidence, bool)
            or not math.isfinite(float(min_confidence))
            or not 0 <= float(min_confidence) <= 1):
        raise ValueError("minimum confidence must be between zero and one")
    reasons = []
    if float(resized_result["overall_confidence"]) < float(min_confidence):
        reasons.append("resized_low_confidence")
    uncertain = sorted(label for label, value in
                       resized_result["assessments"].items()
                       if value == "uncertain")
    if uncertain:
        reasons.append("resized_uncertain:" + ",".join(uncertain))
    conflicts = sorted(value for value in resized_result["class_conflicts"]
                       if value != "none")
    if conflicts:
        reasons.append("resized_class_conflict:" + ",".join(conflicts))
    if resized_result["unknown_candidates"]:
        reasons.append("resized_unknown_candidate")
    if resized_result["slope_geometry"] in {
            "partial_edge_slope", "occluded_slope", "uncertain"}:
        reasons.append("resized_difficult_slope_geometry")
    if resized_result["pallet_quality"] == "uncertain":
        reasons.append("resized_uncertain_pallet_quality")
    disagreements = [label for label in CRITICAL_LABELS
                     if local_result["assessments"][label]
                     != resized_result["assessments"][label]]
    if disagreements:
        reasons.append("local_resized_disagreement:" + ",".join(disagreements))
    if (local_result["pallet_instance_count"]
            != resized_result["pallet_instance_count"]):
        reasons.append("local_resized_pallet_count_disagreement")
    return reasons


def _usage_for_calls(calls):
    observed, missing = [], []
    for call_name, sample_id, row in calls:
        usage = (row.get("expert_metadata", {}).get("external_vlm", {})
                 .get("usage", {}))
        if all(type(usage.get(key)) is int and usage[key] >= 0 for key in
               ("prompt_tokens", "completion_tokens", "total_tokens")):
            observed.append(usage)
        else:
            missing.append(f"{call_name}:{sample_id}")
    return {
        "samples_with_usage": len(observed),
        "samples_missing_usage": len(missing),
        "complete": not missing,
        "prompt_tokens": sum(row["prompt_tokens"] for row in observed),
        "completion_tokens": sum(row["completion_tokens"] for row in observed),
        "total_tokens": sum(row["total_tokens"] for row in observed),
    }


def evaluate_cascade(manifest, manifest_sha256, adjudication, local_packet,
                     hybrid_queue, local_packet_sha256, all_queue,
                     all_queue_sha256, hybrid_queue_sha256, source_packet,
                     resized_packet, *, min_confidence=0.85,
                     estimated_cost_per_call=None,
                     input_cost_per_million_tokens=None,
                     output_cost_per_million_tokens=None):
    rates = (estimated_cost_per_call, input_cost_per_million_tokens,
             output_cost_per_million_tokens)
    if any(value is not None and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value)) or value < 0) for value in rates):
        raise ValueError("cost rates must be finite non-negative numbers")
    if ((input_cost_per_million_tokens is None)
            != (output_cost_per_million_tokens is None)):
        raise ValueError("input and output token rates must be provided together")
    validation = validate_submission(manifest, manifest_sha256, adjudication)
    if validation.get("benchmark_reference_ready") is not True:
        raise ValueError("declared review is incomplete or uncertain")
    reference = {row["sample_id"]: row for row in adjudication["rows"]}
    local_rows = _validate_prediction_packet(local_packet)
    source_rows = _validate_prediction_packet(source_packet, external=True)
    resized_rows = _validate_prediction_packet(resized_packet, external=True)
    samples = hybrid_queue.get("samples") if isinstance(hybrid_queue, dict) else None
    all_samples = all_queue.get("samples") if isinstance(all_queue, dict) else None
    if (hybrid_queue.get("schema_version")
            != "labelagent.mixed-hard-expert-shadow-queue.v1"
            or all_queue.get("schema_version")
            != "labelagent.mixed-hard-expert-shadow-queue.v1"
            or not isinstance(samples, list) or not isinstance(all_samples, list)
            or hybrid_queue.get("sample_count") != len(samples)
            or hybrid_queue.get("source_sample_count") != len(reference)
            or all_queue.get("sample_count") != len(all_samples)
            or hybrid_queue.get("source_suggestions_sha256")
            != local_packet_sha256
            or hybrid_queue.get("candidate_only") is not True
            or hybrid_queue.get("human_truth") is not False
            or hybrid_queue.get("approved_for_training") is not False
            or hybrid_queue.get("test24_used") is not False
            or all_queue.get("candidate_only") is not True
            or all_queue.get("human_truth") is not False
            or all_queue.get("approved_for_training") is not False
            or all_queue.get("test24_used") is not False):
        raise ValueError("hybrid queue safety boundary differs")
    hybrid_ids = {row.get("sample_id") for row in samples}
    all_ids = {row.get("sample_id") for row in all_samples}
    if (len(hybrid_ids) != len(samples) or None in hybrid_ids
            or set(reference) != set(local_rows) or set(reference) != all_ids
            or set(source_rows) != all_ids or set(resized_rows) != hybrid_ids):
        raise ValueError("cascade sample identities differ")
    if local_packet.get("manifest_sha256") != manifest_sha256:
        raise ValueError("local packet manifest binding differs")
    _validate_packet_binding(
        source_packet, manifest_sha256=manifest_sha256,
        queue_sha256=all_queue_sha256, variant_name="source_640x512",
        variant_mode="source", run_name="all34_source_640x512")
    _validate_packet_binding(
        resized_packet, manifest_sha256=manifest_sha256,
        queue_sha256=hybrid_queue_sha256, variant_name="resize_512x410",
        variant_mode="resize_lanczos_jpeg",
        run_name="hybrid17_resize_512x410")
    if source_packet.get("authorization_sha256") != resized_packet.get(
            "authorization_sha256"):
        raise ValueError("cascade packets use different authorizations")
    for sample_id, truth in reference.items():
        expected_sha = truth["source_image_sha256"]
        if (_prediction_image_sha(local_rows[sample_id]) != expected_sha
                or _prediction_image_sha(
                    source_rows[sample_id], external=True) != expected_sha):
            raise ValueError("cascade prediction image hashes differ")
        if sample_id in hybrid_ids and _prediction_image_sha(
                resized_rows[sample_id], external=True) != expected_sha:
            raise ValueError("cascade resized image hash differs")
    local_predictions = {key: row["suggestion"] for key, row in local_rows.items()}
    source_predictions = {key: row["expert_result"] for key, row in source_rows.items()}
    resized_predictions = {key: row["expert_result"] for key, row in resized_rows.items()}
    decisions, escalated = [], set()
    for sample_id in sorted(hybrid_ids):
        reasons = escalation_reasons(
            local_predictions[sample_id], resized_predictions[sample_id],
            min_confidence=min_confidence)
        if reasons:
            escalated.add(sample_id)
        decisions.append({
            "sample_id": sample_id,
            "route": "source_640x512" if reasons else "accept_resize_512x410",
            "reasons": reasons,
        })
    predictions = {}
    latency = []
    calls = []
    for sample_id in reference:
        local_ms = float(local_rows[sample_id]["latency_ms"])
        if sample_id not in hybrid_ids:
            predictions[sample_id] = local_predictions[sample_id]
            latency.append(local_ms)
            continue
        resized_row = resized_rows[sample_id]
        calls.append(("resize_512x410", sample_id, resized_row))
        elapsed = local_ms + float(resized_row["latency_ms"])
        if sample_id in escalated:
            source_row = source_rows[sample_id]
            calls.append(("source_640x512", sample_id, source_row))
            elapsed += float(source_row["latency_ms"])
            predictions[sample_id] = source_predictions[sample_id]
        else:
            predictions[sample_id] = resized_predictions[sample_id]
        latency.append(elapsed)
    usage = _usage_for_calls(calls)
    metrics = _arm_metrics(
        reference, predictions, latency, len(calls), usage,
        estimated_cost_per_call, input_cost_per_million_tokens,
        output_cost_per_million_tokens)
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": {
            "first_pass": "resize_512x410",
            "escalation": "source_640x512",
            "minimum_confidence": float(min_confidence),
            "critical_labels": list(CRITICAL_LABELS),
        },
        "sample_count": len(reference),
        "hybrid_sample_count": len(hybrid_ids),
        "escalated_sample_count": len(escalated),
        "external_calls": len(calls),
        "maximum_external_calls": len(hybrid_ids) * 2,
        "decisions": decisions,
        "metrics": metrics,
        "quality_scope": "semantic_presence_count_and_quality_not_pixel_masks",
        "deployment_gate_passed": False,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "test24_used": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--hybrid-queue", type=Path, required=True)
    parser.add_argument("--all-queue", type=Path, required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--resized-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-confidence", type=float, default=0.85)
    parser.add_argument("--estimated-cost-per-call", type=float)
    parser.add_argument("--input-cost-per-million-tokens", type=float)
    parser.add_argument("--output-cost-per-million-tokens", type=float)
    args = parser.parse_args(argv)
    manifest, manifest_sha = validate_packet(args.packet)
    local = read_json(args.local)
    hybrid, all_queue = read_json(args.hybrid_queue), read_json(args.all_queue)
    result = evaluate_cascade(
        manifest, manifest_sha, read_json(args.adjudication), local, hybrid,
        sha256_file(args.local), all_queue, sha256_file(args.all_queue),
        sha256_file(args.hybrid_queue), read_json(args.source_results),
        read_json(args.resized_results), min_confidence=args.minimum_confidence,
        estimated_cost_per_call=args.estimated_cost_per_call,
        input_cost_per_million_tokens=args.input_cost_per_million_tokens,
        output_cost_per_million_tokens=args.output_cost_per_million_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
