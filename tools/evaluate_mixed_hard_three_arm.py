"""Evaluate local, all-expert and hybrid predictions on declared review labels."""
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_mixed_hard_adjudication_workspace import (
    LABELS, read_json, validate_packet, validate_submission,
)


SCHEMA_VERSION = "labelagent.mixed-hard-three-arm-evaluation.v1"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_prediction_packet(packet, *, external=False):
    rows = packet.get("rows") if isinstance(packet, dict) else None
    if (packet.get("candidate_only") is not True
            or packet.get("human_truth") is not False
            or packet.get("approved_for_training") is not False
            or packet.get("test24_used") is not False
            or not isinstance(rows, list) or not rows):
        raise ValueError("prediction packet safety boundary differs")
    expected_schema = ("labelagent.mixed-hard-external-shadow.v1" if external
                       else "labelagent.local-qwen-mixed-hard-suggestions.v1")
    if packet.get("schema_version") != expected_schema:
        raise ValueError("prediction packet schema differs")
    if (packet.get("status") != "completed"
            or packet.get("samples_passed") != len(rows)
            or packet.get("samples_failed") != 0):
        raise ValueError("prediction packet must be complete with zero failures")
    key = "expert_result" if external else "suggestion"
    mapped = {}
    for row in rows:
        sample_id, prediction = row.get("sample_id"), row.get(key)
        if (not isinstance(sample_id, str) or sample_id in mapped
                or not isinstance(prediction, dict)
                or row.get("candidate_only") is not True
                or row.get("human_truth") is not False
                or row.get("approved_for_training") is not False
                or row.get("test24_used") is not False):
            raise ValueError("prediction row identity or safety boundary differs")
        assessments = prediction.get("assessments")
        if (not isinstance(assessments, dict)
                or set(assessments) != set(LABELS)
                or any(value not in {"present", "absent", "uncertain"}
                       for value in assessments.values())):
            raise ValueError("prediction assessments differ from the benchmark schema")
        latency = row.get("latency_ms")
        if (not isinstance(latency, (int, float)) or isinstance(latency, bool)
                or not math.isfinite(float(latency)) or float(latency) < 0):
            raise ValueError("prediction latency is invalid")
        mapped[sample_id] = row
    return mapped


def _binary_metrics(reference, predictions, label):
    tp = fp = fn = tn = 0
    for sample_id, truth in reference.items():
        actual = truth["class_presence"][label] == "present"
        predicted = predictions[sample_id]["assessments"][label] == "present"
        if actual and predicted:
            tp += 1
        elif not actual and predicted:
            fp += 1
        elif actual and not predicted:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else (1.0 if not fn else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 6), "recall": round(recall, 6),
            "f1": round(f1, 6)}


def _p95(values):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _normalize_pallet_quality(value):
    return {
        "clean": "complete",
        "no_pallet": "not_applicable",
        "holes_or_notches": "hole_error",
        "merged_instances": "merged_instances",
    }.get(value, "unmapped_or_uncertain")


def _external_usage(rows, sample_ids):
    observed, missing = [], []
    for sample_id in sample_ids:
        usage = (rows[sample_id].get("expert_metadata", {})
                 .get("external_vlm", {}).get("usage", {}))
        if all(type(usage.get(key)) is int and usage[key] >= 0 for key in
               ("prompt_tokens", "completion_tokens", "total_tokens")):
            observed.append(usage)
        else:
            missing.append(sample_id)
    return {
        "samples_with_usage": len(observed),
        "samples_missing_usage": len(missing),
        "complete": not missing,
        "prompt_tokens": sum(row["prompt_tokens"] for row in observed),
        "completion_tokens": sum(row["completion_tokens"] for row in observed),
        "total_tokens": sum(row["total_tokens"] for row in observed),
    }


def _arm_metrics(reference, predictions, latencies, external_calls,
                 external_usage, estimated_cost_per_call=None,
                 input_cost_per_million_tokens=None,
                 output_cost_per_million_tokens=None):
    per_label = {label: _binary_metrics(reference, predictions, label)
                 for label in LABELS}
    macro_f1 = statistics.mean(item["f1"] for item in per_label.values())
    pallet_abs = []
    pallet_exact = 0
    pallet_quality_correct = 0
    for sample_id, truth in reference.items():
        prediction = predictions[sample_id]
        predicted_count = prediction.get("pallet_instance_count")
        if type(predicted_count) is not int:
            raise ValueError("prediction pallet count is invalid")
        pallet_abs.append(abs(predicted_count - truth["pallet_instance_count"]))
        pallet_exact += predicted_count == truth["pallet_instance_count"]
        pallet_quality_correct += (
            _normalize_pallet_quality(prediction.get("pallet_quality"))
            == truth["pallet_quality"])
    token_rates_ready = (input_cost_per_million_tokens is not None
                         and output_cost_per_million_tokens is not None)
    if external_calls == 0:
        cost, cost_status, cost_basis = 0.0, "no_external_calls", "none"
    elif token_rates_ready and external_usage["complete"]:
        cost = round(
            external_usage["prompt_tokens"] / 1_000_000
            * input_cost_per_million_tokens
            + external_usage["completion_tokens"] / 1_000_000
            * output_cost_per_million_tokens, 8)
        cost_status, cost_basis = "estimated", "token_usage"
    elif estimated_cost_per_call is not None:
        cost = round(external_calls * estimated_cost_per_call, 8)
        cost_status, cost_basis = "estimated", "per_call_fallback"
    else:
        cost, cost_status, cost_basis = (
            None, "unknown_requires_billing_rate", "unavailable")
    return {
        "per_label": per_label,
        "macro_f1": round(macro_f1, 6),
        "slope_positive_recall": per_label["slope"]["recall"],
        "slope_false_positives": per_label["slope"]["fp"],
        "pallet_count_mae": round(statistics.mean(pallet_abs), 6),
        "pallet_count_exact_rate": round(pallet_exact / len(reference), 6),
        "pallet_quality_accuracy": round(
            pallet_quality_correct / len(reference), 6),
        "latency_mean_ms": round(statistics.mean(latencies), 3),
        "latency_p95_ms": round(_p95(latencies), 3),
        "external_calls": external_calls,
        "external_call_ratio": round(external_calls / len(reference), 6),
        "external_usage": external_usage,
        "estimated_cost": cost,
        "cost_status": cost_status,
        "cost_basis": cost_basis,
    }


def evaluate_three_arm(manifest, manifest_sha256, adjudication, local_packet,
                       external_packet, hybrid_queue,
                       local_packet_sha256,
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
        raise ValueError("declared review is incomplete or uncertain and cannot score the benchmark")
    reference = {row["sample_id"]: row for row in adjudication["rows"]}
    local_rows = _validate_prediction_packet(local_packet)
    external_rows = _validate_prediction_packet(external_packet, external=True)
    if set(reference) != set(local_rows) or set(reference) != set(external_rows):
        raise ValueError("prediction samples differ from the declared review")
    if local_packet.get("manifest_sha256") != manifest_sha256:
        raise ValueError("local suggestions are bound to another review manifest")
    if external_packet.get("manifest_sha256") != manifest_sha256:
        raise ValueError("external suggestions are bound to another review manifest")
    for sample_id, truth in reference.items():
        expected_sha = truth["source_image_sha256"]
        if (local_rows[sample_id].get("source_image_sha256") != expected_sha
                or external_rows[sample_id].get("source_image_sha256") != expected_sha):
            raise ValueError("prediction image hashes differ from the declared review")
    hybrid_samples = hybrid_queue.get("samples") if isinstance(hybrid_queue, dict) else None
    if (hybrid_queue.get("schema_version")
            != "labelagent.mixed-hard-expert-shadow-queue.v1"
            or hybrid_queue.get("candidate_only") is not True
            or hybrid_queue.get("human_truth") is not False
            or hybrid_queue.get("approved_for_training") is not False
            or hybrid_queue.get("test24_used") is not False
            or not isinstance(hybrid_samples, list)
            or hybrid_queue.get("sample_count") != len(hybrid_samples)
            or hybrid_queue.get("source_sample_count") != len(reference)
            or hybrid_queue.get("source_suggestions_sha256") != local_packet_sha256):
        raise ValueError("hybrid queue safety boundary differs")
    hybrid_ids = {row.get("sample_id") for row in hybrid_samples}
    if len(hybrid_ids) != len(hybrid_samples) or not hybrid_ids.issubset(reference):
        raise ValueError("hybrid queue samples differ")
    local_predictions = {sample_id: row["suggestion"]
                         for sample_id, row in local_rows.items()}
    external_predictions = {sample_id: row["expert_result"]
                            for sample_id, row in external_rows.items()}
    hybrid_predictions = {
        sample_id: (external_predictions[sample_id]
                    if sample_id in hybrid_ids else local_predictions[sample_id])
        for sample_id in reference
    }
    local_latency = {sample_id: float(row["latency_ms"])
                     for sample_id, row in local_rows.items()}
    external_latency = {sample_id: float(row["latency_ms"])
                        for sample_id, row in external_rows.items()}
    no_external_usage = _external_usage(external_rows, ())
    all_external_usage = _external_usage(external_rows, reference)
    hybrid_external_usage = _external_usage(external_rows, hybrid_ids)
    arms = {
        "local_only": _arm_metrics(
            reference, local_predictions, list(local_latency.values()), 0,
            no_external_usage, estimated_cost_per_call,
            input_cost_per_million_tokens, output_cost_per_million_tokens),
        "all_external": _arm_metrics(
            reference, external_predictions, list(external_latency.values()),
            len(reference), all_external_usage, estimated_cost_per_call,
            input_cost_per_million_tokens, output_cost_per_million_tokens),
        "hybrid": _arm_metrics(
            reference, hybrid_predictions,
            [local_latency[sample_id]
             + (external_latency[sample_id] if sample_id in hybrid_ids else 0)
             for sample_id in reference],
            len(hybrid_ids), hybrid_external_usage, estimated_cost_per_call,
            input_cost_per_million_tokens, output_cost_per_million_tokens),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "review_manifest_sha256": manifest_sha256,
        "reviewer_declared_id": adjudication["reviewer_declared_id"],
        "sample_count": len(reference),
        "reference_kind": "declared_human_review_not_pixel_truth",
        "arms": arms,
        "deployment_gate_passed": False,
        "deployment_gate_status": "requires_explicit_gate_comparison",
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "test24_used": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--hybrid-queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--estimated-cost-per-call", type=float)
    parser.add_argument("--input-cost-per-million-tokens", type=float)
    parser.add_argument("--output-cost-per-million-tokens", type=float)
    args = parser.parse_args(argv)
    manifest, digest = validate_packet(args.packet)
    result = evaluate_three_arm(
        manifest, digest, read_json(args.adjudication), read_json(args.local),
        read_json(args.external), read_json(args.hybrid_queue),
        sha256_file(args.local),
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
