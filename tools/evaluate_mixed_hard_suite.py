"""Run all mixed-hard offline evaluations from one hash-bound input set."""
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_mixed_hard_adjudication_workspace import read_json, validate_packet
from tools.evaluate_expert_resolution_ab import evaluate_resolution_ab
from tools.evaluate_expert_resolution_cascade import evaluate_cascade
from tools.evaluate_mixed_hard_three_arm import evaluate_three_arm, sha256_file


SCHEMA_VERSION = "labelagent.mixed-hard-evaluation-suite.v1"


def _quality_summary(three_arm, resolution_ab, cascade):
    local = three_arm["arms"]["local_only"]
    hybrid = three_arm["arms"]["hybrid"]
    checks = {
        "hybrid_macro_f1_non_regression": (
            hybrid["macro_f1"] >= local["macro_f1"]),
        "hybrid_slope_recall_strict_improvement": (
            hybrid["slope_positive_recall"]
            > local["slope_positive_recall"]),
        "hybrid_slope_false_positives_non_increase": (
            hybrid["slope_false_positives"]
            <= local["slope_false_positives"]),
        "hybrid_pallet_count_mae_non_increase": (
            hybrid["pallet_count_mae"] <= local["pallet_count_mae"]),
        "hybrid_pallet_quality_accuracy_non_regression": (
            hybrid["pallet_quality_accuracy"]
            >= local["pallet_quality_accuracy"]),
        "resized_macro_f1_non_regression": (
            resolution_ab["delta_resize_minus_source"]["macro_f1"] >= 0),
        "resized_slope_recall_non_regression": (
            resolution_ab["delta_resize_minus_source"][
                "slope_positive_recall"] >= 0),
        "resized_slope_false_positives_non_increase": (
            resolution_ab["delta_resize_minus_source"][
                "slope_false_positives"] <= 0),
    }
    return {
        "measured_checks": checks,
        "measured_checks_all_passed": all(checks.values()),
        "cascade_external_calls": cascade["external_calls"],
        "cascade_escalated_samples": cascade["escalated_sample_count"],
        "unassessed_required_gates": [
            "pixel_mask_boundary_holes_and_merges",
            "vehicle_robot_class_conflict_non_increase",
            "timeout_failure_and_local_fallback",
            "cross_dataset_generalization",
            "workbench_progress_overlay_and_history_recovery",
        ],
        "deployment_gate_passed": False,
        "deployment_gate_status": "measured_semantic_checks_are_not_the_full_deployment_gate",
    }


def evaluate_suite(*, packet_dir, adjudication_path, local_path,
                   hybrid_queue_path, all_queue_path, source_results_path,
                   resized_results_path, min_confidence=0.85,
                   estimated_cost_per_call=None,
                   input_cost_per_million_tokens=None,
                   output_cost_per_million_tokens=None):
    manifest, manifest_sha = validate_packet(packet_dir)
    adjudication = read_json(adjudication_path)
    local = read_json(local_path)
    hybrid_queue = read_json(hybrid_queue_path)
    all_queue = read_json(all_queue_path)
    source = read_json(source_results_path)
    resized = read_json(resized_results_path)
    rates = dict(estimated_cost_per_call=estimated_cost_per_call,
                 input_cost_per_million_tokens=input_cost_per_million_tokens,
                 output_cost_per_million_tokens=output_cost_per_million_tokens)
    local_sha = sha256_file(local_path)
    hybrid_sha = sha256_file(hybrid_queue_path)
    all_sha = sha256_file(all_queue_path)
    three_arm = evaluate_three_arm(
        manifest, manifest_sha, adjudication, local, source, hybrid_queue,
        local_sha, **rates)
    resolution_ab = evaluate_resolution_ab(
        manifest, manifest_sha, adjudication, hybrid_queue, hybrid_sha,
        all_queue, all_sha, source, resized, **rates)
    cascade = evaluate_cascade(
        manifest, manifest_sha, adjudication, local, hybrid_queue, local_sha,
        all_queue, all_sha, hybrid_sha, source, resized,
        min_confidence=min_confidence, **rates)
    return {
        "schema_version": SCHEMA_VERSION,
        "review_manifest_sha256": manifest_sha,
        "adjudication_sha256": sha256_file(adjudication_path),
        "local_results_sha256": local_sha,
        "hybrid_queue_sha256": hybrid_sha,
        "all_queue_sha256": all_sha,
        "source_results_sha256": sha256_file(source_results_path),
        "resized_results_sha256": sha256_file(resized_results_path),
        "three_arm": three_arm,
        "resolution_ab": resolution_ab,
        "resolution_cascade": cascade,
        "gate_summary": _quality_summary(three_arm, resolution_ab, cascade),
        "model_calls_performed": 0,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "test24_used": False,
        "production_deployed": False,
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
    result = evaluate_suite(
        packet_dir=args.packet, adjudication_path=args.adjudication,
        local_path=args.local, hybrid_queue_path=args.hybrid_queue,
        all_queue_path=args.all_queue,
        source_results_path=args.source_results,
        resized_results_path=args.resized_results,
        min_confidence=args.minimum_confidence,
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
