"""Evaluate authorized source-versus-resized external expert results."""
import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_mixed_hard_adjudication_workspace import (
    read_json, validate_packet, validate_submission,
)
from tools.evaluate_mixed_hard_three_arm import (
    _arm_metrics, _external_usage, _validate_prediction_packet, sha256_file,
)


SCHEMA_VERSION = "labelagent.expert-resolution-ab-evaluation.v1"


def _validate_packet_binding(packet, *, manifest_sha256, queue_sha256,
                             variant_name, variant_mode, run_name):
    if (packet.get("manifest_sha256") != manifest_sha256
            or packet.get("queue_sha256") != queue_sha256
            or packet.get("authorization_schema")
            != "labelagent.external-upload-authorization.v3"
            or packet.get("authorized_upload_count") != 51
            or packet.get("authorization_run", {}).get("name") != run_name):
        raise ValueError("external result binding differs from resolution A/B")
    variant = packet.get("image_variant")
    if (not isinstance(variant, dict) or variant.get("name") != variant_name
            or variant.get("mode") != variant_mode):
        raise ValueError("external result image variant differs")


def evaluate_resolution_ab(manifest, manifest_sha256, adjudication,
                           hybrid_queue, hybrid_queue_sha256,
                           all_queue, all_queue_sha256,
                           source_packet, resized_packet,
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
    samples = (hybrid_queue.get("samples")
               if isinstance(hybrid_queue, dict) else None)
    if (hybrid_queue.get("schema_version")
            != "labelagent.mixed-hard-expert-shadow-queue.v1"
            or not isinstance(samples, list) or len(samples) != 17
            or hybrid_queue.get("sample_count") != len(samples)
            or hybrid_queue.get("candidate_only") is not True
            or hybrid_queue.get("human_truth") is not False
            or hybrid_queue.get("approved_for_training") is not False
            or hybrid_queue.get("test24_used") is not False):
        raise ValueError("resolution A/B queue safety boundary differs")
    queue_ids = {row.get("sample_id") for row in samples}
    if len(queue_ids) != len(samples) or None in queue_ids:
        raise ValueError("resolution A/B queue identities differ")
    all_samples = all_queue.get("samples") if isinstance(all_queue, dict) else None
    if (all_queue.get("schema_version")
            != "labelagent.mixed-hard-expert-shadow-queue.v1"
            or not isinstance(all_samples, list) or len(all_samples) != 34
            or all_queue.get("sample_count") != 34
            or all_queue.get("candidate_only") is not True
            or all_queue.get("human_truth") is not False
            or all_queue.get("approved_for_training") is not False
            or all_queue.get("test24_used") is not False):
        raise ValueError("all34 source queue safety boundary differs")
    all_queue_ids = {row.get("sample_id") for row in all_samples}
    if (len(all_queue_ids) != 34 or None in all_queue_ids
            or not queue_ids.issubset(all_queue_ids)):
        raise ValueError("hybrid17 queue is not a subset of all34 source queue")
    source_rows = _validate_prediction_packet(source_packet, external=True)
    resized_rows = _validate_prediction_packet(resized_packet, external=True)
    if set(source_rows) != all_queue_ids or set(resized_rows) != queue_ids:
        raise ValueError("resolution A/B result samples differ from queue")
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
        raise ValueError("resolution A/B results use different authorizations")
    all_reference = {row["sample_id"]: row for row in adjudication["rows"]}
    if not queue_ids.issubset(all_reference):
        raise ValueError("resolution A/B queue differs from declared review")
    reference = {sample_id: all_reference[sample_id] for sample_id in queue_ids}
    for sample_id, truth in reference.items():
        expected_sha = truth["source_image_sha256"]
        source_row, resized_row = source_rows[sample_id], resized_rows[sample_id]
        if (source_row.get("source_image_sha256") != expected_sha
                or resized_row.get("source_image_sha256") != expected_sha):
            raise ValueError("resolution A/B source image hashes differ")
        source_upload = source_row.get("upload_image") or {}
        resized_upload = resized_row.get("upload_image") or {}
        resized_sha = resized_upload.get("upload_image_sha256")
        if (source_upload.get("upload_image_sha256") != expected_sha
                or source_upload.get("mode") != "source"
                or resized_upload.get("name") != "resize_512x410"
                or resized_upload.get("mode") != "resize_lanczos_jpeg"
                or resized_upload.get("width") != 512
                or resized_upload.get("height") != 410
                or not isinstance(resized_sha, str) or len(resized_sha) != 64):
            raise ValueError("resolution A/B upload image attestations differ")
    def metrics(rows):
        predictions = {sample_id: row["expert_result"]
                       for sample_id, row in rows.items()}
        latencies = [float(row["latency_ms"]) for row in rows.values()]
        usage = _external_usage(rows, reference)
        return _arm_metrics(
            reference, predictions, latencies, len(reference), usage,
            estimated_cost_per_call, input_cost_per_million_tokens,
            output_cost_per_million_tokens)
    source_subset = {sample_id: source_rows[sample_id] for sample_id in queue_ids}
    source_metrics, resized_metrics = metrics(source_subset), metrics(resized_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_count": len(reference),
        "reference_kind": "declared_human_review_not_pixel_truth",
        "source_640x512": source_metrics,
        "resize_512x410": resized_metrics,
        "delta_resize_minus_source": {
            "macro_f1": round(resized_metrics["macro_f1"]
                              - source_metrics["macro_f1"], 6),
            "slope_positive_recall": round(
                resized_metrics["slope_positive_recall"]
                - source_metrics["slope_positive_recall"], 6),
            "slope_false_positives": (
                resized_metrics["slope_false_positives"]
                - source_metrics["slope_false_positives"]),
            "latency_mean_ms": round(resized_metrics["latency_mean_ms"]
                                     - source_metrics["latency_mean_ms"], 3),
            "total_tokens": (resized_metrics["external_usage"]["total_tokens"]
                             - source_metrics["external_usage"]["total_tokens"]),
        },
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
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--all-queue", type=Path, required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--resized-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--estimated-cost-per-call", type=float)
    parser.add_argument("--input-cost-per-million-tokens", type=float)
    parser.add_argument("--output-cost-per-million-tokens", type=float)
    args = parser.parse_args(argv)
    manifest, manifest_sha = validate_packet(args.packet)
    queue, all_queue = read_json(args.queue), read_json(args.all_queue)
    result = evaluate_resolution_ab(
        manifest, manifest_sha, read_json(args.adjudication),
        queue, sha256_file(args.queue), all_queue, sha256_file(args.all_queue),
        read_json(args.source_results),
        read_json(args.resized_results), args.estimated_cost_per_call,
        args.input_cost_per_million_tokens,
        args.output_cost_per_million_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

