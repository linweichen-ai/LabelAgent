"""Audit mixed-hard experiment readiness without running models or uploads."""
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_mixed_hard_adjudication_workspace import (
    LABELS, TEMPLATE_SCHEMA, read_json, validate_packet,
    validate_submission,
)
from tools.evaluate_mixed_hard_three_arm import _validate_prediction_packet, sha256_file


SCHEMA_VERSION = "labelagent.mixed-hard-experiment-readiness.v1"


def _template_progress(payload, expected_count):
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if (payload.get("schema_version") != TEMPLATE_SCHEMA
            or not isinstance(rows, list) or len(rows) != expected_count):
        return None
    required = ("candidate_decision", "slope_presence", "class_conflict",
                "pallet_quality")
    completed = 0
    for row in rows:
        presence = row.get("class_presence")
        if (all(row.get(key) is not None for key in required)
                and isinstance(presence, dict) and set(presence) == set(LABELS)
                and all(value is not None for value in presence.values())
                and type(row.get("pallet_instance_count")) is int):
            completed += 1
    return {
        "state": "review_template_not_submission",
        "sample_count": expected_count,
        "completed_rows": completed,
        "remaining_rows": expected_count - completed,
        "expected_export_filename": "mixed-hard-adjudication.json",
        "review_interface": "reviewer.html",
    }


def _queue_status(path, expected_count):
    try:
        payload = read_json(path)
        samples = payload.get("samples") if isinstance(payload, dict) else None
        ready = (payload.get("schema_version")
                 == "labelagent.mixed-hard-expert-shadow-queue.v1"
                 and isinstance(samples, list) and len(samples) == expected_count
                 and payload.get("sample_count") == expected_count
                 and payload.get("candidate_only") is True
                 and payload.get("human_truth") is False
                 and payload.get("approved_for_training") is False
                 and payload.get("test24_used") is False)
        return {"ready": ready, "sample_count": len(samples or []),
                "sha256": sha256_file(path), "error": None if ready else "queue boundary differs"}
    except Exception as error:
        return {"ready": False, "sample_count": 0, "sha256": None,
                "error": f"{type(error).__name__}: {str(error)[:200]}"}


def _prediction_status(path, expected_count, *, external):
    if path is None or not Path(path).is_file():
        return {"ready": False, "sample_count": 0, "sha256": None,
                "error": "missing"}
    try:
        payload = read_json(path)
        rows = _validate_prediction_packet(payload, external=external)
        ready = len(rows) == expected_count
        return {"ready": ready, "sample_count": len(rows),
                "sha256": sha256_file(path), "error": None if ready else "sample count differs"}
    except Exception as error:
        return {"ready": False, "sample_count": 0,
                "sha256": sha256_file(path),
                "error": f"{type(error).__name__}: {str(error)[:200]}"}


def _authorization_status(path):
    if path is None or not Path(path).is_file():
        return {"ready": False, "authorized": False, "sha256": None,
                "schema_version": None, "error": "missing"}
    try:
        payload = read_json(path)
        schema = payload.get("schema_version")
        base_safe = (
                schema in {"labelagent.external-upload-authorization.v2",
                           "labelagent.external-upload-authorization.v3"}
                and payload.get("candidate_only") is True
                and payload.get("human_truth") is False
                and payload.get("approved_for_training") is False
                and payload.get("test24_used") is False)
        if schema == "labelagent.external-upload-authorization.v2":
            safe = (base_safe and payload.get("authorized_sample_count") == 17
                    and payload.get("authorized_upload_count") == 34
                    and payload.get("max_external_calls") == 34)
        else:
            runs = payload.get("authorized_runs")
            safe = (base_safe
                    and payload.get("authorized_unique_source_samples") == 34
                    and payload.get("authorized_upload_count") == 51
                    and payload.get("max_external_calls") == 51
                    and isinstance(runs, list) and len(runs) == 2
                    and {run.get("name") for run in runs}
                    == {"all34_source_640x512",
                        "hybrid17_resize_512x410"})
        authorized = (safe and payload.get("authorized") is True
                      and isinstance(payload.get("authorized_by"), str)
                      and bool(payload["authorized_by"].strip()))
        return {"ready": authorized, "authorized": authorized,
                "schema_version": schema,
                "safe_template": safe, "sha256": sha256_file(path),
                "error": None if authorized else "awaiting explicit user authorization"}
    except Exception as error:
        return {"ready": False, "authorized": False,
                "schema_version": None,
                "sha256": sha256_file(path),
                "error": f"{type(error).__name__}: {str(error)[:200]}"}


def audit_readiness(*, packet_dir, adjudication_path, local_path,
                    hybrid_queue_path, all_queue_path,
                    resolution_authorization_path,
                    all_external_path=None, source_ab_path=None,
                    resized_ab_path=None):
    manifest, manifest_sha = validate_packet(packet_dir)
    review = {"ready": False, "error": None, "manifest_sha256": manifest_sha}
    try:
        adjudication = read_json(adjudication_path)
        progress = _template_progress(adjudication, len(manifest["rows"]))
        if progress is not None:
            review.update(progress)
            review["error"] = (
                f"human review has not been exported; {progress['remaining_rows']} "
                "rows remain")
        else:
            validation = validate_submission(manifest, manifest_sha, adjudication)
            review.update(ready=validation.get("benchmark_reference_ready") is True,
                          validation=validation,
                          state=("benchmark_ready" if validation.get(
                              "benchmark_reference_ready") else
                                 "complete_but_uncertain"))
            if not review["ready"]:
                review["error"] = "declared review is incomplete or uncertain"
    except Exception as error:
        review["error"] = f"{type(error).__name__}: {str(error)[:200]}"
    local = _prediction_status(local_path, 34, external=False)
    hybrid_queue = _queue_status(hybrid_queue_path, 17)
    all_queue = _queue_status(all_queue_path, 34)
    all_external = _prediction_status(all_external_path, 34, external=True)
    authorization = _authorization_status(resolution_authorization_path)
    source_ab = _prediction_status(
        source_ab_path or all_external_path, 34, external=True)
    resized_ab = _prediction_status(resized_ab_path, 17, external=True)
    three_arm_ready = all(item["ready"] for item in
                          (review, local, hybrid_queue, all_queue, all_external))
    resolution_ab_ready = all(item["ready"] for item in
                              (review, hybrid_queue, authorization,
                               source_ab, resized_ab))
    blockers = []
    for name, item in (
            ("declared_review", review), ("local34", local),
            ("hybrid17_queue", hybrid_queue), ("all34_queue", all_queue),
            ("all34_external_results", all_external),
            ("resolution_ab_authorization", authorization),
            ("source34_reused_for_ab", source_ab),
            ("resized17_ab_results", resized_ab)):
        if not item["ready"]:
            blockers.append({"component": name, "reason": item.get("error")})
    combined_authorization = (authorization.get("schema_version")
                              == "labelagent.external-upload-authorization.v3")
    external_actions = ([
        "run authorized combined all34 source and hybrid17 resized shadows"
    ] if combined_authorization and (
        not all_external["ready"] or not resized_ab["ready"]) else [
        "run complete all-34 external shadow" if not all_external["ready"] else None,
        "run both authorized 17-frame resolution variants"
        if not source_ab["ready"] or not resized_ab["ready"] else None,
    ])
    next_actions = [item for item in [
        ("complete 34-row review in reviewer.html and export "
         "mixed-hard-adjudication.json") if not review["ready"] else None,
        "obtain explicit resolution A/B authorization" if not authorization["ready"] else None,
        *external_actions,
    ] if item is not None]
    return {
        "schema_version": SCHEMA_VERSION,
        "components": {
            "declared_review": review, "local34": local,
            "hybrid17_queue": hybrid_queue, "all34_queue": all_queue,
            "all34_external_results": all_external,
            "resolution_ab_authorization": authorization,
            "source34_reused_for_ab": source_ab,
            "resized17_ab_results": resized_ab,
        },
        "gates": {
            "three_arm_quality_evaluation_ready": three_arm_ready,
            "resolution_ab_quality_evaluation_ready": resolution_ab_ready,
            "production_deployment_ready": False,
        },
        "blocking_components": blockers,
        "next_actions": next_actions,
        "external_call_performed": False,
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "test24_used": False,
        "production_deployed": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--hybrid-queue", type=Path, required=True)
    parser.add_argument("--all-queue", type=Path, required=True)
    parser.add_argument("--resolution-authorization", type=Path, required=True)
    parser.add_argument("--all-external", type=Path)
    parser.add_argument("--source-ab", type=Path)
    parser.add_argument("--resized-ab", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_readiness(
        packet_dir=args.packet, adjudication_path=args.adjudication,
        local_path=args.local, hybrid_queue_path=args.hybrid_queue,
        all_queue_path=args.all_queue,
        resolution_authorization_path=args.resolution_authorization,
        all_external_path=args.all_external, source_ab_path=args.source_ab,
        resized_ab_path=args.resized_ab)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(args.output)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

