"""Compile a fail-closed LabelAgent batch plan without calling any model.

This module is deliberately CPU-only and side-effect free.  It consumes an
already hashed candidate manifest plus a one-shot resource readback and emits a
plan object.  It has no SAM/VLM client, CUDA import, process launcher, or file
writer.  A separate explicitly authorized executor may consume an admitted
plan in the future; this module can never do so.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


HASH = re.compile(r"^[0-9a-f]{64}$")
VERSION = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
PIPELINE = ("sam_candidate", "vlm_review", "agent_action", "vlm_recheck")
REVIEW_ROUTES = {
    "unknown_or_new_label": "p1_high",
    "low_confidence": "p2_normal",
    "vlm_sam_conflict": "p1_high",
    "qc_failed": "p0_critical",
}
REVIEW_SOURCE_STAGES = {
    "unknown_or_new_label": "vlm_review",
    "low_confidence": "vlm_review",
    "vlm_sam_conflict": "vlm_recheck",
    "qc_failed": "quality_check",
}


def canonical_sha256(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(name, value):
    if not isinstance(value, str) or not HASH.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def validate_candidate_manifest(manifest):
    if manifest.get("schema_version") != "labelagent.candidate-batch-manifest.v2":
        raise ValueError("candidate manifest v2 is required")
    if manifest.get("split") != "train72":
        raise ValueError("test24_is_evaluation_only_candidate_generation_forbidden")
    if manifest.get("split_policy", {}).get("evaluation_only") is not False:
        raise ValueError("train72 manifest must not be evaluation-only")
    if manifest.get("write_policy") != {
        "mode": "append_only_explicit_new_version",
        "candidate_root": manifest.get("write_policy", {}).get("candidate_root"),
        "overwrite_existing_candidate": False,
        "overwrite_human_truth": False,
    }:
        raise ValueError("append-only candidate write policy is required")
    if manifest.get("entry_count") != len(manifest.get("entries", [])):
        raise ValueError("manifest entry_count mismatch")
    if manifest.get("review_routing", {}).get("automatic_truth_promotion") is not False:
        raise ValueError("automatic truth promotion is forbidden")
    manifest_body = {
        key: value for key, value in manifest.items()
        if key != "manifest_content_sha256"
    }
    if manifest.get("manifest_content_sha256") != canonical_sha256(manifest_body):
        raise ValueError("candidate manifest content SHA-256 mismatch")

    for index, entry in enumerate(manifest["entries"]):
        prefix = f"entries[{index}]"
        if entry.get("split") != "train72" or entry.get("evaluation_only") is not False:
            raise ValueError(f"{prefix} is not an eligible train72 record")
        for field in (
            "image_sha256", "preannotation_sha256", "sam3_model_sha256",
            "vlm_config_sha256", "pipeline_config_sha256", "input_id",
            "idempotency_key",
        ):
            _require_hash(f"{prefix}.{field}", entry.get(field))
        if not all(entry.get(field) is False for field in (
            "truth_eligible", "approved_for_training",
            "automatic_truth_promotion", "training_return_allowed",
        )):
            raise ValueError(f"{prefix} violates candidate/truth separation")
        if entry.get("candidate_only") is not True:
            raise ValueError(f"{prefix} must remain candidate-only")
        if entry.get("output_write_mode") != "append_only_explicit_new_version":
            raise ValueError(f"{prefix} is not append-only")
        if not str(entry.get("output_candidate_path", "")).startswith(
            "candidate_outputs/"
        ):
            raise ValueError(f"{prefix} has an unsafe candidate path")


def validate_resource_readback(readback):
    if readback.get("schema_version") != "labelagent.batch-resource-readback.v1":
        raise ValueError("batch resource readback v1 is required")
    if readback.get("mode") != "one_shot_read_only_no_polling":
        raise ValueError("resource readback must be one-shot and read-only")
    execution = readback.get("execution", {})
    expected = {
        "gpu_model_calls": 0,
        "inference_service_calls": 0,
        "processes_created": False,
        "remote_mutation_performed": False,
    }
    for field, value in expected.items():
        if execution.get(field) != value:
            raise ValueError(f"unsafe resource readback execution field: {field}")
    admission = readback.get("admission", {})
    if not isinstance(admission.get("candidate_generation_allowed"), bool):
        raise ValueError("resource admission decision is missing")
    if admission["candidate_generation_allowed"] and admission.get("blockers"):
        raise ValueError("resource admission cannot pass with blockers")
    if not admission["candidate_generation_allowed"] and not admission.get("blockers"):
        raise ValueError("rejected resource admission requires blockers")
    readback_body = {
        key: value for key, value in readback.items()
        if key != "readback_content_sha256"
    }
    if readback.get("readback_content_sha256") != canonical_sha256(readback_body):
        raise ValueError("resource readback content SHA-256 mismatch")


def _stage_records(admitted):
    records = []
    for index, stage in enumerate(PIPELINE):
        if admitted:
            status = "planned"
            reason = None
        elif index == 0:
            status = "blocked"
            reason = "resource_gate_not_passed"
        else:
            status = "not_started"
            reason = "upstream_not_run_due_to_resource_gate"
        records.append({
            "stage": stage,
            "status": status,
            "attempt": 0,
            "failure_reason": reason,
            "artifact_refs": [],
        })
    return records


def build_execution_plan(manifest, resource_readback, plan_version, generated_at):
    validate_candidate_manifest(manifest)
    validate_resource_readback(resource_readback)
    if not VERSION.fullmatch(plan_version):
        raise ValueError("plan_version is not a safe append-only namespace")

    admitted = resource_readback["admission"]["candidate_generation_allowed"]
    records = []
    for entry in manifest["entries"]:
        record_identity = {
            "plan_version": plan_version,
            "input_id": entry["input_id"],
            "idempotency_key": entry["idempotency_key"],
        }
        records.append({
            "record_id": canonical_sha256(record_identity),
            "idempotency_key": entry["idempotency_key"],
            "split": "train72",
            "sequence": entry["sequence"],
            "frame_id": entry["frame_id"],
            "camera": entry["camera"],
            "source_image_sha256": entry["image_sha256"],
            "preannotation_sha256": entry["preannotation_sha256"],
            "model_sha256": {
                "sam3": entry["sam3_model_sha256"],
                "vlm": entry["vlm_config_sha256"],
            },
            "config_sha256": entry["pipeline_config_sha256"],
            "output_candidate_path": entry["output_candidate_path"],
            "output_write_mode": "append_only_explicit_new_version",
            "candidate_only": True,
            "truth_eligible": False,
            "automatic_truth_promotion": False,
            "stages": _stage_records(admitted),
            "review_routes": [
                {"reason_code": reason, "priority": priority,
                 "target_state": "human_review_required"}
                for reason, priority in REVIEW_ROUTES.items()
            ],
        })

    resource_body = {
        key: value for key, value in resource_readback.items()
        if key != "readback_content_sha256"
    }
    body = {
        "schema_version": "labelagent.batch-execution-plan.v1",
        "plan_version": plan_version,
        "generated_at": generated_at,
        "execution_mode": "static_cpu_plan_only",
        "plan_state": "planned_not_executed" if admitted else "rejected_resource_gate",
        "resource_admission": {
            "allowed": admitted,
            "blockers": list(resource_readback["admission"]["blockers"]),
            "readback_content_sha256": canonical_sha256(resource_body),
        },
        "input_manifest": {
            "manifest_version": manifest["manifest_version"],
            "manifest_content_sha256": manifest["manifest_content_sha256"],
            "split": "train72",
            "entry_count": manifest["entry_count"],
        },
        "pipeline": list(PIPELINE),
        "record_count": len(records),
        "records": records,
        "output_policy": {
            "layer": "machine_candidate",
            "write_mode": "append_only_explicit_new_version",
            "candidate_only": True,
            "truth_eligible": False,
            "automatic_truth_promotion": False,
            "overwrite_existing_candidate": False,
            "overwrite_human_truth": False,
        },
        "execution_counters": {
            "gpu_calls": 0,
            "inference_service_calls": 0,
            "candidate_artifacts_written": 0,
            "truth_artifacts_written": 0,
            "processes_created": 0,
        },
    }
    return {**body, "plan_content_sha256": canonical_sha256(body)}


def build_rejection_receipt(plan, manifest_path, resource_readback_path):
    if plan["resource_admission"]["allowed"]:
        raise ValueError("rejection receipt requires a rejected plan")
    body = {
        "schema_version": "labelagent.batch-admission-receipt.v1",
        "generated_at": plan["generated_at"],
        "status": "rejected_fail_closed",
        "plan_version": plan["plan_version"],
        "plan_content_sha256": plan["plan_content_sha256"],
        "input_manifest": {
            "path": str(manifest_path).replace("\\", "/"),
            "sha256": file_sha256(manifest_path),
            "split": "train72",
            "entry_count": plan["input_manifest"]["entry_count"],
        },
        "resource_readback": {
            "path": str(resource_readback_path).replace("\\", "/"),
            "sha256": file_sha256(resource_readback_path),
            "content_sha256": plan["resource_admission"]["readback_content_sha256"],
        },
        "blockers": list(plan["resource_admission"]["blockers"]),
        "execution": {
            "plan_generated": True,
            "gpu_calls": 0,
            "inference_service_calls": 0,
            "candidate_artifacts_written": 0,
            "truth_artifacts_written": 0,
            "processes_created": 0,
            "remote_mutation_performed": False,
        },
        "candidate_only": True,
        "truth_promotion_allowed": False,
        "next_action": "wait_for_resource_gate_then_explicit_execution",
    }
    return {**body, "receipt_content_sha256": canonical_sha256(body)}


def build_static_review_queue_fixture(plan, manifest, queue_version):
    """Build clearly synthetic routing examples; never materialize candidates."""
    if not VERSION.fullmatch(queue_version):
        raise ValueError("queue_version is not a safe append-only namespace")
    if plan.get("execution_mode") != "static_cpu_plan_only":
        raise ValueError("static fixture requires a CPU-only plan")
    if plan.get("execution_counters", {}).get("candidate_artifacts_written") != 0:
        raise ValueError("static fixture cannot follow materialized candidates")
    validate_candidate_manifest(manifest)
    record = plan["records"][0]
    source = manifest["entries"][0]
    entries = []
    for reason, priority in REVIEW_ROUTES.items():
        candidate_sha = canonical_sha256({
            "fixture": True,
            "record_id": record["record_id"],
            "reason_code": reason,
        })
        queue_identity = {
            "queue_version": queue_version,
            "idempotency_key": record["idempotency_key"],
            "reason_code": reason,
            "candidate_sha256": candidate_sha,
        }
        entries.append({
            "queue_item_id": canonical_sha256(queue_identity),
            "idempotency_key": canonical_sha256({
                **queue_identity, "operation": "enqueue_human_review"
            }),
            "state": "human_review_required",
            "reason_code": reason,
            "priority": priority,
            "source_stage": REVIEW_SOURCE_STAGES[reason],
            "candidate_ref": {
                "artifact_id": canonical_sha256({
                    "candidate_sha256": candidate_sha,
                    "kind": "synthetic_contract_fixture",
                }),
                "path": record["output_candidate_path"],
                "sha256": candidate_sha,
                "artifact_state": "contract_fixture_not_materialized",
            },
            "provenance": {
                "dataset_id": manifest["dataset_id"],
                "source_batch": manifest["source_batch"],
                "split": "train72",
                "sequence": source["sequence"],
                "frame_id": source["frame_id"],
                "camera": source["camera"],
                "source_image_sha256": source["image_sha256"],
                "preannotation_sha256": source["preannotation_sha256"],
                "sam3_model_sha256": source["sam3_model_sha256"],
                "vlm_config_sha256": source["vlm_config_sha256"],
                "pipeline_config_sha256": source["pipeline_config_sha256"],
                "execution_plan_sha256": plan["plan_content_sha256"],
            },
            "review_status": "pending",
            "candidate_only": True,
            "truth_eligible": False,
            "automatic_truth_promotion": False,
            "synthetic_test_only": True,
            "created_at": plan["generated_at"],
        })
    body = {
        "schema_version": "labelagent.human-review-queue.v1",
        "queue_version": queue_version,
        "generated_at": plan["generated_at"],
        "sample_kind": "static_contract_fixture",
        "write_policy": {
            "mode": "append_only_explicit_new_version",
            "overwrite_existing_queue_item": False,
            "overwrite_human_truth": False,
        },
        "entry_count": len(entries),
        "entries": entries,
        "truth_promotion_allowed": False,
    }
    return {**body, "queue_content_sha256": canonical_sha256(body)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--resource-readback", type=Path, required=True)
    parser.add_argument("--plan-version", required=True)
    parser.add_argument("--generated-at", required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    readback = json.loads(args.resource_readback.read_text(encoding="utf-8"))
    plan = build_execution_plan(
        manifest, readback, args.plan_version, args.generated_at
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
