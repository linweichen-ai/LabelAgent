"""CPU-only contracts for offline CVAT review export and return.

This module deliberately contains no CVAT client, network, GPU, OSS, or process
launch code.  A machine candidate can only be exported as a preannotation.
CVAT return metadata is fail-closed and, for train72 only, delegates to the
existing immutable truth-promotion contract.  test24 can create an immutable
evaluation truth version but can never return data to training.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re

from .factory_data_contract import (
    digest,
    evaluate_truth_promotion,
    validate_ingress_event,
    validate_label_registry,
    validate_qc_event,
    validate_truth_promotion_event,
)


HASH = re.compile(r"^[0-9a-f]{64}$")
DATASET_SPLITS = {"train72": "train", "test24": "test"}
PRIORITIES = {"critical", "high", "normal", "low"}
REASON_CODES = {
    "UNKNOWN_LABEL", "NEW_LABEL_CANDIDATE", "LOW_CONFIDENCE",
    "VLM_SAM_CONFLICT", "QC_FAILURE", "HUMAN_GATE_REQUIRED",
}
DECISIONS = {"accepted", "corrected"}
REVIEW_EVENT_TYPES = {
    "cvat.review.preannotation_exported",
    "cvat.review.return.accepted",
    "cvat.review.return.rejected",
    "cvat.evaluation_truth.versioned",
}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _hash(value):
    return hashlib.sha256(value if isinstance(value, bytes)
                          else canonical_json(value)).hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise ValueError(f"{name} must be non-empty text")
    return value


def _sha(value, name):
    if not isinstance(value, str) or not HASH.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _time(value, name):
    _text(value, name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include timezone")


def build_export_contract(payload, registry, *, occurred_at):
    """Build a static, append-only CVAT preannotation export event."""
    validate_label_registry(registry)
    dataset_role = payload["dataset_role"]
    if dataset_role not in DATASET_SPLITS:
        raise ValueError("only train72 or test24 may be exported")
    if payload.get("split") != DATASET_SPLITS[dataset_role]:
        raise ValueError("dataset role and split do not match")
    routing = payload["routing"]
    reasons = routing.get("reason_codes")
    if not isinstance(reasons, list) or not reasons or any(
            value not in REASON_CODES for value in reasons):
        raise ValueError("review reason codes are missing or invalid")
    labels = [item["name"] for item in registry["labels"]]
    body = {
        "schema_version": "labelagent.cvat-offline-export.v1",
        "event_type": "cvat.review.preannotation_exported",
        "occurred_at": occurred_at,
        "dataset_role": dataset_role,
        "split": payload["split"],
        "identity": payload["identity"],
        "cvat_binding": payload["cvat_binding"],
        "artifacts": payload["artifacts"],
        "label_registry": {
            "registry_version": registry["registry_version"],
            "registry_content_sha256": registry["registry_content_sha256"],
            "allowed_labels": labels,
            "floor_slope_distinct": True,
            "downstream_mapping": registry["downstream_mapping"],
        },
        "routing": routing,
        "annotation_layer": "machine_preannotation",
        "review_state": "human_review_required",
        "completion": {
            "human_completed": False,
            "completed_by": None,
            "completion_claim_allowed": False,
        },
        "write_policy": {
            "mode": "append_only_new_version",
            "overwrite_candidate": False,
            "overwrite_truth": False,
            "truth_promotion_allowed": False,
            "network_or_upload_performed": False,
        },
    }
    body["idempotency_key"] = digest({
        "dataset_role": dataset_role,
        "input_id": body["identity"]["input_id"],
        "candidate_sha256": body["artifacts"]["candidate_sha256"],
        "package_version": body["cvat_binding"]["package_version"],
    })
    body["event_id"] = _hash(body)
    validate_export_contract(body)
    return body


def validate_export_contract(event):
    required = {
        "schema_version", "event_type", "occurred_at", "dataset_role",
        "split", "identity", "cvat_binding", "artifacts",
        "label_registry", "routing", "annotation_layer", "review_state",
        "completion", "write_policy", "idempotency_key", "event_id",
    }
    if not isinstance(event, dict) or set(event) != required:
        raise ValueError("CVAT export fields are incomplete or unknown")
    if event["schema_version"] != "labelagent.cvat-offline-export.v1" or event["event_type"] != "cvat.review.preannotation_exported":
        raise ValueError("invalid CVAT export event")
    _time(event["occurred_at"], "occurred_at")
    if event.get("dataset_role") not in DATASET_SPLITS or event.get("split") != DATASET_SPLITS[event["dataset_role"]]:
        raise ValueError("CVAT export split isolation failed")
    identity = event["identity"]
    if set(identity) != {"dataset_id", "source_batch", "sequence", "frame_id", "camera", "input_id"}:
        raise ValueError("CVAT export identity is incomplete")
    for field in ("dataset_id", "source_batch", "sequence", "frame_id", "camera"):
        _text(identity[field], field)
    _sha(identity["input_id"], "input_id")
    binding = event["cvat_binding"]
    if set(binding) != {"task_id", "job_id", "task_version", "job_version", "package_version"}:
        raise ValueError("CVAT binding is incomplete")
    for field in binding:
        _text(binding[field], field)
    artifacts = event["artifacts"]
    if set(artifacts) != {
        "image_uri", "image_sha256", "input_sha256", "candidate_uri",
        "candidate_sha256", "model_sha256", "config_sha256",
        "preannotation_uri", "preannotation_sha256",
    }:
        raise ValueError("CVAT export artifact bindings are incomplete")
    for field in ("image_uri", "candidate_uri", "preannotation_uri"):
        _text(artifacts[field], field)
    for field in artifacts:
        if field.endswith("sha256"):
            _sha(artifacts[field], field)
    registry = event["label_registry"]
    if set(registry) != {"registry_version", "registry_content_sha256", "allowed_labels", "floor_slope_distinct", "downstream_mapping"}:
        raise ValueError("CVAT export registry binding is incomplete")
    _text(registry["registry_version"], "registry_version")
    _sha(registry["registry_content_sha256"], "registry_content_sha256")
    if registry["floor_slope_distinct"] is not True or not {"floor", "slope"}.issubset(set(registry["allowed_labels"])):
        raise ValueError("floor and slope must remain distinct")
    mapping = registry["downstream_mapping"]
    if mapping.get("target_label") != "traversable_area" or mapping.get("operation") != "union" or mapping.get("source_labels") != ["floor", "slope"]:
        raise ValueError("downstream traversable union is invalid")
    routing = event["routing"]
    if set(routing) != {"reason_codes", "priority", "provenance", "unknown_or_new_label_pending"}:
        raise ValueError("CVAT routing fields are incomplete")
    if not routing["reason_codes"] or any(value not in REASON_CODES for value in routing["reason_codes"]):
        raise ValueError("CVAT routing reason code is invalid")
    if routing["priority"] not in PRIORITIES:
        raise ValueError("CVAT routing priority is invalid")
    if routing["provenance"] not in {"real", "simulation"}:
        raise ValueError("CVAT routing provenance is invalid")
    pending_expected = any(value in {"UNKNOWN_LABEL", "NEW_LABEL_CANDIDATE"} for value in routing["reason_codes"])
    if routing["unknown_or_new_label_pending"] is not pending_expected:
        raise ValueError("unknown/new-label state must remain pending")
    if event["annotation_layer"] != "machine_preannotation" or event["review_state"] != "human_review_required":
        raise ValueError("CVAT export cannot claim human truth")
    if event["completion"] != {"human_completed": False, "completed_by": None, "completion_claim_allowed": False}:
        raise ValueError("preannotation cannot be marked human-completed")
    if event["write_policy"] != {
        "mode": "append_only_new_version", "overwrite_candidate": False,
        "overwrite_truth": False, "truth_promotion_allowed": False,
        "network_or_upload_performed": False,
    }:
        raise ValueError("CVAT export write policy is unsafe")
    for field in ("idempotency_key", "event_id"):
        _sha(event[field], field)
    expected_key = digest({
        "dataset_role": event["dataset_role"],
        "input_id": identity["input_id"],
        "candidate_sha256": artifacts["candidate_sha256"],
        "package_version": binding["package_version"],
    })
    if event["idempotency_key"] != expected_key:
        raise ValueError("CVAT export idempotency key mismatch")
    if event["event_id"] != _hash({key: value for key, value in event.items() if key != "event_id"}):
        raise ValueError("CVAT export event hash mismatch")


def _return_blockers(export, payload, registry):
    blockers = []
    try:
        validate_export_contract(export)
        validate_label_registry(registry)
    except ValueError:
        return ["EXPORT_OR_REGISTRY_INVALID"]
    if payload.get("export_event_id") != export["event_id"]:
        blockers.append("EXPORT_EVENT_BINDING_MISMATCH")
    if payload.get("dataset_role") != export["dataset_role"] or payload.get("split") != export["split"]:
        blockers.append("SPLIT_MIX_OR_DATASET_ROLE_MISMATCH")
    if payload.get("identity") != export["identity"]:
        blockers.append("SOURCE_IDENTITY_MISMATCH")
    if payload.get("cvat_binding") != export["cvat_binding"]:
        blockers.append("CVAT_TASK_JOB_VERSION_DRIFT")
    completion = payload.get("completion", {})
    if completion.get("status") != "human_completed" or completion.get("actor_type") != "human":
        blockers.append("MACHINE_OR_AUTOMATIC_COMPLETION_FORBIDDEN")
    reviewer = completion.get("reviewer", {})
    if reviewer.get("identity_assurance") != "authenticated":
        blockers.append("AUTHENTICATED_REVIEWER_REQUIRED")
    try:
        _text(reviewer.get("reviewer_id"), "reviewer_id")
        _time(reviewer.get("reviewed_at"), "reviewed_at")
    except ValueError:
        blockers.append("REVIEWER_OR_REVIEW_TIME_MISSING")
    if completion.get("decision") not in DECISIONS:
        blockers.append("HUMAN_DECISION_INVALID")
    bindings = payload.get("artifact_bindings", {})
    artifacts = export["artifacts"]
    expected = {
        "source_image_sha256": artifacts["image_sha256"],
        "input_sha256": artifacts["input_sha256"],
        "candidate_sha256": artifacts["candidate_sha256"],
        "model_sha256": artifacts["model_sha256"],
        "config_sha256": artifacts["config_sha256"],
        "label_registry_sha256": registry["registry_content_sha256"],
    }
    if bindings != expected:
        blockers.append("ARTIFACT_HASH_BINDING_MISMATCH")
    annotation = payload.get("annotation_export", {})
    if annotation.get("sha256") != annotation.get("verified_sha256"):
        blockers.append("ANNOTATION_EXPORT_HASH_TAMPERED")
    try:
        _text(annotation.get("uri"), "annotation_export.uri")
        _sha(annotation.get("sha256"), "annotation_export.sha256")
    except ValueError:
        blockers.append("ANNOTATION_EXPORT_EVIDENCE_MISSING")
    labels = payload.get("annotation_labels")
    allowed = set(export["label_registry"]["allowed_labels"])
    if not isinstance(labels, list) or any(value not in allowed for value in labels):
        blockers.append("LABEL_OUTSIDE_REGISTERED_ONTOLOGY")
    diff = payload.get("before_after_diff", {})
    truth = payload.get("truth", {})
    if diff.get("before_sha256") != artifacts["candidate_sha256"] or diff.get("after_sha256") != truth.get("sha256"):
        blockers.append("BEFORE_AFTER_DIFF_BINDING_MISMATCH")
    for field in ("added", "removed", "modified"):
        if isinstance(diff.get(field), bool) or not isinstance(diff.get(field), int) or diff.get(field) < 0:
            blockers.append("BEFORE_AFTER_DIFF_INVALID")
            break
    try:
        _text(diff.get("artifact_uri"), "diff.artifact_uri")
        _sha(diff.get("artifact_sha256"), "diff.artifact_sha256")
        _text(truth.get("artifact_id"), "truth.artifact_id")
        _text(truth.get("uri"), "truth.uri")
        _sha(truth.get("sha256"), "truth.sha256")
    except ValueError:
        blockers.append("TRUTH_OR_DIFF_EVIDENCE_MISSING")
    version = payload.get("version", {})
    if version.get("split") != export["split"]:
        blockers.append("TRUTH_VERSION_SPLIT_MISMATCH")
    if not all(version.get(field) is expected_value for field, expected_value in (
        ("immutable", True), ("append_only", True),
        ("overwrite_previous_truth", False), ("previous_truth_preserved", True),
    )):
        blockers.append("TRUTH_VERSION_NOT_APPEND_ONLY")
    if version.get("truth_sha256") != truth.get("sha256"):
        blockers.append("TRUTH_VERSION_HASH_MISMATCH")
    for field in ("version_id", "parent_version", "manifest_uri"):
        try:
            _text(version.get(field), field)
        except ValueError:
            blockers.append("TRUTH_VERSION_EVIDENCE_MISSING")
            break
    try:
        _sha(version.get("manifest_sha256"), "manifest_sha256")
    except ValueError:
        blockers.append("TRUTH_VERSION_EVIDENCE_MISSING")
    approval = payload.get("approval", {})
    if approval.get("decision") != "approved" or approval.get("reviewer_id") != reviewer.get("reviewer_id"):
        blockers.append("APPROVAL_RECEIPT_MISSING_OR_UNBOUND")
    if approval.get("annotation_export_sha256") != annotation.get("sha256") or approval.get("truth_sha256") != truth.get("sha256"):
        blockers.append("APPROVAL_ARTIFACT_BINDING_MISMATCH")
    for field in ("approval_id", "receipt_uri"):
        try:
            _text(approval.get(field), field)
        except ValueError:
            blockers.append("APPROVAL_EVIDENCE_MISSING")
            break
    try:
        _sha(approval.get("receipt_sha256"), "receipt_sha256")
        _time(approval.get("approved_at"), "approved_at")
    except ValueError:
        blockers.append("APPROVAL_EVIDENCE_MISSING")
    return sorted(set(blockers))


def _rejection(export, payload, blockers, occurred_at):
    body = {
        "schema_version": "labelagent.cvat-return-rejection.v1",
        "event_type": "cvat.review.return.rejected",
        "occurred_at": occurred_at,
        "export_event_id": export.get("event_id"),
        "dataset_role": export.get("dataset_role"),
        "split": export.get("split"),
        "return_version_id": payload.get("version", {}).get("version_id"),
        "status": "rejected_fail_closed",
        "blockers": sorted(set(blockers)),
        "truth_version_created": False,
        "training_return_allowed": False,
        "next_state": "human_review_required",
    }
    body["idempotency_key"] = digest({
        "export_event_id": body["export_event_id"],
        "return_version_id": body["return_version_id"],
        "blockers": body["blockers"],
    })
    body["event_id"] = _hash(body)
    validate_review_event(body)
    return body


def evaluate_cvat_return(export, payload, registry, *, ingress=None, qc=None,
                         occurred_at):
    """Validate an offline CVAT return and route train/test truth safely."""
    blockers = _return_blockers(export, payload, registry)
    if blockers:
        return _rejection(export, payload, blockers, occurred_at)
    if export["dataset_role"] == "test24":
        body = {
            "schema_version": "labelagent.cvat-evaluation-truth.v1",
            "event_type": "cvat.evaluation_truth.versioned",
            "occurred_at": occurred_at,
            "export_event_id": export["event_id"],
            "dataset_role": "test24",
            "split": "test",
            "identity": export["identity"],
            "review": payload["completion"],
            "annotation_export": payload["annotation_export"],
            "artifact_bindings": payload["artifact_bindings"],
            "before_after_diff": payload["before_after_diff"],
            "truth": payload["truth"],
            "version": payload["version"],
            "approval": payload["approval"],
            "evaluation_truth": True,
            "training_return_allowed": False,
            "truth_eligible_for_training": False,
            "write_policy": {"mode": "append_only", "overwrite_previous_truth": False},
        }
        body["idempotency_key"] = digest({
            "export_event_id": export["event_id"],
            "version_id": payload["version"]["version_id"],
            "truth_sha256": payload["truth"]["sha256"],
        })
        body["event_id"] = _hash(body)
        validate_review_event(body)
        return body

    if ingress is None or qc is None:
        return _rejection(export, payload, ["INGRESS_OR_QC_EVIDENCE_MISSING"], occurred_at)
    try:
        validate_ingress_event(ingress)
        validate_qc_event(qc)
    except ValueError:
        return _rejection(export, payload, ["INGRESS_OR_QC_EVIDENCE_INVALID"], occurred_at)
    identity = export["identity"]
    source_image_sha = export["artifacts"]["image_sha256"]
    request = {
        "source": {
            "dataset_role": "train_candidate", "synthetic_test_only": False,
            "source_kind": "real", "dataset_id": identity["dataset_id"],
            "source_batch": identity["source_batch"], "sequence": identity["sequence"],
            "frame_id": identity["frame_id"], "camera": identity["camera"],
            "ingress_event_id": ingress["event_id"], "qc_event_id": qc["event_id"],
        },
        "candidate": {
            "artifact_id": digest({"candidate": export["artifacts"]["candidate_sha256"]}),
            "uri": export["artifacts"]["candidate_uri"],
            "sha256": export["artifacts"]["candidate_sha256"],
            "source_image_sha256": source_image_sha, "layer": "machine_candidate",
        },
        "review": {
            "review_id": payload["approval"]["approval_id"],
            "review_origin": "genuine_human_review",
            "reviewer_declared_id": payload["completion"]["reviewer"]["reviewer_id"],
            "identity_assurance": "authenticated",
            "reviewed_at": payload["completion"]["reviewer"]["reviewed_at"],
            "decision": payload["completion"]["decision"],
            "receipt_uri": payload["approval"]["receipt_uri"],
            "receipt_sha256": payload["approval"]["receipt_sha256"],
            "candidate_sha256": export["artifacts"]["candidate_sha256"],
            "source_image_sha256": source_image_sha,
        },
        "before_after_diff": payload["before_after_diff"],
        "truth": {**payload["truth"], "layer": "human_truth"},
        "version": payload["version"],
        "approval": {
            "approval_id": payload["approval"]["approval_id"], "decision": "approved",
            "receipt_uri": payload["approval"]["receipt_uri"],
            "receipt_sha256": payload["approval"]["receipt_sha256"],
            "review_id": payload["approval"]["approval_id"],
            "candidate_sha256": export["artifacts"]["candidate_sha256"],
            "source_image_sha256": source_image_sha,
            "truth_sha256": payload["truth"]["sha256"],
            "approved_at": payload["approval"]["approved_at"],
            "approver_declared_id": payload["completion"]["reviewer"]["reviewer_id"],
            "identity_assurance": "authenticated",
        },
        "direct_machine_preannotation_promotion": False,
    }
    promotion = evaluate_truth_promotion(ingress, qc, request, occurred_at=occurred_at)
    if promotion["event_type"] != "truth.version.promoted":
        return _rejection(export, payload, ["IMMUTABLE_TRUTH_PROMOTION_REJECTED"] + promotion["blockers"], occurred_at)
    validate_truth_promotion_event(promotion)
    body = {
        "schema_version": "labelagent.cvat-return-accepted.v1",
        "event_type": "cvat.review.return.accepted",
        "occurred_at": occurred_at,
        "export_event_id": export["event_id"],
        "dataset_role": "train72", "split": "train",
        "annotation_export_sha256": payload["annotation_export"]["sha256"],
        "truth_promotion_event": promotion,
        "training_return_allowed": True,
    }
    body["idempotency_key"] = digest({
        "export_event_id": export["event_id"],
        "promotion_event_id": promotion["event_id"],
    })
    body["event_id"] = _hash(body)
    validate_review_event(body)
    return body


def validate_review_event(event):
    event_type = event.get("event_type")
    if event_type == "cvat.review.preannotation_exported":
        return validate_export_contract(event)
    if event_type not in REVIEW_EVENT_TYPES:
        raise ValueError("unsupported CVAT review event type")
    required_by_type = {
        "cvat.review.return.rejected": {
            "schema_version", "event_type", "occurred_at", "export_event_id",
            "dataset_role", "split", "return_version_id", "status", "blockers",
            "truth_version_created", "training_return_allowed", "next_state",
            "idempotency_key", "event_id",
        },
        "cvat.evaluation_truth.versioned": {
            "schema_version", "event_type", "occurred_at", "export_event_id",
            "dataset_role", "split", "identity", "review", "annotation_export",
            "artifact_bindings", "before_after_diff", "truth", "version",
            "approval", "evaluation_truth", "training_return_allowed",
            "truth_eligible_for_training", "write_policy", "idempotency_key",
            "event_id",
        },
        "cvat.review.return.accepted": {
            "schema_version", "event_type", "occurred_at", "export_event_id",
            "dataset_role", "split", "annotation_export_sha256",
            "truth_promotion_event", "training_return_allowed",
            "idempotency_key", "event_id",
        },
    }
    if not isinstance(event, dict) or set(event) != required_by_type[event_type]:
        raise ValueError("CVAT review event fields are incomplete or unknown")
    _time(event.get("occurred_at"), "occurred_at")
    _sha(event.get("idempotency_key"), "idempotency_key")
    _sha(event.get("event_id"), "event_id")
    if event["event_id"] != _hash({key: value for key, value in event.items() if key != "event_id"}):
        raise ValueError("CVAT review event hash mismatch")
    if event_type == "cvat.review.return.rejected":
        if event.get("schema_version") != "labelagent.cvat-return-rejection.v1":
            raise ValueError("invalid CVAT rejection schema version")
        _sha(event.get("export_event_id"), "export_event_id")
        if event.get("status") != "rejected_fail_closed" or not event.get("blockers"):
            raise ValueError("CVAT rejection must contain blockers")
        if event.get("truth_version_created") is not False or event.get("training_return_allowed") is not False:
            raise ValueError("CVAT rejection cannot create training truth")
    elif event_type == "cvat.evaluation_truth.versioned":
        if event.get("schema_version") != "labelagent.cvat-evaluation-truth.v1":
            raise ValueError("invalid CVAT evaluation truth schema version")
        _sha(event.get("export_event_id"), "export_event_id")
        if event.get("dataset_role") != "test24" or event.get("split") != "test":
            raise ValueError("evaluation truth must remain in test24")
        if event.get("evaluation_truth") is not True or event.get("training_return_allowed") is not False or event.get("truth_eligible_for_training") is not False:
            raise ValueError("test24 truth cannot return to training")
        if event.get("write_policy") != {"mode": "append_only", "overwrite_previous_truth": False}:
            raise ValueError("evaluation truth must be append-only")
        review = event.get("review", {})
        reviewer = review.get("reviewer", {})
        if review.get("status") != "human_completed" or review.get("actor_type") != "human" or review.get("decision") not in DECISIONS:
            raise ValueError("evaluation truth requires explicit human completion")
        if reviewer.get("identity_assurance") != "authenticated":
            raise ValueError("evaluation truth reviewer must be authenticated")
        _text(reviewer.get("reviewer_id"), "reviewer_id")
        _time(reviewer.get("reviewed_at"), "reviewed_at")
        annotation = event.get("annotation_export", {})
        _sha(annotation.get("sha256"), "annotation_export.sha256")
        if annotation.get("sha256") != annotation.get("verified_sha256"):
            raise ValueError("evaluation annotation export hash mismatch")
        for value in event.get("artifact_bindings", {}).values():
            _sha(value, "artifact binding")
        diff, truth, version = event.get("before_after_diff", {}), event.get("truth", {}), event.get("version", {})
        _sha(truth.get("sha256"), "truth.sha256")
        if diff.get("after_sha256") != truth.get("sha256") or version.get("truth_sha256") != truth.get("sha256"):
            raise ValueError("evaluation truth/diff/version hash binding mismatch")
        if version.get("split") != "test" or not all(version.get(field) is expected for field, expected in (
            ("immutable", True), ("append_only", True),
            ("overwrite_previous_truth", False), ("previous_truth_preserved", True),
        )):
            raise ValueError("evaluation version is not immutable test truth")
    elif event_type == "cvat.review.return.accepted":
        if event.get("schema_version") != "labelagent.cvat-return-accepted.v1":
            raise ValueError("invalid CVAT accepted return schema version")
        if event.get("dataset_role") != "train72" or event.get("split") != "train" or event.get("training_return_allowed") is not True:
            raise ValueError("only train72 may return to training")
        _sha(event.get("export_event_id"), "export_event_id")
        _sha(event.get("annotation_export_sha256"), "annotation_export_sha256")
        validate_truth_promotion_event(event.get("truth_promotion_event"))


def _record(event, sequence, previous):
    record = {
        "sequence": sequence,
        "previous_record_sha256": previous,
        "event_type": event["event_type"],
        "idempotency_key": event["idempotency_key"],
        "event_sha256": digest(event),
        "event": event,
    }
    record["record_sha256"] = digest(record)
    return record


def build_review_ledger_snapshot(events, *, ledger_id, generated_at,
                                 sample_kind):
    records, previous = [], None
    versions = set()
    for index, event in enumerate(events, 1):
        validate_review_event(event)
        version = event.get("version", {}).get("version_id")
        if version is None and event["event_type"] == "cvat.review.return.accepted":
            version = event["truth_promotion_event"]["version"]["version_id"]
        if version and version in versions:
            raise ValueError("CVAT truth/evaluation version IDs must be unique")
        if version:
            versions.add(version)
        record = _record(event, index, previous)
        records.append(record)
        previous = record["record_sha256"]
    body = {
        "schema_version": "labelagent.cvat-review-ledger.v1",
        "ledger_id": ledger_id, "generated_at": generated_at,
        "sample_kind": sample_kind, "append_only": True,
        "record_count": len(records), "records": records,
        "head_record_sha256": previous,
    }
    body["snapshot_sha256"] = digest(body)
    validate_review_ledger_snapshot(body)
    return body


def validate_review_ledger_snapshot(snapshot):
    if snapshot.get("schema_version") != "labelagent.cvat-review-ledger.v1" or snapshot.get("append_only") is not True:
        raise ValueError("invalid CVAT review ledger")
    _time(snapshot.get("generated_at"), "generated_at")
    records = snapshot.get("records")
    if not isinstance(records, list) or snapshot.get("record_count") != len(records):
        raise ValueError("CVAT ledger record count mismatch")
    previous = None
    versions = set()
    for index, record in enumerate(records, 1):
        if record.get("sequence") != index or record.get("previous_record_sha256") != previous:
            raise ValueError("CVAT ledger hash chain mismatch")
        event = record.get("event")
        validate_review_event(event)
        if record.get("event_sha256") != digest(event):
            raise ValueError("CVAT ledger event hash mismatch")
        if record.get("record_sha256") != digest({key: value for key, value in record.items() if key != "record_sha256"}):
            raise ValueError("CVAT ledger record hash mismatch")
        version = event.get("version", {}).get("version_id")
        if version is None and event["event_type"] == "cvat.review.return.accepted":
            version = event["truth_promotion_event"]["version"]["version_id"]
        if version and version in versions:
            raise ValueError("duplicate CVAT truth/evaluation version")
        if version:
            versions.add(version)
        previous = record["record_sha256"]
    if snapshot.get("head_record_sha256") != previous:
        raise ValueError("CVAT ledger head mismatch")
    if snapshot.get("snapshot_sha256") != digest({key: value for key, value in snapshot.items() if key != "snapshot_sha256"}):
        raise ValueError("CVAT ledger snapshot hash mismatch")


class AppendOnlyCvatReviewLedger:
    """Exclusive-create local review ledger; no CVAT/network operations."""

    def __init__(self, root, ledger_id):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger_id = _text(ledger_id, "ledger_id")

    def records(self):
        result = []
        for path in sorted(self.root.glob("[0-9][0-9][0-9][0-9][0-9][0-9]-*.json")):
            result.append(json.loads(path.read_text(encoding="utf-8")))
        snapshot = build_review_ledger_snapshot(
            [item["event"] for item in result], ledger_id=self.ledger_id,
            generated_at="1970-01-01T00:00:00Z", sample_kind="runtime_readback")
        if result != snapshot["records"]:
            raise ValueError("stored CVAT ledger differs from canonical chain")
        return result

    def append(self, event):
        validate_review_event(event)
        records = self.records()
        event_sha = digest(event)
        for record in records:
            if record["idempotency_key"] == event["idempotency_key"]:
                if record["event_sha256"] != event_sha:
                    raise ValueError("CVAT idempotency replay changed immutable event")
                return record, True
        version = event.get("version", {}).get("version_id")
        if version is None and event["event_type"] == "cvat.review.return.accepted":
            version = event["truth_promotion_event"]["version"]["version_id"]
        if version:
            for record in records:
                old = record["event"].get("version", {}).get("version_id")
                if old is None and record["event"]["event_type"] == "cvat.review.return.accepted":
                    old = record["event"]["truth_promotion_event"]["version"]["version_id"]
                if old == version:
                    raise ValueError("CVAT truth/evaluation version IDs are immutable and unique")
        previous = records[-1]["record_sha256"] if records else None
        record = _record(event, len(records) + 1, previous)
        path = self.root / f"{record['sequence']:06d}-{record['record_sha256']}.json"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
        except Exception:
            path.unlink(missing_ok=True)
            raise
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError("CVAT ledger readback mismatch")
        return record, False
