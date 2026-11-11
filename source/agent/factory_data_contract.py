"""Append-only contracts for factory ingress, QC, and human truth promotion.

The module has no network, OSS, CUDA, model-service, or process-launch code.
It validates supplied metadata and can append immutable JSON records to a local
hash chain.  Failed QC is always represented by an explicit quarantine event;
truth promotion is allowed only after a passed QC event and a genuine human
review with exact artifact bindings.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re

from .downstream_mapping import TRAVERSABLE_AREA_MAPPING


HASH = re.compile(r"^[0-9a-f]{64}$")
CAMERAS = ("camera_0", "camera_1", "camera_2", "camera_3")
QC_CHECKS = (
    "file_integrity",
    "four_camera_completeness",
    "camera_frame_alignment",
    "timestamp_alignment",
    "calibration_presence",
    "calibration_version_consistency",
    "pose_odom_continuity",
)
QC_REASON_CODES = {
    "file_integrity": "FILE_CORRUPT",
    "four_camera_completeness": "FOUR_CAMERA_INCOMPLETE",
    "camera_frame_alignment": "CAMERA_FRAME_MISALIGNED",
    "timestamp_alignment": "CAMERA_TIMESTAMP_DELTA_EXCEEDED",
    "calibration_presence": "CALIBRATION_MISSING",
    "calibration_version_consistency": "CALIBRATION_VERSION_MISMATCH",
    "pose_odom_continuity": "POSE_ODOM_DISCONTINUITY",
}
EVENT_TYPES = {
    "factory.frame_bundle.ingested",
    "factory.frame_bundle.qc",
    "truth.version.promoted",
    "truth.promotion_rejected",
}


def canonical_json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(
        value if isinstance(value, bytes) else canonical_json(value)
    ).hexdigest()


def _require_text(value, name, limit=2000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_hash(value, name):
    if not isinstance(value, str) or not HASH.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_time(value, name):
    _require_text(value, name)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be ISO-8601") from error
    if result.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return result


def _artifact(value, name, *, nullable=False):
    if value is None and nullable:
        return
    if not isinstance(value, dict) or set(value) != {
        "artifact_id", "uri", "sha256", "bytes", "integrity"
    }:
        raise ValueError(f"{name} artifact fields are incomplete")
    _require_hash(value["artifact_id"], f"{name}.artifact_id")
    _require_text(value["uri"], f"{name}.uri")
    _require_hash(value["sha256"], f"{name}.sha256")
    if isinstance(value["bytes"], bool) or not isinstance(value["bytes"], int) or value["bytes"] < 0:
        raise ValueError(f"{name}.bytes must be non-negative")
    if value["integrity"] not in {"verified", "corrupt"}:
        raise ValueError(f"{name}.integrity is invalid")


def _event_hash(body):
    return digest(body)


def build_ingress_event(payload, *, occurred_at):
    """Normalize and hash an observed frame bundle; missing inputs remain visible."""
    body = {
        "schema_version": "labelagent.factory-ingress-event.v1",
        "event_type": "factory.frame_bundle.ingested",
        "occurred_at": occurred_at,
        "identity": payload["identity"],
        "provenance": payload["provenance"],
        "expected_cameras": list(CAMERAS),
        "cameras": payload["cameras"],
        "calibration": payload["calibration"],
        "motion": payload["motion"],
        "write_policy": {
            "mode": "append_only",
            "overwrite_source": False,
            "remote_mutation": False,
            "silent_skip": False,
        },
    }
    identity = body["identity"]
    sources = {
        "camera_sha256": [item["artifact"]["sha256"] for item in body["cameras"]],
        "calibration_sha256": (
            body["calibration"]["artifact"]["sha256"]
            if body["calibration"].get("artifact") else None
        ),
        "pose_sha256": (
            body["motion"]["pose"]["artifact"]["sha256"]
            if body["motion"]["pose"].get("artifact") else None
        ),
        "odom_sha256": (
            body["motion"]["odom"]["artifact"]["sha256"]
            if body["motion"]["odom"].get("artifact") else None
        ),
    }
    body["input_id"] = digest({
        "dataset_id": identity["dataset_id"],
        "source_batch": identity["source_batch"],
        "sequence": identity["sequence"],
        "frame_id": identity["frame_id"],
        "capture_time": identity["capture_time"],
        "source_kind": body["provenance"]["source_kind"],
    })
    body["idempotency_key"] = digest({
        "input_id": body["input_id"],
        "sources": sources,
        "expected_calibration_version": body["calibration"]["expected_version"],
    })
    body["event_id"] = _event_hash(body)
    validate_ingress_event(body)
    return body


def validate_ingress_event(event):
    required = {
        "schema_version", "event_type", "occurred_at", "identity", "provenance",
        "expected_cameras", "cameras", "calibration", "motion", "write_policy",
        "input_id", "idempotency_key", "event_id",
    }
    if not isinstance(event, dict) or set(event) != required:
        raise ValueError("ingress event fields are incomplete or unknown")
    if event["schema_version"] != "labelagent.factory-ingress-event.v1" or event["event_type"] != "factory.frame_bundle.ingested":
        raise ValueError("invalid ingress event type or schema")
    _require_time(event["occurred_at"], "occurred_at")
    identity = event["identity"]
    if not isinstance(identity, dict) or set(identity) != {
        "dataset_id", "source_batch", "sequence", "frame_id", "capture_time"
    }:
        raise ValueError("ingress identity fields are incomplete")
    for field in ("dataset_id", "source_batch", "sequence", "frame_id"):
        _require_text(identity[field], f"identity.{field}")
    _require_time(identity["capture_time"], "identity.capture_time")

    provenance = event["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {
        "source_kind", "simulation"
    }:
        raise ValueError("provenance fields are incomplete")
    if provenance["source_kind"] not in {"real", "sim"}:
        raise ValueError("source_kind must be real or sim")
    simulation = provenance["simulation"]
    if provenance["source_kind"] == "real":
        if simulation is not None:
            raise ValueError("real provenance cannot contain simulation metadata")
    else:
        if not isinstance(simulation, dict) or set(simulation) != {
            "simulator", "simulator_version", "scene", "random_seed"
        }:
            raise ValueError("sim provenance requires complete simulation metadata")
        for field in ("simulator", "simulator_version", "scene"):
            _require_text(simulation[field], f"simulation.{field}")
        if isinstance(simulation["random_seed"], bool) or not isinstance(simulation["random_seed"], int):
            raise ValueError("simulation.random_seed must be an integer")

    if event["expected_cameras"] != list(CAMERAS):
        raise ValueError("expected_cameras must be the four canonical cameras")
    cameras = event["cameras"]
    if not isinstance(cameras, list) or len(cameras) > 4:
        raise ValueError("cameras must be an observed array of at most four items")
    seen = set()
    for index, camera in enumerate(cameras):
        if not isinstance(camera, dict) or set(camera) != {
            "camera", "frame_id", "captured_at", "artifact"
        }:
            raise ValueError(f"cameras[{index}] fields are incomplete")
        if camera["camera"] not in CAMERAS or camera["camera"] in seen:
            raise ValueError("camera identifiers must be canonical and unique")
        seen.add(camera["camera"])
        _require_text(camera["frame_id"], f"cameras[{index}].frame_id")
        _require_time(camera["captured_at"], f"cameras[{index}].captured_at")
        _artifact(camera["artifact"], f"cameras[{index}]")

    calibration = event["calibration"]
    if not isinstance(calibration, dict) or set(calibration) != {
        "present", "expected_version", "observed_version", "camera_ids", "artifact"
    }:
        raise ValueError("calibration fields are incomplete")
    if not isinstance(calibration["present"], bool):
        raise ValueError("calibration.present must be boolean")
    _require_text(calibration["expected_version"], "calibration.expected_version")
    if calibration["present"]:
        _require_text(calibration["observed_version"], "calibration.observed_version")
        _artifact(calibration["artifact"], "calibration")
    elif calibration["observed_version"] is not None or calibration["artifact"] is not None:
        raise ValueError("missing calibration cannot carry observed artifact metadata")
    if not isinstance(calibration["camera_ids"], list) or len(calibration["camera_ids"]) != len(set(calibration["camera_ids"])) or not set(calibration["camera_ids"]) <= set(CAMERAS):
        raise ValueError("calibration.camera_ids must be unique canonical cameras")

    motion = event["motion"]
    if not isinstance(motion, dict) or set(motion) != {"pose", "odom"}:
        raise ValueError("motion must contain pose and odom")
    for name in ("pose", "odom"):
        stream = motion[name]
        if not isinstance(stream, dict) or set(stream) != {
            "present", "artifact", "max_gap_ms", "discontinuity_count"
        }:
            raise ValueError(f"motion.{name} fields are incomplete")
        if not isinstance(stream["present"], bool):
            raise ValueError(f"motion.{name}.present must be boolean")
        if stream["present"]:
            _artifact(stream["artifact"], f"motion.{name}")
        elif stream["artifact"] is not None:
            raise ValueError(f"missing motion.{name} cannot carry an artifact")
        if isinstance(stream["max_gap_ms"], bool) or not isinstance(stream["max_gap_ms"], (int, float)) or stream["max_gap_ms"] < 0:
            raise ValueError(f"motion.{name}.max_gap_ms must be non-negative")
        if isinstance(stream["discontinuity_count"], bool) or not isinstance(stream["discontinuity_count"], int) or stream["discontinuity_count"] < 0:
            raise ValueError(f"motion.{name}.discontinuity_count must be non-negative")

    if event["write_policy"] != {
        "mode": "append_only", "overwrite_source": False,
        "remote_mutation": False, "silent_skip": False,
    }:
        raise ValueError("ingress must be append-only and read-only to source")
    for field in ("input_id", "idempotency_key", "event_id"):
        _require_hash(event[field], field)
    expected_input_id = digest({
        "dataset_id": identity["dataset_id"],
        "source_batch": identity["source_batch"],
        "sequence": identity["sequence"],
        "frame_id": identity["frame_id"],
        "capture_time": identity["capture_time"],
        "source_kind": provenance["source_kind"],
    })
    if event["input_id"] != expected_input_id:
        raise ValueError("ingress input_id mismatch")
    sources = {
        "camera_sha256": [item["artifact"]["sha256"] for item in cameras],
        "calibration_sha256": (
            calibration["artifact"]["sha256"]
            if calibration["artifact"] else None
        ),
        "pose_sha256": (
            motion["pose"]["artifact"]["sha256"]
            if motion["pose"]["artifact"] else None
        ),
        "odom_sha256": (
            motion["odom"]["artifact"]["sha256"]
            if motion["odom"]["artifact"] else None
        ),
    }
    expected_idempotency = digest({
        "input_id": event["input_id"], "sources": sources,
        "expected_calibration_version": calibration["expected_version"],
    })
    if event["idempotency_key"] != expected_idempotency:
        raise ValueError("ingress idempotency_key mismatch")
    expected_event_id = _event_hash({
        key: value for key, value in event.items() if key != "event_id"
    })
    if event["event_id"] != expected_event_id:
        raise ValueError("ingress event_id mismatch")


def _check(check_id, passed, observed, artifact_ids):
    return {
        "check_id": check_id,
        "status": "pass" if passed else "fail",
        "reason_code": None if passed else QC_REASON_CODES[check_id],
        "observed": observed,
        "evidence_artifact_ids": sorted(set(artifact_ids)),
    }


def evaluate_qc(ingress, *, occurred_at, policy):
    validate_ingress_event(ingress)
    if set(policy) != {
        "policy_version", "policy_sha256", "max_camera_timestamp_delta_ms",
        "max_pose_gap_ms", "max_odom_gap_ms",
    }:
        raise ValueError("QC policy fields are incomplete")
    _require_text(policy["policy_version"], "policy.policy_version")
    _require_hash(policy["policy_sha256"], "policy.policy_sha256")
    for field in ("max_camera_timestamp_delta_ms", "max_pose_gap_ms", "max_odom_gap_ms"):
        if isinstance(policy[field], bool) or not isinstance(policy[field], (int, float)) or policy[field] < 0:
            raise ValueError(f"policy.{field} must be non-negative")

    cameras = ingress["cameras"]
    calibration = ingress["calibration"]
    motion = ingress["motion"]
    artifacts = [item["artifact"] for item in cameras]
    if calibration["artifact"] is not None:
        artifacts.append(calibration["artifact"])
    for stream in motion.values():
        if stream["artifact"] is not None:
            artifacts.append(stream["artifact"])
    all_ids = [item["artifact_id"] for item in artifacts]
    corrupt = [item["artifact_id"] for item in artifacts
               if item["integrity"] != "verified"]

    actual_cameras = {item["camera"] for item in cameras}
    expected_cameras = set(CAMERAS)
    mismatched_frames = [item["camera"] for item in cameras
                         if item["frame_id"] != ingress["identity"]["frame_id"]]
    root_time = _require_time(ingress["identity"]["capture_time"], "capture_time")
    timestamp_deltas = [
        abs((_require_time(item["captured_at"], "camera captured_at") - root_time)
            .total_seconds() * 1000)
        for item in cameras
    ]
    max_delta = max(timestamp_deltas, default=float("inf"))
    calibration_present = calibration["present"] and calibration["artifact"] is not None
    calibration_consistent = (
        calibration_present
        and calibration["observed_version"] == calibration["expected_version"]
        and set(calibration["camera_ids"]) == expected_cameras
    )
    pose = motion["pose"]
    odom = motion["odom"]
    motion_ok = (
        pose["present"] and odom["present"]
        and pose["artifact"] is not None and odom["artifact"] is not None
        and pose["artifact"]["integrity"] == "verified"
        and odom["artifact"]["integrity"] == "verified"
        and pose["max_gap_ms"] <= policy["max_pose_gap_ms"]
        and odom["max_gap_ms"] <= policy["max_odom_gap_ms"]
        and pose["discontinuity_count"] == 0
        and odom["discontinuity_count"] == 0
    )
    checks = [
        _check("file_integrity", not corrupt,
               "corrupt=" + ",".join(corrupt), all_ids),
        _check("four_camera_completeness", actual_cameras == expected_cameras,
               "present=" + ",".join(sorted(actual_cameras)),
               [item["artifact"]["artifact_id"] for item in cameras]),
        _check("camera_frame_alignment", not mismatched_frames,
               "mismatched=" + ",".join(sorted(mismatched_frames)),
               [item["artifact"]["artifact_id"] for item in cameras]),
        _check("timestamp_alignment",
               bool(cameras) and max_delta <= policy["max_camera_timestamp_delta_ms"],
               f"max_delta_ms={max_delta}",
               [item["artifact"]["artifact_id"] for item in cameras]),
        _check("calibration_presence", calibration_present,
               f"present={calibration_present}",
               [calibration["artifact"]["artifact_id"]]
               if calibration["artifact"] else []),
        _check("calibration_version_consistency", calibration_consistent,
               "expected=" + calibration["expected_version"] +
               ";observed=" + str(calibration["observed_version"]),
               [calibration["artifact"]["artifact_id"]]
               if calibration["artifact"] else []),
        _check("pose_odom_continuity", motion_ok,
               "pose_gap_ms=" + str(pose["max_gap_ms"]) +
               ";odom_gap_ms=" + str(odom["max_gap_ms"]) +
               ";pose_breaks=" + str(pose["discontinuity_count"]) +
               ";odom_breaks=" + str(odom["discontinuity_count"]),
               [stream["artifact"]["artifact_id"] for stream in (pose, odom)
                if stream["artifact"]]),
    ]
    failures = [item for item in checks if item["status"] == "fail"]
    outcome = "passed" if not failures else "quarantined"
    quarantine = None
    if failures:
        quarantine_body = {
            "ingress_event_id": ingress["event_id"],
            "failed_check_ids": [item["check_id"] for item in failures],
            "reason_codes": [item["reason_code"] for item in failures],
            "path": "quarantine/" + ingress["input_id"] + "/" +
                    policy["policy_version"] + ".json",
        }
        quarantine = {
            "quarantine_id": digest(quarantine_body),
            **quarantine_body,
            "write_mode": "append_only",
            "release_allowed": False,
            "silent_skip": False,
            "required_action": "human_review_or_reingest",
        }
    body = {
        "schema_version": "labelagent.factory-qc-event.v1",
        "event_type": "factory.frame_bundle.qc",
        "occurred_at": occurred_at,
        "input_id": ingress["input_id"],
        "ingress_event_id": ingress["event_id"],
        "idempotency_key": digest({
            "ingress_event_id": ingress["event_id"],
            "policy_sha256": policy["policy_sha256"],
        }),
        "policy": dict(policy),
        "checks": checks,
        "outcome": outcome,
        "candidate_pipeline_allowed": outcome == "passed",
        "quarantine": quarantine,
        "silent_skip": False,
        "write_policy": {
            "mode": "append_only", "overwrite_qc": False,
            "overwrite_source": False,
        },
    }
    body["event_id"] = _event_hash(body)
    validate_qc_event(body)
    return body


def validate_qc_event(event):
    required = {
        "schema_version", "event_type", "occurred_at", "input_id",
        "ingress_event_id", "idempotency_key", "policy", "checks", "outcome",
        "candidate_pipeline_allowed", "quarantine", "silent_skip",
        "write_policy", "event_id",
    }
    if not isinstance(event, dict) or set(event) != required:
        raise ValueError("QC event fields are incomplete or unknown")
    if event["schema_version"] != "labelagent.factory-qc-event.v1" or event["event_type"] != "factory.frame_bundle.qc":
        raise ValueError("invalid QC event type or schema")
    _require_time(event["occurred_at"], "occurred_at")
    for field in ("input_id", "ingress_event_id", "idempotency_key", "event_id"):
        _require_hash(event[field], field)
    policy = event["policy"]
    if not isinstance(policy, dict) or set(policy) != {
        "policy_version", "policy_sha256", "max_camera_timestamp_delta_ms",
        "max_pose_gap_ms", "max_odom_gap_ms",
    }:
        raise ValueError("QC policy fields are incomplete")
    _require_text(policy["policy_version"], "policy.policy_version")
    _require_hash(policy["policy_sha256"], "policy.policy_sha256")
    for field in ("max_camera_timestamp_delta_ms", "max_pose_gap_ms", "max_odom_gap_ms"):
        if isinstance(policy[field], bool) or not isinstance(policy[field], (int, float)) or policy[field] < 0:
            raise ValueError(f"policy.{field} must be non-negative")
    expected_idempotency = digest({
        "ingress_event_id": event["ingress_event_id"],
        "policy_sha256": policy["policy_sha256"],
    })
    if event["idempotency_key"] != expected_idempotency:
        raise ValueError("QC idempotency_key mismatch")
    checks = event["checks"]
    if not isinstance(checks, list) or [item.get("check_id") for item in checks] != list(QC_CHECKS):
        raise ValueError("QC must contain every required check exactly once in canonical order")
    for item in checks:
        if set(item) != {
            "check_id", "status", "reason_code", "observed",
            "evidence_artifact_ids",
        }:
            raise ValueError("QC check fields are incomplete")
        if item["status"] not in {"pass", "fail"}:
            raise ValueError("QC check status is invalid")
        if item["status"] == "pass" and item["reason_code"] is not None:
            raise ValueError("passed QC checks cannot carry a failure reason")
        if item["status"] == "fail" and item["reason_code"] != QC_REASON_CODES[item["check_id"]]:
            raise ValueError("failed QC check reason code is missing or incorrect")
        _require_text(item["observed"], "QC observed evidence")
        if not isinstance(item["evidence_artifact_ids"], list) or len(item["evidence_artifact_ids"]) != len(set(item["evidence_artifact_ids"])):
            raise ValueError("QC evidence artifact IDs must be unique")
        for artifact_id in item["evidence_artifact_ids"]:
            _require_hash(artifact_id, "QC evidence artifact_id")
    failed = [item for item in checks if item["status"] == "fail"]
    if failed:
        if event["outcome"] != "quarantined" or event["candidate_pipeline_allowed"] is not False:
            raise ValueError("any QC failure must fail closed into quarantine")
        quarantine = event["quarantine"]
        if not isinstance(quarantine, dict) or set(quarantine) != {
            "quarantine_id", "ingress_event_id", "failed_check_ids",
            "reason_codes", "path", "write_mode", "release_allowed",
            "silent_skip", "required_action",
        }:
            raise ValueError("quarantine fields are incomplete")
        if quarantine["failed_check_ids"] != [item["check_id"] for item in failed] or quarantine["reason_codes"] != [item["reason_code"] for item in failed]:
            raise ValueError("quarantine must enumerate every failed QC check")
        if quarantine["release_allowed"] is not False or quarantine["silent_skip"] is not False or quarantine["write_mode"] != "append_only":
            raise ValueError("quarantine must remain append-only and fail closed")
        quarantine_body = {
            "ingress_event_id": event["ingress_event_id"],
            "failed_check_ids": quarantine["failed_check_ids"],
            "reason_codes": quarantine["reason_codes"],
            "path": quarantine["path"],
        }
        if quarantine["quarantine_id"] != digest(quarantine_body):
            raise ValueError("quarantine_id mismatch")
    elif event["outcome"] != "passed" or event["candidate_pipeline_allowed"] is not True or event["quarantine"] is not None:
        raise ValueError("all QC checks must pass before candidate processing")
    if event["silent_skip"] is not False or event["write_policy"] != {
        "mode": "append_only", "overwrite_qc": False, "overwrite_source": False,
    }:
        raise ValueError("QC cannot silently skip or overwrite evidence")
    expected_event_id = _event_hash({
        key: value for key, value in event.items() if key != "event_id"
    })
    if event["event_id"] != expected_event_id:
        raise ValueError("QC event_id mismatch")


def validate_label_registry(registry):
    if registry.get("schema_version") != "labelagent.label-registry.v1":
        raise ValueError("label registry v1 is required")
    if registry.get("append_only") is not True:
        raise ValueError("label registry must be append-only")
    labels = registry.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("label registry is empty")
    names = [item.get("name") for item in labels]
    identifiers = [item.get("class_id") for item in labels]
    if len(names) != len(set(names)) or len(identifiers) != len(set(identifiers)):
        raise ValueError("label names and class IDs must be unique")
    if "floor" not in names or "slope" not in names or names.index("floor") == names.index("slope"):
        raise ValueError("floor and slope must remain distinct registered labels")
    if registry.get("training_semantics") != {
        "floor_slope_distinct": True, "permanent_merge_allowed": False,
    }:
        raise ValueError("training semantics must preserve floor and slope")
    if registry.get("downstream_mapping") != TRAVERSABLE_AREA_MAPPING:
        raise ValueError("only downstream may union floor and slope")
    if registry.get("unknown_policy") != {
        "unknown": "human_review_required",
        "new_label_candidate": "human_review_required",
        "ontology_mutation_allowed": False,
        "automatic_truth_promotion": False,
    }:
        raise ValueError("unknown/new labels must remain behind human review")
    supplied = registry.get("registry_content_sha256")
    _require_hash(supplied, "registry_content_sha256")
    expected = digest({
        key: value for key, value in registry.items()
        if key != "registry_content_sha256"
    })
    if supplied != expected:
        raise ValueError("label registry content hash mismatch")


def _promotion_blockers(ingress, qc, request):
    blockers = []
    if qc.get("outcome") != "passed" or not qc.get("candidate_pipeline_allowed"):
        blockers.append("qc_not_passed")
    if ingress["provenance"]["source_kind"] != "real":
        blockers.append("simulation_source_truth_promotion_forbidden")
    source = request.get("source", {})
    identity = ingress["identity"]
    if source.get("source_kind") != "real":
        blockers.append("promotion_source_kind_must_be_real")
    if any(source.get(field) != identity[field] for field in (
        "dataset_id", "source_batch", "sequence", "frame_id"
    )):
        blockers.append("promotion_source_identity_mismatch")
    if source.get("synthetic_test_only") is not False:
        blockers.append("synthetic_or_fixture_truth_promotion_forbidden")
    if source.get("dataset_role") not in {"train_candidate", "val_candidate"}:
        blockers.append("test_or_unknown_dataset_role_truth_promotion_forbidden")
    if source.get("ingress_event_id") != ingress["event_id"] or source.get("qc_event_id") != qc["event_id"]:
        blockers.append("source_event_binding_mismatch")

    review = request.get("review", {})
    if review.get("review_origin") != "genuine_human_review":
        blockers.append("genuine_human_review_missing")
    if review.get("decision") not in {"accepted", "corrected"}:
        blockers.append("human_decision_missing_or_invalid")
    try:
        _require_text(review.get("reviewer_declared_id"), "reviewer_declared_id")
        _require_time(review.get("reviewed_at"), "reviewed_at")
    except ValueError:
        blockers.append("reviewer_or_review_time_missing")
    if review.get("identity_assurance") not in {"self_declared", "authenticated"}:
        blockers.append("reviewer_identity_assurance_missing")

    candidate = request.get("candidate", {})
    truth = request.get("truth", {})
    selected_camera = request.get("source", {}).get("camera")
    camera = next((item for item in ingress["cameras"]
                   if item["camera"] == selected_camera), None)
    source_sha = camera["artifact"]["sha256"] if camera else None
    if candidate.get("source_image_sha256") != source_sha or review.get("source_image_sha256") != source_sha:
        blockers.append("source_image_hash_binding_mismatch")
    if review.get("candidate_sha256") != candidate.get("sha256"):
        blockers.append("candidate_hash_binding_mismatch")
    if candidate.get("layer") != "machine_candidate":
        blockers.append("candidate_layer_invalid")
    if request.get("direct_machine_preannotation_promotion") is not False:
        blockers.append("direct_machine_preannotation_promotion_forbidden")
    if truth.get("layer") != "human_truth":
        blockers.append("truth_layer_must_be_human_truth")

    diff = request.get("before_after_diff", {})
    if diff.get("before_sha256") != candidate.get("sha256") or diff.get("after_sha256") != truth.get("sha256"):
        blockers.append("before_after_diff_binding_mismatch")
    for field in ("added", "removed", "modified"):
        if isinstance(diff.get(field), bool) or not isinstance(diff.get(field), int) or diff.get(field) < 0:
            blockers.append("before_after_diff_counts_invalid")
            break

    version = request.get("version", {})
    if version.get("split") not in {"train", "val"}:
        blockers.append("test_split_truth_promotion_forbidden")
    if not all(version.get(field) is expected for field, expected in (
        ("immutable", True), ("append_only", True),
        ("overwrite_previous_truth", False), ("previous_truth_preserved", True),
    )):
        blockers.append("truth_version_not_immutable_append_only")
    if version.get("truth_sha256") != truth.get("sha256"):
        blockers.append("truth_version_hash_binding_mismatch")

    approval = request.get("approval", {})
    if approval.get("decision") != "approved" or approval.get("review_id") != review.get("review_id"):
        blockers.append("approval_receipt_missing_or_unbound")
    if approval.get("identity_assurance") not in {"self_declared", "authenticated"}:
        blockers.append("approval_identity_assurance_missing")
    if (approval.get("candidate_sha256") != candidate.get("sha256")
            or approval.get("source_image_sha256") != source_sha
            or approval.get("truth_sha256") != truth.get("sha256")):
        blockers.append("approval_artifact_hash_binding_mismatch")
    for container, fields in (
        (candidate, ("artifact_id", "uri", "sha256", "source_image_sha256")),
        (truth, ("artifact_id", "uri", "sha256", "layer")),
        (review, ("review_id", "receipt_uri", "receipt_sha256",
                  "candidate_sha256", "source_image_sha256")),
        (diff, ("artifact_uri", "artifact_sha256", "before_sha256", "after_sha256")),
        (version, ("version_id", "manifest_uri", "manifest_sha256", "truth_sha256")),
        (approval, ("approval_id", "receipt_uri", "receipt_sha256",
                    "review_id", "candidate_sha256", "source_image_sha256",
                    "truth_sha256", "approved_at", "approver_declared_id")),
    ):
        for field in fields:
            value = container.get(field)
            try:
                if field.endswith("sha256"):
                    _require_hash(value, field)
                elif field.endswith("_at"):
                    _require_time(value, field)
                else:
                    _require_text(value, field)
            except ValueError:
                blockers.append("required_promotion_evidence_missing")
                break
    return sorted(set(blockers))


def evaluate_truth_promotion(ingress, qc, request, *, occurred_at):
    validate_ingress_event(ingress)
    validate_qc_event(qc)
    blockers = _promotion_blockers(ingress, qc, request)
    common = {
        "occurred_at": occurred_at,
        "input_id": ingress["input_id"],
        "ingress_event_id": ingress["event_id"],
        "qc_event_id": qc["event_id"],
        "idempotency_key": digest({
            "input_id": ingress["input_id"],
            "review_id": request.get("review", {}).get("review_id"),
            "candidate_sha256": request.get("candidate", {}).get("sha256"),
            "truth_sha256": request.get("truth", {}).get("sha256"),
            "version_id": request.get("version", {}).get("version_id"),
        }),
    }
    if blockers:
        body = {
            "schema_version": "labelagent.truth-promotion-rejection.v1",
            "event_type": "truth.promotion_rejected",
            **common,
            "status": "rejected_fail_closed",
            "blockers": blockers,
            "truth_version_created": False,
            "truth_artifact_written": False,
            "old_truth_modified": False,
            "next_state": "human_review_required",
        }
        body["event_id"] = _event_hash(body)
        validate_truth_rejection(body)
        return body

    body = {
        "schema_version": "labelagent.truth-promotion-event.v1",
        "event_type": "truth.version.promoted",
        **common,
        "source": request["source"],
        "candidate": request["candidate"],
        "review": request["review"],
        "before_after_diff": request["before_after_diff"],
        "truth": request["truth"],
        "version": request["version"],
        "approval": request["approval"],
        "direct_machine_preannotation_promotion": False,
        "write_policy": {
            "mode": "append_only_new_truth_version",
            "overwrite_previous_truth": False,
            "old_truth_preserved": True,
        },
    }
    body["event_id"] = _event_hash(body)
    validate_truth_promotion_event(body)
    return body


def validate_truth_rejection(event):
    required = {
        "schema_version", "event_type", "occurred_at", "input_id",
        "ingress_event_id", "qc_event_id", "idempotency_key", "status",
        "blockers", "truth_version_created", "truth_artifact_written",
        "old_truth_modified", "next_state", "event_id",
    }
    if not isinstance(event, dict) or set(event) != required:
        raise ValueError("truth rejection fields are incomplete or unknown")
    if event.get("schema_version") != "labelagent.truth-promotion-rejection.v1" or event.get("event_type") != "truth.promotion_rejected":
        raise ValueError("invalid truth promotion rejection")
    if event.get("status") != "rejected_fail_closed" or not event.get("blockers"):
        raise ValueError("truth rejection requires blockers")
    if event.get("truth_version_created") is not False or event.get("truth_artifact_written") is not False or event.get("old_truth_modified") is not False:
        raise ValueError("rejected promotion cannot mutate truth")
    if event.get("next_state") != "human_review_required":
        raise ValueError("rejected promotion must return to human review")
    _require_time(event["occurred_at"], "occurred_at")
    for field in ("input_id", "ingress_event_id", "qc_event_id",
                  "idempotency_key", "event_id"):
        _require_hash(event[field], field)
    expected = _event_hash({key: value for key, value in event.items()
                            if key != "event_id"})
    if event.get("event_id") != expected:
        raise ValueError("truth rejection event_id mismatch")


def validate_truth_promotion_event(event):
    required = {
        "schema_version", "event_type", "occurred_at", "input_id",
        "ingress_event_id", "qc_event_id", "idempotency_key", "source",
        "candidate", "review", "before_after_diff", "truth", "version",
        "approval", "direct_machine_preannotation_promotion", "write_policy",
        "event_id",
    }
    if not isinstance(event, dict) or set(event) != required:
        raise ValueError("truth promotion fields are incomplete or unknown")
    if event["schema_version"] != "labelagent.truth-promotion-event.v1" or event["event_type"] != "truth.version.promoted":
        raise ValueError("invalid truth promotion event")
    _require_time(event["occurred_at"], "occurred_at")
    for field in ("input_id", "ingress_event_id", "qc_event_id",
                  "idempotency_key", "event_id"):
        _require_hash(event[field], field)
    expected_fields = {
        "source": {"dataset_role", "synthetic_test_only", "camera",
                   "source_kind", "dataset_id", "source_batch", "sequence",
                   "frame_id", "ingress_event_id", "qc_event_id"},
        "candidate": {"artifact_id", "uri", "sha256",
                      "source_image_sha256", "layer"},
        "review": {"review_id", "review_origin", "reviewer_declared_id",
                   "identity_assurance", "reviewed_at", "decision",
                   "receipt_uri", "receipt_sha256", "candidate_sha256",
                   "source_image_sha256"},
        "before_after_diff": {"artifact_uri", "artifact_sha256",
                              "before_sha256", "after_sha256", "added",
                              "removed", "modified"},
        "truth": {"artifact_id", "uri", "sha256", "layer"},
        "version": {"version_id", "parent_version", "split", "manifest_uri",
                    "manifest_sha256", "truth_sha256", "immutable",
                    "append_only", "overwrite_previous_truth",
                    "previous_truth_preserved"},
        "approval": {"approval_id", "decision", "receipt_uri",
                     "receipt_sha256", "review_id", "candidate_sha256",
                     "source_image_sha256", "truth_sha256", "approved_at",
                     "approver_declared_id", "identity_assurance"},
    }
    for name, fields in expected_fields.items():
        if not isinstance(event[name], dict) or set(event[name]) != fields:
            raise ValueError(f"truth promotion {name} fields are incomplete or unknown")
    if event["source"].get("synthetic_test_only") is not False or event["source"].get("dataset_role") not in {"train_candidate", "val_candidate"}:
        raise ValueError("synthetic/test data cannot be promoted")
    if event["source"].get("source_kind") != "real":
        raise ValueError("only real factory sources can be promoted")
    if (event["source"]["ingress_event_id"] != event["ingress_event_id"]
            or event["source"]["qc_event_id"] != event["qc_event_id"]
            or event["source"]["camera"] not in CAMERAS):
        raise ValueError("truth promotion source binding mismatch")
    if event["review"].get("review_origin") != "genuine_human_review" or event["review"].get("decision") not in {"accepted", "corrected"}:
        raise ValueError("genuine human review is required")
    if event["review"].get("identity_assurance") not in {"self_declared", "authenticated"}:
        raise ValueError("human identity assurance must be explicit")
    _require_text(event["review"]["reviewer_declared_id"], "reviewer_declared_id")
    _require_time(event["review"]["reviewed_at"], "reviewed_at")
    if event["candidate"].get("layer") != "machine_candidate":
        raise ValueError("candidate layer must be machine_candidate")
    if event["truth"].get("layer") != "human_truth":
        raise ValueError("promoted artifact must be human_truth")
    for container, fields in (
        (event["candidate"], ("artifact_id", "sha256", "source_image_sha256")),
        (event["review"], ("receipt_sha256", "candidate_sha256", "source_image_sha256")),
        (event["before_after_diff"], ("artifact_sha256", "before_sha256", "after_sha256")),
        (event["truth"], ("artifact_id", "sha256")),
        (event["version"], ("manifest_sha256", "truth_sha256")),
        (event["approval"], ("receipt_sha256", "candidate_sha256",
                             "source_image_sha256", "truth_sha256")),
    ):
        for field in fields:
            _require_hash(container[field], field)
    for container, fields in (
        (event["candidate"], ("uri",)),
        (event["review"], ("review_id", "receipt_uri")),
        (event["before_after_diff"], ("artifact_uri",)),
        (event["truth"], ("uri",)),
        (event["version"], ("version_id", "manifest_uri")),
        (event["approval"], ("approval_id", "receipt_uri",
                             "approver_declared_id")),
    ):
        for field in fields:
            _require_text(container[field], field)
    diff = event["before_after_diff"]
    if (diff["before_sha256"] != event["candidate"]["sha256"]
            or diff["after_sha256"] != event["truth"]["sha256"]):
        raise ValueError("before/after diff hash binding mismatch")
    for field in ("added", "removed", "modified"):
        if isinstance(diff[field], bool) or not isinstance(diff[field], int) or diff[field] < 0:
            raise ValueError("before/after diff counts must be non-negative")
    version = event["version"]
    if version["split"] not in {"train", "val"} or version["truth_sha256"] != event["truth"]["sha256"]:
        raise ValueError("truth version split or truth hash binding is invalid")
    if not all(version[field] is expected for field, expected in (
        ("immutable", True), ("append_only", True),
        ("overwrite_previous_truth", False), ("previous_truth_preserved", True),
    )):
        raise ValueError("truth version must be immutable and preserve old truth")
    approval = event["approval"]
    if (approval["decision"] != "approved"
            or approval["review_id"] != event["review"]["review_id"]
            or approval["candidate_sha256"] != event["candidate"]["sha256"]
            or approval["source_image_sha256"] != event["candidate"]["source_image_sha256"]
            or approval["truth_sha256"] != event["truth"]["sha256"]):
        raise ValueError("approval receipt is not bound to review and artifacts")
    if approval["identity_assurance"] not in {"self_declared", "authenticated"}:
        raise ValueError("approval identity assurance must be explicit")
    _require_time(approval["approved_at"], "approved_at")
    if event["direct_machine_preannotation_promotion"] is not False:
        raise ValueError("machine preannotations cannot be directly promoted")
    if event["write_policy"] != {
        "mode": "append_only_new_truth_version",
        "overwrite_previous_truth": False,
        "old_truth_preserved": True,
    }:
        raise ValueError("truth promotion must preserve old truth")
    expected = _event_hash({key: value for key, value in event.items()
                            if key != "event_id"})
    if event["event_id"] != expected:
        raise ValueError("truth promotion event_id mismatch")


def validate_domain_event(event):
    event_type = event.get("event_type") if isinstance(event, dict) else None
    if event_type == "factory.frame_bundle.ingested":
        validate_ingress_event(event)
    elif event_type == "factory.frame_bundle.qc":
        validate_qc_event(event)
    elif event_type == "truth.version.promoted":
        validate_truth_promotion_event(event)
    elif event_type == "truth.promotion_rejected":
        validate_truth_rejection(event)
    else:
        raise ValueError("unsupported factory ledger event type")


def _record(event, sequence, previous_record_sha256):
    validate_domain_event(event)
    body = {
        "schema_version": "labelagent.factory-ledger-record.v1",
        "sequence": sequence,
        "previous_record_sha256": previous_record_sha256,
        "idempotency_key": event["idempotency_key"],
        "event_type": event["event_type"],
        "event_sha256": digest(event),
        "event": event,
    }
    return {**body, "record_sha256": digest(body)}


def _validate_ledger_context(records, event):
    events = [item["event"] for item in records]
    if event["event_type"] == "factory.frame_bundle.qc":
        ingress = next((item for item in events
                        if item["event_type"] == "factory.frame_bundle.ingested"
                        and item["event_id"] == event["ingress_event_id"]), None)
        if ingress is None or ingress["input_id"] != event["input_id"]:
            raise ValueError("QC event is not bound to an earlier ingress event")
        rebuilt_qc = evaluate_qc(
            ingress, occurred_at=event["occurred_at"], policy=event["policy"])
        if rebuilt_qc != event:
            raise ValueError("QC event does not match fail-closed evaluation of ingress")
    if event["event_type"] in {"truth.version.promoted", "truth.promotion_rejected"}:
        ingress = next((item for item in events
                        if item["event_type"] == "factory.frame_bundle.ingested"
                        and item["event_id"] == event["ingress_event_id"]), None)
        qc = next((item for item in events
                   if item["event_type"] == "factory.frame_bundle.qc"
                   and item["event_id"] == event["qc_event_id"]), None)
        if ingress is None or qc is None:
            raise ValueError("truth decision requires earlier ingress and QC events")
        if (ingress["input_id"] != event["input_id"]
                or qc["input_id"] != event["input_id"]
                or qc["ingress_event_id"] != ingress["event_id"]):
            raise ValueError("truth decision input/event linkage mismatch")
        if event["event_type"] == "truth.version.promoted" and qc["outcome"] != "passed":
            raise ValueError("quarantined input cannot produce a truth version")
        if event["event_type"] == "truth.version.promoted":
            request = {
                key: event[key] for key in (
                    "source", "candidate", "review", "before_after_diff",
                    "truth", "version", "approval",
                    "direct_machine_preannotation_promotion",
                )
            }
            rebuilt = evaluate_truth_promotion(
                ingress, qc, request, occurred_at=event["occurred_at"])
            if rebuilt != event:
                raise ValueError(
                    "truth promotion is not bound to ingress, QC, and human evidence")


def build_ledger_snapshot(events, *, ledger_id, generated_at, sample_kind):
    records = []
    previous = None
    seen = {}
    versions = set()
    for event in events:
        event_sha = digest(event)
        key = event["idempotency_key"]
        if key in seen:
            if seen[key] != event_sha:
                raise ValueError("idempotency key replay changed immutable event")
            continue
        if event["event_type"] == "truth.version.promoted":
            version_id = event["version"]["version_id"]
            if version_id in versions:
                raise ValueError("truth version IDs are immutable and unique")
            versions.add(version_id)
        _validate_ledger_context(records, event)
        record = _record(event, len(records) + 1, previous)
        records.append(record)
        previous = record["record_sha256"]
        seen[key] = event_sha
    body = {
        "schema_version": "labelagent.factory-ledger-snapshot.v1",
        "ledger_id": ledger_id,
        "generated_at": generated_at,
        "sample_kind": sample_kind,
        "append_only": True,
        "overwrite_allowed": False,
        "record_count": len(records),
        "head_record_sha256": previous,
        "records": records,
    }
    return {**body, "snapshot_sha256": digest(body)}


def validate_ledger_snapshot(snapshot):
    if snapshot.get("schema_version") != "labelagent.factory-ledger-snapshot.v1":
        raise ValueError("factory ledger snapshot v1 is required")
    if snapshot.get("append_only") is not True or snapshot.get("overwrite_allowed") is not False:
        raise ValueError("factory ledger must be append-only")
    records = snapshot.get("records")
    if not isinstance(records, list) or snapshot.get("record_count") != len(records):
        raise ValueError("ledger record_count mismatch")
    previous = None
    for index, record in enumerate(records, 1):
        if record.get("sequence") != index or record.get("previous_record_sha256") != previous:
            raise ValueError("ledger sequence or previous hash mismatch")
        event = record.get("event")
        validate_domain_event(event)
        if record.get("event_sha256") != digest(event):
            raise ValueError("ledger event hash mismatch")
        expected = digest({key: value for key, value in record.items()
                           if key != "record_sha256"})
        if record.get("record_sha256") != expected:
            raise ValueError("ledger record hash mismatch")
        previous = record["record_sha256"]
    if snapshot.get("head_record_sha256") != previous:
        raise ValueError("ledger head hash mismatch")
    expected_snapshot = digest({key: value for key, value in snapshot.items()
                                if key != "snapshot_sha256"})
    if snapshot.get("snapshot_sha256") != expected_snapshot:
        raise ValueError("ledger snapshot hash mismatch")


def _exclusive_json(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        Path(path).unlink(missing_ok=True)
        raise


class AppendOnlyFactoryLedger:
    """Filesystem ledger with exclusive-create records and hash-chain readback."""

    def __init__(self, root, ledger_id):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger_id = _require_text(ledger_id, "ledger_id")

    def records(self):
        result = []
        for path in sorted(self.root.glob("[0-9][0-9][0-9][0-9][0-9][0-9]-*.json")):
            result.append(json.loads(path.read_text(encoding="utf-8")))
        snapshot = build_ledger_snapshot(
            [item["event"] for item in result], ledger_id=self.ledger_id,
            generated_at="1970-01-01T00:00:00Z", sample_kind="runtime_readback",
        )
        # The rebuilt records prove the stored chain, sequence, and content.
        for stored, rebuilt in zip(result, snapshot["records"]):
            if stored != rebuilt:
                raise ValueError("stored ledger record differs from canonical hash chain")
        return result

    def append(self, event):
        validate_domain_event(event)
        records = self.records()
        event_sha = digest(event)
        for record in records:
            if record["idempotency_key"] == event["idempotency_key"]:
                if record["event_sha256"] != event_sha:
                    raise ValueError("idempotency key replay changed immutable event")
                return record, True
        if event["event_type"] == "truth.version.promoted":
            version_id = event["version"]["version_id"]
            if any(item["event"].get("version", {}).get("version_id") == version_id
                   for item in records):
                raise ValueError("truth version IDs are immutable and unique")
        _validate_ledger_context(records, event)
        previous = records[-1]["record_sha256"] if records else None
        record = _record(event, len(records) + 1, previous)
        path = self.root / f"{record['sequence']:06d}-{record['record_sha256']}.json"
        _exclusive_json(path, record)
        readback = json.loads(path.read_text(encoding="utf-8"))
        if readback != record:
            raise ValueError("ledger record readback mismatch")
        return record, False
