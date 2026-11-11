"""Build deterministic, non-production factory contract fixtures on CPU only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.agent.factory_data_contract import (
    build_ingress_event,
    build_ledger_snapshot,
    digest,
    evaluate_qc,
    evaluate_truth_promotion,
)


POLICY_BODY = {
    "policy_version": "factory-qc-static-20261111-v1",
    "max_camera_timestamp_delta_ms": 50,
    "max_pose_gap_ms": 100,
    "max_odom_gap_ms": 100,
}
POLICY = {**POLICY_BODY, "policy_sha256": digest(POLICY_BODY)}


def artifact(name, *, integrity="verified"):
    return {
        "artifact_id": digest({"fixture_artifact": name}),
        "uri": "example://factory-contract/" + name,
        "sha256": digest(("fixture-bytes:" + name).encode("utf-8")),
        "bytes": 1000 + len(name),
        "integrity": integrity,
    }


def motion_stream(name, *, max_gap_ms, discontinuity_count=0):
    return {
        "present": True,
        "artifact": artifact(name),
        "max_gap_ms": max_gap_ms,
        "discontinuity_count": discontinuity_count,
    }


def valid_ingress():
    frame_id = "1789897538282478905"
    payload = {
        "identity": {
            "dataset_id": "EXAMPLE_FACTORY_DATASET",
            "source_batch": "EXAMPLE_BATCH_A",
            "sequence": "000",
            "frame_id": frame_id,
            "capture_time": "2026-11-11T10:00:00+08:00",
        },
        "provenance": {"source_kind": "real", "simulation": None},
        "cameras": [
            {"camera": f"camera_{index}", "frame_id": frame_id,
             "captured_at": f"2026-11-11T10:00:00.{index * 5:03d}+08:00",
             "artifact": artifact(f"pass/camera_{index}.png")}
            for index in range(4)
        ],
        "calibration": {
            "present": True,
            "expected_version": "calib-factory-v3",
            "observed_version": "calib-factory-v3",
            "camera_ids": [f"camera_{index}" for index in range(4)],
            "artifact": artifact("pass/calibration.json"),
        },
        "motion": {
            "pose": motion_stream("pass/pose.txt", max_gap_ms=20),
            "odom": motion_stream("pass/odom.mcap", max_gap_ms=30),
        },
    }
    return build_ingress_event(
        payload, occurred_at="2026-11-11T10:00:01+08:00")


def quarantined_ingress():
    frame_id = "1789897539282478905"
    payload = {
        "identity": {
            "dataset_id": "EXAMPLE_FACTORY_DATASET",
            "source_batch": "EXAMPLE_BATCH_A",
            "sequence": "001",
            "frame_id": frame_id,
            "capture_time": "2026-11-11T10:01:00+08:00",
        },
        "provenance": {"source_kind": "real", "simulation": None},
        "cameras": [
            {"camera": "camera_0", "frame_id": frame_id,
             "captured_at": "2026-11-11T10:01:00.000+08:00",
             "artifact": artifact("quarantine/camera_0.png")},
            {"camera": "camera_1", "frame_id": "WRONG_FRAME",
             "captured_at": "2026-11-11T10:01:00.250+08:00",
             "artifact": artifact("quarantine/camera_1.png", integrity="corrupt")},
            {"camera": "camera_2", "frame_id": frame_id,
             "captured_at": "2026-11-11T10:01:00.010+08:00",
             "artifact": artifact("quarantine/camera_2.png")},
        ],
        "calibration": {
            "present": True,
            "expected_version": "calib-factory-v3",
            "observed_version": "calib-factory-v2",
            "camera_ids": ["camera_0", "camera_1", "camera_2"],
            "artifact": artifact("quarantine/calibration-old.json"),
        },
        "motion": {
            "pose": motion_stream(
                "quarantine/pose.txt", max_gap_ms=350,
                discontinuity_count=2),
            "odom": motion_stream(
                "quarantine/odom.mcap", max_gap_ms=420,
                discontinuity_count=3),
        },
    }
    return build_ingress_event(
        payload, occurred_at="2026-11-11T10:01:01+08:00")


def rejected_promotion(ingress, qc):
    camera_sha = ingress["cameras"][0]["artifact"]["sha256"]
    candidate_sha = digest(b"static-machine-preannotation")
    truth_sha = digest(b"static-unapproved-truth-placeholder")
    request = {
        "source": {
            "dataset_role": "test",
            "synthetic_test_only": True,
            "source_kind": "real",
            "dataset_id": ingress["identity"]["dataset_id"],
            "source_batch": ingress["identity"]["source_batch"],
            "sequence": ingress["identity"]["sequence"],
            "frame_id": ingress["identity"]["frame_id"],
            "camera": "camera_0",
            "ingress_event_id": ingress["event_id"],
            "qc_event_id": qc["event_id"],
        },
        "candidate": {
            "artifact_id": digest({"candidate": candidate_sha}),
            "uri": "example://factory-contract/candidate.json",
            "sha256": candidate_sha,
            "source_image_sha256": camera_sha,
            "layer": "machine_candidate",
        },
        "review": {
            "review_id": "STATIC_MACHINE_REVIEW_NOT_HUMAN",
            "review_origin": "machine_preannotation",
            "reviewer_declared_id": "STATIC_AUTOMATION",
            "identity_assurance": "self_declared",
            "reviewed_at": "2026-11-11T10:02:00+08:00",
            "decision": "accepted",
            "receipt_uri": "example://factory-contract/machine-review.json",
            "receipt_sha256": digest(b"machine-review"),
            "candidate_sha256": candidate_sha,
            "source_image_sha256": camera_sha,
        },
        "before_after_diff": {
            "artifact_uri": "example://factory-contract/diff.json",
            "artifact_sha256": digest(b"static-diff"),
            "before_sha256": candidate_sha,
            "after_sha256": truth_sha,
            "added": 0, "removed": 0, "modified": 0,
        },
        "truth": {
            "artifact_id": digest({"truth": truth_sha}),
            "uri": "example://factory-contract/unapproved-truth.json",
            "sha256": truth_sha,
            "layer": "human_truth",
        },
        "version": {
            "version_id": "STATIC_FORBIDDEN_TEST_VERSION",
            "parent_version": None,
            "split": "test",
            "manifest_uri": "example://factory-contract/test-manifest.json",
            "manifest_sha256": digest(b"static-test-manifest"),
            "truth_sha256": truth_sha,
            "immutable": True,
            "append_only": True,
            "overwrite_previous_truth": False,
            "previous_truth_preserved": True,
        },
        "approval": {
            "approval_id": "STATIC_MACHINE_APPROVAL_NOT_HUMAN",
            "decision": "approved",
            "receipt_uri": "example://factory-contract/machine-approval.json",
            "receipt_sha256": digest(b"machine-approval"),
            "review_id": "STATIC_MACHINE_REVIEW_NOT_HUMAN",
            "candidate_sha256": candidate_sha,
            "source_image_sha256": camera_sha,
            "truth_sha256": truth_sha,
            "approved_at": "2026-11-11T10:02:01+08:00",
            "approver_declared_id": "STATIC_AUTOMATION",
            "identity_assurance": "self_declared",
        },
        "direct_machine_preannotation_promotion": True,
    }
    return evaluate_truth_promotion(
        ingress, qc, request, occurred_at="2026-11-11T10:02:02+08:00")


def build_static_acceptance():
    accepted_ingress = valid_ingress()
    accepted_qc = evaluate_qc(
        accepted_ingress, occurred_at="2026-11-11T10:00:02+08:00",
        policy=POLICY)
    rejected_ingress = quarantined_ingress()
    rejected_qc = evaluate_qc(
        rejected_ingress, occurred_at="2026-11-11T10:01:02+08:00",
        policy=POLICY)
    truth_rejection = rejected_promotion(accepted_ingress, accepted_qc)
    ledger = build_ledger_snapshot(
        [accepted_ingress, accepted_qc, rejected_ingress, rejected_qc,
         truth_rejection],
        ledger_id="factory-static-contract-20261111-v1",
        generated_at="2026-11-11T18:00:00+08:00",
        sample_kind="static_contract_fixture",
    )
    return {
        "pass_example": {
            "schema_version": "labelagent.factory-static-example.v1",
            "sample_kind": "static_contract_fixture",
            "claim_boundary": "not_production_data_not_human_truth",
            "ingress": accepted_ingress,
            "qc": accepted_qc,
        },
        "quarantine_example": {
            "schema_version": "labelagent.factory-static-example.v1",
            "sample_kind": "static_contract_fixture",
            "claim_boundary": "not_production_data_not_human_truth",
            "ingress": rejected_ingress,
            "qc": rejected_qc,
        },
        "truth_rejection_example": {
            "schema_version": "labelagent.factory-static-example.v1",
            "sample_kind": "static_contract_fixture",
            "claim_boundary": "machine_synthetic_test_input_never_promoted",
            "rejection": truth_rejection,
        },
        "ledger": ledger,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=(
        "pass", "quarantine", "truth-rejection", "ledger", "all"
    ), default="all")
    args = parser.parse_args()
    result = build_static_acceptance()
    key = {
        "pass": "pass_example",
        "quarantine": "quarantine_example",
        "truth-rejection": "truth_rejection_example",
        "ledger": "ledger",
    }.get(args.kind)
    payload = result if key is None else result[key]
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
