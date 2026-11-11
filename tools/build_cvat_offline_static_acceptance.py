"""Build deterministic CVAT offline-contract fixtures without CVAT or GPU."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.agent.cvat_offline_contract import (
    build_export_contract,
    build_review_ledger_snapshot,
    evaluate_cvat_return,
)
from source.agent.factory_data_contract import digest
from tools.build_factory_data_static_acceptance import valid_ingress


ROOT = Path(__file__).resolve().parents[1]
STAMP = "2026-11-11T16:00:00+08:00"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def export_payload(ingress, role):
    image_sha = ingress["cameras"][0]["artifact"]["sha256"]
    reason = "LOW_CONFIDENCE" if role == "train72" else "VLM_SAM_CONFLICT"
    return {
        "dataset_role": role,
        "split": "train" if role == "train72" else "test",
        "identity": {
            "dataset_id": ingress["identity"]["dataset_id"],
            "source_batch": ingress["identity"]["source_batch"],
            "sequence": ingress["identity"]["sequence"],
            "frame_id": ingress["identity"]["frame_id"],
            "camera": "camera_0", "input_id": ingress["input_id"],
        },
        "cvat_binding": {
            "task_id": "STATIC-OFFLINE-TASK-" + role,
            "job_id": "STATIC-OFFLINE-JOB-" + role,
            "task_version": "task-v1", "job_version": "job-v1",
            "package_version": "package-v1",
        },
        "artifacts": {
            "image_uri": "example://cvat-static/image.png",
            "image_sha256": image_sha,
            "input_sha256": ingress["input_id"],
            "candidate_uri": "example://cvat-static/candidate.json",
            "candidate_sha256": digest(("candidate:" + role).encode()),
            "model_sha256": digest(b"static-model-v1"),
            "config_sha256": digest(b"static-config-v1"),
            "preannotation_uri": "example://cvat-static/preannotation.xml",
            "preannotation_sha256": digest(("preannotation:" + role).encode()),
        },
        "routing": {
            "reason_codes": [reason], "priority": "high",
            "provenance": "real", "unknown_or_new_label_pending": False,
        },
    }


def human_return(export):
    truth_sha = digest(("static-human-evaluation-truth:" + export["dataset_role"]).encode())
    annotation_sha = digest(("static-annotation-export:" + export["dataset_role"]).encode())
    version_id = "STATIC-EVALUATION-TRUTH-V1"
    return {
        "export_event_id": export["event_id"], "dataset_role": export["dataset_role"],
        "split": export["split"], "identity": copy.deepcopy(export["identity"]),
        "cvat_binding": copy.deepcopy(export["cvat_binding"]),
        "completion": {
            "status": "human_completed", "actor_type": "human",
            "reviewer": {"reviewer_id": "STATIC-AUTHENTICATED-REVIEWER",
                         "identity_assurance": "authenticated",
                         "reviewed_at": "2026-11-11T16:01:00+08:00"},
            "decision": "corrected",
        },
        "artifact_bindings": {
            "source_image_sha256": export["artifacts"]["image_sha256"],
            "input_sha256": export["artifacts"]["input_sha256"],
            "candidate_sha256": export["artifacts"]["candidate_sha256"],
            "model_sha256": export["artifacts"]["model_sha256"],
            "config_sha256": export["artifacts"]["config_sha256"],
            "label_registry_sha256": export["label_registry"]["registry_content_sha256"],
        },
        "annotation_export": {"uri": "example://cvat-static/human-annotations.json",
                              "sha256": annotation_sha, "verified_sha256": annotation_sha},
        "annotation_labels": ["floor", "slope"],
        "before_after_diff": {
            "artifact_uri": "example://cvat-static/diff.json",
            "artifact_sha256": digest(b"static-human-diff"),
            "before_sha256": export["artifacts"]["candidate_sha256"],
            "after_sha256": truth_sha, "added": 1, "removed": 0, "modified": 2,
        },
        "truth": {"artifact_id": digest({"truth": truth_sha}),
                  "uri": "example://cvat-static/evaluation-truth.json", "sha256": truth_sha},
        "version": {"version_id": version_id, "parent_version": "evaluation-v0",
                    "split": "test", "manifest_uri": "example://cvat-static/evaluation-manifest.json",
                    "manifest_sha256": digest(b"static-evaluation-manifest"),
                    "truth_sha256": truth_sha, "immutable": True, "append_only": True,
                    "overwrite_previous_truth": False, "previous_truth_preserved": True},
        "approval": {"approval_id": "STATIC-EVALUATION-APPROVAL",
                     "decision": "approved", "reviewer_id": "STATIC-AUTHENTICATED-REVIEWER",
                     "approved_at": "2026-11-11T16:01:01+08:00",
                     "receipt_uri": "example://cvat-static/evaluation-approval.json",
                     "receipt_sha256": digest(b"static-evaluation-approval"),
                     "annotation_export_sha256": annotation_sha, "truth_sha256": truth_sha},
    }


def build_static_acceptance():
    registry = load(ROOT / "configs/labelagent_label_registry.json")
    ingress = valid_ingress()
    train_export = build_export_contract(export_payload(ingress, "train72"), registry, occurred_at=STAMP)
    test_export = build_export_contract(export_payload(ingress, "test24"), registry, occurred_at=STAMP)
    evaluation = evaluate_cvat_return(test_export, human_return(test_export), registry,
                                      occurred_at="2026-11-11T16:02:00+08:00")
    bad = human_return(train_export)
    bad["version"]["split"] = "train"
    bad["version"]["version_id"] = "STATIC-REJECTED-MACHINE-V1"
    bad["completion"]["actor_type"] = "machine"
    bad["completion"]["status"] = "automatic_completed"
    rejected = evaluate_cvat_return(train_export, bad, registry,
                                    occurred_at="2026-11-11T16:02:01+08:00")
    ledger = build_review_ledger_snapshot(
        [train_export, test_export, evaluation, rejected],
        ledger_id="cvat-offline-static-review-ledger-20261111",
        generated_at="2026-11-11T16:03:00+08:00",
        sample_kind="synthetic_contract_fixture_not_production_truth")
    outputs = {
        ROOT / "examples/cvat_train72_preannotation_export_static.json": train_export,
        ROOT / "examples/cvat_test24_human_completed_return_static.json": human_return(test_export),
        ROOT / "examples/cvat_test24_evaluation_return_static.json": evaluation,
        ROOT / "examples/cvat_machine_completion_rejected_static.json": rejected,
        ROOT / "docs/test-results/cvat-offline-review-ledger-static-20261111.json": ledger,
    }
    receipt = {
        "schema_version": "labelagent.cvat-offline-static-validation.v1",
        "generated_at": "2026-11-11T16:04:00+08:00",
        "sample_kind": "synthetic_contract_fixture_not_production_truth",
        "execution_boundary": {"cpu_only": True, "cvat_connected": False,
                               "uploaded": False, "gpu_called": False,
                               "oss_mutated": False, "remote_service_called": False},
        "contracts": {"train72_export_is_preannotation_only": True,
                      "test24_evaluation_truth_only": True,
                      "test24_training_return_allowed": False,
                      "machine_completion_fail_closed": True,
                      "append_only_review_ledger": True,
                      "floor_slope_distinct": True,
                      "downstream_traversable_union": True},
        "artifact_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): digest(value)
                            for path, value in outputs.items()},
        "production_truth_versions_created": 0,
        "production_training_returns_created": 0,
        "cnb_status": "pending_not_contacted",
    }
    receipt["receipt_sha256"] = digest({key: value for key, value in receipt.items()
                                        if key != "receipt_sha256"})
    outputs[ROOT / "docs/test-results/cvat-offline-static-validation-20261111.json"] = receipt
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    outputs = build_static_acceptance()
    if not args.write:
        print(json.dumps({str(path.relative_to(ROOT)): value for path, value in outputs.items()},
                         ensure_ascii=False, indent=2))
        return
    for path, value in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"append-only output already exists: {path}")
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"readback mismatch: {path}")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
