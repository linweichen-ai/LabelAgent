import copy
import json
from pathlib import Path
import tempfile
import unittest

from jsonschema import Draft202012Validator, FormatChecker

from source.agent.cvat_offline_contract import (
    AppendOnlyCvatReviewLedger,
    build_export_contract,
    build_review_ledger_snapshot,
    evaluate_cvat_return,
    validate_export_contract,
    validate_review_event,
    validate_review_ledger_snapshot,
)
from source.agent.factory_data_contract import digest, evaluate_qc
from tools.build_factory_data_static_acceptance import POLICY, valid_ingress


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def export_payload(ingress, role="train72", *, reason="LOW_CONFIDENCE"):
    image_sha = ingress["cameras"][0]["artifact"]["sha256"]
    candidate_sha = digest(("candidate:" + role).encode())
    return {
        "dataset_role": role,
        "split": "train" if role == "train72" else "test",
        "identity": {
            "dataset_id": ingress["identity"]["dataset_id"],
            "source_batch": ingress["identity"]["source_batch"],
            "sequence": ingress["identity"]["sequence"],
            "frame_id": ingress["identity"]["frame_id"],
            "camera": "camera_0",
            "input_id": ingress["input_id"],
        },
        "cvat_binding": {
            "task_id": "offline-task-" + role,
            "job_id": "offline-job-" + role,
            "task_version": "task-v1",
            "job_version": "job-v1",
            "package_version": "package-v1",
        },
        "artifacts": {
            "image_uri": "example://cvat/image.png",
            "image_sha256": image_sha,
            "input_sha256": ingress["input_id"],
            "candidate_uri": "example://cvat/candidate.json",
            "candidate_sha256": candidate_sha,
            "model_sha256": digest(b"model-v1"),
            "config_sha256": digest(b"config-v1"),
            "preannotation_uri": "example://cvat/preannotation.xml",
            "preannotation_sha256": digest(b"preannotation-v1"),
        },
        "routing": {
            "reason_codes": [reason],
            "priority": "high",
            "provenance": "real",
            "unknown_or_new_label_pending": reason in {"UNKNOWN_LABEL", "NEW_LABEL_CANDIDATE"},
        },
    }


def return_payload(export, *, version_id=None):
    truth_sha = digest(("human-truth:" + export["dataset_role"]).encode())
    annotation_sha = digest(("annotation-export:" + export["dataset_role"]).encode())
    version_id = version_id or ("truth-train-v1" if export["dataset_role"] == "train72" else "eval-test-v1")
    return {
        "export_event_id": export["event_id"],
        "dataset_role": export["dataset_role"],
        "split": export["split"],
        "identity": copy.deepcopy(export["identity"]),
        "cvat_binding": copy.deepcopy(export["cvat_binding"]),
        "completion": {
            "status": "human_completed", "actor_type": "human",
            "reviewer": {
                "reviewer_id": "cvat-user-17",
                "identity_assurance": "authenticated",
                "reviewed_at": "2026-11-11T16:01:00+08:00",
            },
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
        "annotation_export": {
            "uri": "example://cvat/human-annotations.json",
            "sha256": annotation_sha, "verified_sha256": annotation_sha,
        },
        "annotation_labels": ["floor", "slope", "pallet"],
        "before_after_diff": {
            "artifact_uri": "example://cvat/diff.json",
            "artifact_sha256": digest(b"diff-v1"),
            "before_sha256": export["artifacts"]["candidate_sha256"],
            "after_sha256": truth_sha,
            "added": 1, "removed": 1, "modified": 2,
        },
        "truth": {
            "artifact_id": digest({"truth": truth_sha}),
            "uri": "example://cvat/truth.json", "sha256": truth_sha,
        },
        "version": {
            "version_id": version_id, "parent_version": "truth-v0",
            "split": export["split"],
            "manifest_uri": "example://cvat/manifest.json",
            "manifest_sha256": digest(("manifest:" + version_id).encode()),
            "truth_sha256": truth_sha, "immutable": True,
            "append_only": True, "overwrite_previous_truth": False,
            "previous_truth_preserved": True,
        },
        "approval": {
            "approval_id": "approval-" + version_id,
            "decision": "approved", "reviewer_id": "cvat-user-17",
            "approved_at": "2026-11-11T16:01:01+08:00",
            "receipt_uri": "example://cvat/approval.json",
            "receipt_sha256": digest(("approval:" + version_id).encode()),
            "annotation_export_sha256": annotation_sha,
            "truth_sha256": truth_sha,
        },
    }


class CvatOfflineContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load(ROOT / "configs/labelagent_label_registry.json")
        cls.ingress = valid_ingress()
        cls.qc = evaluate_qc(cls.ingress, policy=POLICY, occurred_at="2026-11-11T10:00:02+08:00")

    def make_export(self, role="train72", reason="LOW_CONFIDENCE"):
        return build_export_contract(export_payload(self.ingress, role, reason=reason), self.registry,
                                     occurred_at="2026-11-11T16:00:00+08:00")

    def test_adapter_has_no_network_gpu_or_cvat_client_code(self):
        source = (ROOT / "source/agent/cvat_offline_contract.py").read_text(encoding="utf-8")
        for forbidden in ("import requests", "import torch", "import socket",
                          "import subprocess", "import oss2", "cvat_sdk"):
            self.assertNotIn(forbidden, source)

    def test_export_is_preannotation_only_and_split_isolated(self):
        for role, split in (("train72", "train"), ("test24", "test")):
            event = self.make_export(role)
            validate_export_contract(event)
            self.assertEqual(event["split"], split)
            self.assertEqual(event["annotation_layer"], "machine_preannotation")
            self.assertFalse(event["completion"]["human_completed"])
            self.assertFalse(event["write_policy"]["truth_promotion_allowed"])
        invalid = export_payload(self.ingress, "test24")
        invalid["split"] = "train"
        with self.assertRaisesRegex(ValueError, "split"):
            build_export_contract(invalid, self.registry, occurred_at="2026-11-11T16:00:00+08:00")

    def test_unknown_new_label_stays_pending(self):
        event = self.make_export("train72", "NEW_LABEL_CANDIDATE")
        self.assertTrue(event["routing"]["unknown_or_new_label_pending"])
        self.assertEqual(event["review_state"], "human_review_required")

    def test_floor_slope_remain_distinct_with_downstream_union(self):
        event = self.make_export()
        self.assertIn("floor", event["label_registry"]["allowed_labels"])
        self.assertIn("slope", event["label_registry"]["allowed_labels"])
        self.assertEqual(event["label_registry"]["downstream_mapping"]["source_labels"], ["floor", "slope"])

    def test_train72_valid_human_return_delegates_to_truth_promotion(self):
        export = self.make_export("train72")
        event = evaluate_cvat_return(export, return_payload(export), self.registry,
                                     ingress=self.ingress, qc=self.qc,
                                     occurred_at="2026-11-11T16:02:00+08:00")
        validate_review_event(event)
        self.assertEqual(event["event_type"], "cvat.review.return.accepted")
        self.assertEqual(event["truth_promotion_event"]["event_type"], "truth.version.promoted")
        self.assertTrue(event["training_return_allowed"])
        errors = list(Draft202012Validator(
            load(ROOT / "schemas/cvat-return-accepted.schema.json"),
            format_checker=FormatChecker()).iter_errors(event))
        self.assertEqual([], errors)

    def test_test24_human_return_is_evaluation_truth_only(self):
        export = self.make_export("test24")
        event = evaluate_cvat_return(export, return_payload(export), self.registry,
                                     occurred_at="2026-11-11T16:02:00+08:00")
        validate_review_event(event)
        self.assertEqual(event["event_type"], "cvat.evaluation_truth.versioned")
        self.assertFalse(event["training_return_allowed"])
        self.assertFalse(event["truth_eligible_for_training"])

    def test_fail_closed_return_cases(self):
        mutations = {
            "machine_completion": lambda p: p["completion"].update(actor_type="machine"),
            "automatic_completion": lambda p: p["completion"].update(status="automatic_completed"),
            "missing_reviewer": lambda p: p["completion"]["reviewer"].update(reviewer_id=""),
            "task_version_drift": lambda p: p["cvat_binding"].update(task_version="task-v2"),
            "split_mix": lambda p: p.update(split="test"),
            "label_outside_registry": lambda p: p["annotation_labels"].append("mystery_new_class"),
            "tampered_export": lambda p: p["annotation_export"].update(verified_sha256=digest(b"different")),
            "candidate_hash_mismatch": lambda p: p["artifact_bindings"].update(candidate_sha256=digest(b"other")),
        }
        export = self.make_export("train72")
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                payload = return_payload(export, version_id="bad-" + name)
                mutate(payload)
                event = evaluate_cvat_return(export, payload, self.registry,
                                             ingress=self.ingress, qc=self.qc,
                                             occurred_at="2026-11-11T16:02:00+08:00")
                self.assertEqual(event["event_type"], "cvat.review.return.rejected")
                self.assertFalse(event["truth_version_created"])
                self.assertFalse(event["training_return_allowed"])

    def test_train72_requires_ingress_and_qc(self):
        export = self.make_export("train72")
        event = evaluate_cvat_return(export, return_payload(export), self.registry,
                                     occurred_at="2026-11-11T16:02:00+08:00")
        self.assertIn("INGRESS_OR_QC_EVIDENCE_MISSING", event["blockers"])

    def test_review_ledger_idempotency_tamper_and_duplicate_version(self):
        export = self.make_export("test24")
        evaluation = evaluate_cvat_return(export, return_payload(export), self.registry,
                                          occurred_at="2026-11-11T16:02:00+08:00")
        snapshot = build_review_ledger_snapshot([export, evaluation], ledger_id="static-ledger",
                                                generated_at="2026-11-11T16:03:00+08:00",
                                                sample_kind="synthetic_contract_fixture")
        validate_review_ledger_snapshot(snapshot)
        tampered = copy.deepcopy(snapshot)
        tampered["records"][0]["event"]["review_state"] = "human_completed"
        with self.assertRaises(ValueError):
            validate_review_ledger_snapshot(tampered)
        with tempfile.TemporaryDirectory() as directory:
            ledger = AppendOnlyCvatReviewLedger(directory, "runtime-ledger")
            first, replay = ledger.append(export)
            self.assertFalse(replay)
            second, replay = ledger.append(export)
            self.assertTrue(replay)
            self.assertEqual(first, second)
            ledger.append(evaluation)
            duplicate = copy.deepcopy(evaluation)
            duplicate["idempotency_key"] = digest(b"new-key")
            duplicate["event_id"] = digest({key: value for key, value in duplicate.items() if key != "event_id"})
            with self.assertRaisesRegex(ValueError, "version IDs"):
                ledger.append(duplicate)

    def test_emitted_static_files_validate_exactly(self):
        cases = (
            ("examples/cvat_train72_preannotation_export_static.json", "schemas/cvat-offline-export.schema.json"),
            ("examples/cvat_test24_human_completed_return_static.json", "schemas/cvat-offline-return.schema.json"),
            ("examples/cvat_test24_evaluation_return_static.json", "schemas/cvat-evaluation-truth.schema.json"),
            ("examples/cvat_machine_completion_rejected_static.json", "schemas/cvat-return-rejection.schema.json"),
            ("docs/test-results/cvat-offline-review-ledger-static-20261111.json", "schemas/cvat-review-ledger.schema.json"),
            ("docs/test-results/cvat-offline-static-validation-20261111.json", "schemas/cvat-offline-static-validation-receipt.schema.json"),
        )
        for document, schema in cases:
            with self.subTest(document=document):
                errors = list(Draft202012Validator(load(ROOT / schema), format_checker=FormatChecker()).iter_errors(load(ROOT / document)))
                self.assertEqual([], errors)

    def test_static_receipt_proves_no_cvat_gpu_or_production_truth(self):
        receipt = load(ROOT / "docs/test-results/cvat-offline-static-validation-20261111.json")
        self.assertFalse(receipt["execution_boundary"]["cvat_connected"])
        self.assertFalse(receipt["execution_boundary"]["uploaded"])
        self.assertFalse(receipt["execution_boundary"]["gpu_called"])
        self.assertEqual(receipt["production_truth_versions_created"], 0)
        self.assertEqual(receipt["cnb_status"], "pending_not_contacted")
        for relative, expected in receipt["artifact_sha256"].items():
            self.assertEqual(digest(load(ROOT / relative)), expected, relative)


if __name__ == "__main__":
    unittest.main()
