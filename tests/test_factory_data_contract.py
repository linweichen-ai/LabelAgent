import copy
import json
from pathlib import Path
import tempfile
import unittest

from jsonschema import Draft202012Validator, FormatChecker

from source.agent.factory_data_contract import (
    AppendOnlyFactoryLedger,
    build_ingress_event,
    digest,
    evaluate_qc,
    evaluate_truth_promotion,
    validate_ingress_event,
    validate_label_registry,
    validate_ledger_snapshot,
    validate_qc_event,
)
from tools.build_factory_data_static_acceptance import (
    POLICY,
    build_static_acceptance,
    valid_ingress,
)


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def payload_from_ingress(event):
    return copy.deepcopy({
        "identity": event["identity"],
        "provenance": event["provenance"],
        "cameras": event["cameras"],
        "calibration": event["calibration"],
        "motion": event["motion"],
    })


def positive_promotion_request(ingress, qc, *, version_id="truth-v1",
                               review_id="review-human-001"):
    source_sha = ingress["cameras"][0]["artifact"]["sha256"]
    candidate_sha = digest(("candidate:" + review_id).encode())
    truth_sha = digest(("human-truth:" + review_id).encode())
    return {
        "source": {
            "dataset_role": "train_candidate",
            "synthetic_test_only": False,
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
            "uri": "review://candidate/" + review_id,
            "sha256": candidate_sha,
            "source_image_sha256": source_sha,
            "layer": "machine_candidate",
        },
        "review": {
            "review_id": review_id,
            "review_origin": "genuine_human_review",
            "reviewer_declared_id": "reviewer-A",
            "identity_assurance": "self_declared",
            "reviewed_at": "2026-11-11T10:03:00+08:00",
            "decision": "corrected",
            "receipt_uri": "review://receipt/" + review_id,
            "receipt_sha256": digest(("review-receipt:" + review_id).encode()),
            "candidate_sha256": candidate_sha,
            "source_image_sha256": source_sha,
        },
        "before_after_diff": {
            "artifact_uri": "review://diff/" + review_id,
            "artifact_sha256": digest(("diff:" + review_id).encode()),
            "before_sha256": candidate_sha,
            "after_sha256": truth_sha,
            "added": 1,
            "removed": 0,
            "modified": 2,
        },
        "truth": {
            "artifact_id": digest({"truth": truth_sha}),
            "uri": "truth://artifact/" + version_id,
            "sha256": truth_sha,
            "layer": "human_truth",
        },
        "version": {
            "version_id": version_id,
            "parent_version": "truth-v0",
            "split": "train",
            "manifest_uri": "truth://manifest/" + version_id,
            "manifest_sha256": digest(("manifest:" + version_id).encode()),
            "truth_sha256": truth_sha,
            "immutable": True,
            "append_only": True,
            "overwrite_previous_truth": False,
            "previous_truth_preserved": True,
        },
        "approval": {
            "approval_id": "approval-" + review_id,
            "decision": "approved",
            "receipt_uri": "review://approval/" + review_id,
            "receipt_sha256": digest(("approval:" + review_id).encode()),
            "review_id": review_id,
            "candidate_sha256": candidate_sha,
            "source_image_sha256": source_sha,
            "truth_sha256": truth_sha,
            "approved_at": "2026-11-11T10:03:01+08:00",
            "approver_declared_id": "approver-A",
            "identity_assurance": "self_declared",
        },
        "direct_machine_preannotation_promotion": False,
    }


class FactoryDataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generated = build_static_acceptance()
        cls.pass_example = load(ROOT / "examples/factory_ingress_qc_pass_static.json")
        cls.quarantine_example = load(
            ROOT / "examples/factory_ingress_qc_quarantine_static.json")
        cls.truth_rejection_example = load(
            ROOT / "examples/factory_truth_promotion_rejected_static.json")
        cls.ledger = load(
            ROOT / "docs/test-results/factory-data-append-only-ledger-static-20261111.json")

    def schema_errors(self, name, value):
        schema = load(ROOT / "schemas" / name)
        return list(Draft202012Validator(
            schema, format_checker=FormatChecker()).iter_errors(value))

    def assert_schema_valid(self, name, value):
        errors = self.schema_errors(name, value)
        self.assertEqual(errors, [], [error.message for error in errors])

    def test_static_examples_are_deterministic_and_explicitly_non_production(self):
        self.assertEqual(self.generated["pass_example"], self.pass_example)
        self.assertEqual(self.generated["quarantine_example"],
                         self.quarantine_example)
        self.assertEqual(self.generated["truth_rejection_example"],
                         self.truth_rejection_example)
        self.assertEqual(self.generated["ledger"], self.ledger)
        for item in (self.pass_example, self.quarantine_example,
                     self.truth_rejection_example):
            self.assertEqual(item["sample_kind"], "static_contract_fixture")
        self.assertNotIn("truth.version.promoted", {
            item["event_type"] for item in self.ledger["records"]})

    def test_ingress_binds_identity_provenance_objects_hashes_and_calibration(self):
        ingress = self.pass_example["ingress"]
        validate_ingress_event(ingress)
        self.assert_schema_valid("factory-data-ingress-event.schema.json", ingress)
        self.assertEqual(set(item["camera"] for item in ingress["cameras"]),
                         {f"camera_{index}" for index in range(4)})
        self.assertEqual(ingress["provenance"], {
            "source_kind": "real", "simulation": None})
        self.assertEqual(ingress["calibration"]["observed_version"],
                         "calib-factory-v3")
        self.assertTrue(all(item["artifact"]["uri"].startswith("example://")
                            for item in ingress["cameras"]))
        self.assertFalse(ingress["write_policy"]["remote_mutation"])

    def test_sim_provenance_requires_simulator_scene_seed_and_version(self):
        payload = payload_from_ingress(valid_ingress())
        payload["provenance"] = {"source_kind": "sim", "simulation": None}
        with self.assertRaisesRegex(ValueError, "complete simulation metadata"):
            build_ingress_event(
                payload, occurred_at="2026-11-11T10:04:00+08:00")
        payload["provenance"]["simulation"] = {
            "simulator": "EXAMPLE_SIM",
            "simulator_version": "1.0",
            "scene": "slope-shadow",
            "random_seed": 42,
        }
        event = build_ingress_event(
            payload, occurred_at="2026-11-11T10:04:00+08:00")
        validate_ingress_event(event)
        self.assertEqual(event["provenance"]["source_kind"], "sim")

    def test_qc_pass_requires_all_seven_checks(self):
        qc = self.pass_example["qc"]
        validate_qc_event(qc)
        self.assert_schema_valid("factory-data-qc-event.schema.json", qc)
        self.assertEqual(qc["outcome"], "passed")
        self.assertTrue(qc["candidate_pipeline_allowed"])
        self.assertEqual(len(qc["checks"]), 7)
        self.assertTrue(all(item["status"] == "pass" for item in qc["checks"]))

    def test_corruption_camera_alignment_timestamp_calibration_and_motion_quarantine(self):
        qc = self.quarantine_example["qc"]
        validate_qc_event(qc)
        self.assert_schema_valid("factory-data-qc-event.schema.json", qc)
        expected = {
            "FILE_CORRUPT", "FOUR_CAMERA_INCOMPLETE",
            "CAMERA_FRAME_MISALIGNED", "CAMERA_TIMESTAMP_DELTA_EXCEEDED",
            "CALIBRATION_VERSION_MISMATCH", "POSE_ODOM_DISCONTINUITY",
        }
        self.assertEqual(set(qc["quarantine"]["reason_codes"]), expected)
        self.assertEqual(qc["outcome"], "quarantined")
        self.assertFalse(qc["candidate_pipeline_allowed"])
        self.assertFalse(qc["quarantine"]["release_allowed"])
        self.assertFalse(qc["quarantine"]["silent_skip"])

    def test_missing_calibration_is_quarantined_not_silently_skipped(self):
        payload = payload_from_ingress(valid_ingress())
        payload["calibration"] = {
            "present": False,
            "expected_version": "calib-factory-v3",
            "observed_version": None,
            "camera_ids": [],
            "artifact": None,
        }
        ingress = build_ingress_event(
            payload, occurred_at="2026-11-11T10:05:00+08:00")
        qc = evaluate_qc(
            ingress, occurred_at="2026-11-11T10:05:01+08:00",
            policy=POLICY)
        self.assertEqual(qc["outcome"], "quarantined")
        self.assertIn("CALIBRATION_MISSING", qc["quarantine"]["reason_codes"])
        self.assertIn("CALIBRATION_VERSION_MISMATCH",
                      qc["quarantine"]["reason_codes"])
        self.assertFalse(qc["silent_skip"])

    def test_missing_camera_enters_quarantine(self):
        payload = payload_from_ingress(valid_ingress())
        payload["cameras"] = payload["cameras"][:3]
        ingress = build_ingress_event(
            payload, occurred_at="2026-11-11T10:06:00+08:00")
        qc = evaluate_qc(
            ingress, occurred_at="2026-11-11T10:06:01+08:00",
            policy=POLICY)
        self.assertIn("FOUR_CAMERA_INCOMPLETE",
                      qc["quarantine"]["reason_codes"])
        self.assertFalse(qc["candidate_pipeline_allowed"])

    def test_qc_failure_cannot_be_forged_as_passed(self):
        forged = copy.deepcopy(self.quarantine_example["qc"])
        forged.update(outcome="passed", candidate_pipeline_allowed=True,
                      quarantine=None)
        forged["event_id"] = digest({
            key: value for key, value in forged.items() if key != "event_id"})
        with self.assertRaisesRegex(ValueError, "must fail closed"):
            validate_qc_event(forged)
        missing = copy.deepcopy(self.pass_example["qc"])
        del missing["checks"][-1]
        missing["event_id"] = digest({
            key: value for key, value in missing.items() if key != "event_id"})
        with self.assertRaisesRegex(ValueError, "every required check"):
            validate_qc_event(missing)

    def test_label_registry_preserves_floor_slope_and_routes_unknown_to_human(self):
        registry = load(ROOT / "configs/labelagent_label_registry.json")
        validate_label_registry(registry)
        self.assert_schema_valid("labelagent-label-registry.schema.json", registry)
        labels = {item["name"]: item["class_id"] for item in registry["labels"]}
        self.assertNotEqual(labels["floor"], labels["slope"])
        self.assertEqual(registry["downstream_mapping"]["source_labels"],
                         ["floor", "slope"])
        self.assertEqual(registry["downstream_mapping"]["target_label"],
                         "traversable_area")
        self.assertEqual(registry["unknown_policy"]["unknown"],
                         "human_review_required")
        self.assertFalse(registry["unknown_policy"]["ontology_mutation_allowed"])

    def test_only_complete_genuine_human_review_can_build_truth_version(self):
        ingress = self.pass_example["ingress"]
        qc = self.pass_example["qc"]
        event = evaluate_truth_promotion(
            ingress, qc, positive_promotion_request(ingress, qc),
            occurred_at="2026-11-11T10:03:02+08:00")
        self.assertEqual(event["event_type"], "truth.version.promoted")
        self.assert_schema_valid(
            "immutable-truth-promotion-event.schema.json", event)
        self.assertEqual(event["review"]["review_origin"],
                         "genuine_human_review")
        self.assertFalse(event["version"]["overwrite_previous_truth"])
        self.assertTrue(event["version"]["previous_truth_preserved"])
        self.assertEqual(event["before_after_diff"]["before_sha256"],
                         event["candidate"]["sha256"])
        self.assertEqual(event["before_after_diff"]["after_sha256"],
                         event["truth"]["sha256"])

    def test_machine_synthetic_and_test_inputs_are_rejected_without_truth(self):
        rejection = self.truth_rejection_example["rejection"]
        self.assert_schema_valid(
            "truth-promotion-rejection-receipt.schema.json", rejection)
        self.assertEqual(rejection["event_type"], "truth.promotion_rejected")
        self.assertIn("direct_machine_preannotation_promotion_forbidden",
                      rejection["blockers"])
        self.assertIn("synthetic_or_fixture_truth_promotion_forbidden",
                      rejection["blockers"])
        self.assertIn("test_split_truth_promotion_forbidden",
                      rejection["blockers"])
        self.assertFalse(rejection["truth_version_created"])
        self.assertFalse(rejection["truth_artifact_written"])
        self.assertFalse(rejection["old_truth_modified"])

    def test_missing_review_hash_diff_or_approval_evidence_is_rejected(self):
        ingress = self.pass_example["ingress"]
        qc = self.pass_example["qc"]
        mutations = {
            "not_human": lambda value: value["review"].update(
                review_origin="machine_preannotation"),
            "reviewer_missing": lambda value: value["review"].update(
                reviewer_declared_id=None),
            "candidate_hash_mismatch": lambda value: value["review"].update(
                candidate_sha256=digest(b"wrong-candidate")),
            "source_hash_mismatch": lambda value: value["review"].update(
                source_image_sha256=digest(b"wrong-source")),
            "diff_missing": lambda value: value["before_after_diff"].pop(
                "artifact_sha256"),
            "approval_missing": lambda value: value["approval"].pop(
                "receipt_sha256"),
        }
        for name, mutate in mutations.items():
            request = positive_promotion_request(
                ingress, qc, review_id="review-" + name)
            mutate(request)
            event = evaluate_truth_promotion(
                ingress, qc, request,
                occurred_at="2026-11-11T10:07:00+08:00")
            with self.subTest(name=name):
                self.assertEqual(event["event_type"],
                                 "truth.promotion_rejected")
                self.assertFalse(event["truth_version_created"])

    def test_quarantine_and_simulation_cannot_produce_truth_version(self):
        bad_ingress = self.quarantine_example["ingress"]
        bad_qc = self.quarantine_example["qc"]
        request = positive_promotion_request(bad_ingress, bad_qc)
        rejected = evaluate_truth_promotion(
            bad_ingress, bad_qc, request,
            occurred_at="2026-11-11T10:08:00+08:00")
        self.assertIn("qc_not_passed", rejected["blockers"])

        payload = payload_from_ingress(valid_ingress())
        payload["provenance"] = {
            "source_kind": "sim",
            "simulation": {"simulator": "EXAMPLE_SIM", "simulator_version": "1",
                           "scene": "factory", "random_seed": 7},
        }
        sim_ingress = build_ingress_event(
            payload, occurred_at="2026-11-11T10:08:01+08:00")
        sim_qc = evaluate_qc(
            sim_ingress, occurred_at="2026-11-11T10:08:02+08:00",
            policy=POLICY)
        request = positive_promotion_request(sim_ingress, sim_qc,
                                             review_id="review-sim")
        rejected = evaluate_truth_promotion(
            sim_ingress, sim_qc, request,
            occurred_at="2026-11-11T10:08:03+08:00")
        self.assertIn("simulation_source_truth_promotion_forbidden",
                      rejected["blockers"])

    def test_append_only_ledger_is_idempotent_and_version_unique(self):
        ingress = self.pass_example["ingress"]
        qc = self.pass_example["qc"]
        promotion = evaluate_truth_promotion(
            ingress, qc, positive_promotion_request(ingress, qc),
            occurred_at="2026-11-11T10:03:02+08:00")
        with tempfile.TemporaryDirectory() as directory:
            ledger = AppendOnlyFactoryLedger(directory, "runtime-test-ledger")
            ledger.append(ingress)
            ledger.append(qc)
            _, replay = ledger.append(promotion)
            self.assertFalse(replay)
            _, replay = ledger.append(promotion)
            self.assertTrue(replay)
            conflict = copy.deepcopy(promotion)
            conflict["occurred_at"] = "2026-11-11T10:03:03+08:00"
            conflict["event_id"] = digest({
                key: value for key, value in conflict.items()
                if key != "event_id"})
            with self.assertRaisesRegex(ValueError, "replay changed"):
                ledger.append(conflict)

            second_request = positive_promotion_request(
                ingress, qc, version_id="truth-v1", review_id="review-human-002")
            duplicate_version = evaluate_truth_promotion(
                ingress, qc, second_request,
                occurred_at="2026-11-11T10:04:02+08:00")
            with self.assertRaisesRegex(ValueError, "version IDs"):
                ledger.append(duplicate_version)
            self.assertEqual(len(ledger.records()), 3)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 3)

    def test_static_ledger_schema_chain_and_tamper_detection(self):
        validate_ledger_snapshot(self.ledger)
        self.assert_schema_valid("factory-evidence-ledger.schema.json",
                                 self.ledger)
        self.assertEqual(self.ledger["record_count"], 5)
        self.assertEqual(self.ledger["records"][-1]["event_type"],
                         "truth.promotion_rejected")
        tampered = copy.deepcopy(self.ledger)
        tampered["records"][0]["event"]["identity"]["frame_id"] = "tampered"
        with self.assertRaisesRegex(ValueError, "(input_id|event_id) mismatch"):
            validate_ledger_snapshot(tampered)

    def test_static_validation_receipt_schema_and_artifact_hashes(self):
        receipt = load(
            ROOT / "docs/test-results/factory-data-static-validation-20261111.json"
        )
        self.assert_schema_valid(
            "factory-data-static-validation-receipt.schema.json", receipt)
        for artifact in receipt["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertTrue(path.is_file(), artifact["path"])
            self.assertEqual(digest(path.read_bytes()), artifact["sha256"],
                             artifact["path"])
        ledger = load(ROOT / receipt["ledger"]["path"])
        self.assertEqual(digest((ROOT / receipt["ledger"]["path"]).read_bytes()),
                         receipt["ledger"]["sha256"])
        self.assertEqual(ledger["snapshot_sha256"],
                         receipt["ledger"]["snapshot_sha256"])
        self.assertEqual(receipt["execution_boundary"]["gpu_calls"], 0)
        self.assertEqual(receipt["execution_boundary"]["oss_writes"], 0)
        self.assertFalse(receipt["truth_gate"]["production_truth_version_created"])


if __name__ == "__main__":
    unittest.main()
