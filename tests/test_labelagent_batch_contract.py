import copy
import hashlib
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator, FormatChecker

from tools.labelagent_batch_contract import (
    build_execution_plan,
    build_rejection_receipt,
    build_static_review_queue_fixture,
    canonical_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/test-results"
GENERATED_AT = "2026-11-11T17:40:27+08:00"
PLAN_VERSION = "labelagent-batch-plan-20261111-v1"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def content_hash(value, field):
    return canonical_sha256({key: item for key, item in value.items()
                             if key != field})


class LabelAgentBatchContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = load(RESULTS / "candidate-batch-manifest-train72-20261111.json")
        cls.test = load(RESULTS / "candidate-batch-manifest-test24-20261111.json")
        cls.readback = load(RESULTS / "labelagent-batch-resource-readback-20261111.json")
        cls.plan = load(RESULTS / "labelagent-batch-execution-plan-rejected-20261111.json")
        cls.receipt = load(RESULTS / "labelagent-batch-admission-rejected-20261111.json")
        cls.queue = load(RESULTS / "labelagent-human-review-queue-static-sample-20261111.json")

    def schema_errors(self, schema_name, value):
        schema = load(ROOT / "schemas" / schema_name)
        return list(Draft202012Validator(
            schema, format_checker=FormatChecker()).iter_errors(value))

    def assert_schema_valid(self, schema_name, value):
        errors = self.schema_errors(schema_name, value)
        self.assertEqual(errors, [], [error.message for error in errors])

    def assert_schema_invalid(self, schema_name, value):
        self.assertTrue(self.schema_errors(schema_name, value))

    def rehash_manifest(self, manifest):
        manifest["manifest_content_sha256"] = content_hash(
            manifest, "manifest_content_sha256")

    def rehash_readback(self, readback):
        readback["readback_content_sha256"] = content_hash(
            readback, "readback_content_sha256")

    def test_current_resource_readback_is_exact_fail_closed_and_schema_valid(self):
        self.assert_schema_valid(
            "labelagent-batch-resource-readback.schema.json", self.readback)
        self.assertEqual(
            self.readback["readback_content_sha256"],
            content_hash(self.readback, "readback_content_sha256"),
        )
        self.assertFalse(self.readback["admission"]["candidate_generation_allowed"])
        self.assertEqual(len(self.readback["admission"]["blockers"]), 3)
        self.assertEqual(self.readback["execution"]["gpu_model_calls"], 0)
        self.assertEqual(self.readback["execution"]["inference_service_calls"], 0)
        self.assertFalse(self.readback["execution"]["remote_mutation_performed"])
        self.assertFalse(self.readback["data_gate"]["human_truth_available"])

    def test_full_train72_plan_is_deterministic_and_every_record_is_bound(self):
        rebuilt = build_execution_plan(
            self.train, self.readback, PLAN_VERSION, GENERATED_AT)
        self.assertEqual(rebuilt, self.plan)
        self.assert_schema_valid(
            "labelagent-batch-execution-plan.schema.json", self.plan)
        self.assertEqual(self.plan["record_count"], 72)
        self.assertEqual(len(self.plan["records"]), 72)
        self.assertEqual(self.plan["plan_content_sha256"],
                         content_hash(self.plan, "plan_content_sha256"))
        for manifest_entry, record in zip(
            self.train["entries"], self.plan["records"]
        ):
            self.assertEqual(record["split"], "train72")
            self.assertEqual(record["sequence"], manifest_entry["sequence"])
            self.assertEqual(record["frame_id"], manifest_entry["frame_id"])
            self.assertEqual(record["camera"], manifest_entry["camera"])
            self.assertEqual(record["source_image_sha256"],
                             manifest_entry["image_sha256"])
            self.assertEqual(record["preannotation_sha256"],
                             manifest_entry["preannotation_sha256"])
            self.assertEqual(record["model_sha256"]["sam3"],
                             manifest_entry["sam3_model_sha256"])
            self.assertEqual(record["model_sha256"]["vlm"],
                             manifest_entry["vlm_config_sha256"])
            self.assertEqual(record["config_sha256"],
                             manifest_entry["pipeline_config_sha256"])
            self.assertEqual(record["idempotency_key"],
                             manifest_entry["idempotency_key"])
            self.assertEqual(record["output_candidate_path"],
                             manifest_entry["output_candidate_path"])
            self.assertTrue(record["candidate_only"])
            self.assertFalse(record["truth_eligible"])
            self.assertFalse(record["automatic_truth_promotion"])
            self.assertEqual([item["stage"] for item in record["stages"]], [
                "sam_candidate", "vlm_review", "agent_action", "vlm_recheck"
            ])
            self.assertEqual(record["stages"][0]["status"], "blocked")
            self.assertEqual(record["stages"][0]["failure_reason"],
                             "resource_gate_not_passed")
            self.assertTrue(all(not item["artifact_refs"]
                                for item in record["stages"]))

    def test_test24_is_rejected_even_if_resource_gate_is_forged_open(self):
        admitted = copy.deepcopy(self.readback)
        admitted["admission"] = {
            "candidate_generation_allowed": True,
            "blockers": [],
        }
        self.rehash_readback(admitted)
        with self.assertRaisesRegex(
            ValueError, "test24_is_evaluation_only_candidate_generation_forbidden"
        ):
            build_execution_plan(
                self.test, admitted, "forbidden-test24-plan", GENERATED_AT)

    def test_forged_test24_plan_is_rejected_by_schema(self):
        forged = copy.deepcopy(self.plan)
        forged["input_manifest"]["split"] = "test24"
        forged["records"][0]["split"] = "test24"
        forged["plan_content_sha256"] = content_hash(
            forged, "plan_content_sha256")
        self.assert_schema_invalid(
            "labelagent-batch-execution-plan.schema.json", forged)

    def test_truth_eligible_or_automatic_promotion_is_rejected_semantically(self):
        for field in ("truth_eligible", "automatic_truth_promotion"):
            unsafe = copy.deepcopy(self.train)
            unsafe["entries"][0][field] = True
            self.rehash_manifest(unsafe)
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "candidate/truth separation"
            ):
                build_execution_plan(
                    unsafe, self.readback, PLAN_VERSION, GENERATED_AT)

    def test_overwrite_or_truth_output_path_is_rejected(self):
        unsafe = copy.deepcopy(self.train)
        unsafe["entries"][0]["output_candidate_path"] = (
            "truth/000/1789897538282478905_camera_0.json"
        )
        self.rehash_manifest(unsafe)
        with self.assertRaisesRegex(ValueError, "unsafe candidate path"):
            build_execution_plan(
                unsafe, self.readback, PLAN_VERSION, GENERATED_AT)

        unsafe = copy.deepcopy(self.train)
        unsafe["write_policy"]["overwrite_human_truth"] = True
        self.rehash_manifest(unsafe)
        with self.assertRaisesRegex(ValueError, "append-only candidate write policy"):
            build_execution_plan(
                unsafe, self.readback, PLAN_VERSION, GENERATED_AT)

    def test_resource_rejection_never_creates_calls_outputs_or_truth(self):
        self.assertEqual(self.plan["plan_state"], "rejected_resource_gate")
        self.assertFalse(self.plan["resource_admission"]["allowed"])
        self.assertEqual(self.plan["execution_counters"], {
            "gpu_calls": 0,
            "inference_service_calls": 0,
            "candidate_artifacts_written": 0,
            "truth_artifacts_written": 0,
            "processes_created": 0,
        })
        rebuilt = build_rejection_receipt(
            self.plan,
            RESULTS / "candidate-batch-manifest-train72-20261111.json",
            RESULTS / "labelagent-batch-resource-readback-20261111.json",
        )
        # Stored receipt intentionally uses repository-relative paths.
        rebuilt["input_manifest"]["path"] = self.receipt["input_manifest"]["path"]
        rebuilt["resource_readback"]["path"] = self.receipt["resource_readback"]["path"]
        rebuilt["receipt_content_sha256"] = content_hash(
            rebuilt, "receipt_content_sha256")
        self.assertEqual(rebuilt, self.receipt)
        self.assert_schema_valid(
            "labelagent-batch-admission-receipt.schema.json", self.receipt)
        self.assertEqual(self.receipt["receipt_content_sha256"],
                         content_hash(self.receipt, "receipt_content_sha256"))
        self.assertFalse(self.receipt["truth_promotion_allowed"])

    def test_admitted_resource_still_only_compiles_a_zero_call_plan(self):
        admitted = copy.deepcopy(self.readback)
        admitted["admission"] = {
            "candidate_generation_allowed": True,
            "blockers": [],
        }
        self.rehash_readback(admitted)
        plan = build_execution_plan(
            self.train, admitted, "admitted-static-plan", GENERATED_AT)
        self.assertEqual(plan["plan_state"], "planned_not_executed")
        self.assertTrue(plan["resource_admission"]["allowed"])
        self.assertEqual(plan["execution_counters"]["gpu_calls"], 0)
        self.assertEqual(plan["execution_counters"]["inference_service_calls"], 0)
        self.assertTrue(all(
            stage["status"] == "planned"
            for record in plan["records"] for stage in record["stages"]
        ))

    def test_human_review_fixture_covers_all_routes_and_full_provenance(self):
        rebuilt = build_static_review_queue_fixture(
            self.plan, self.train, "human-review-queue-static-20261111-v1")
        self.assertEqual(rebuilt, self.queue)
        self.assert_schema_valid(
            "labelagent-human-review-queue.schema.json", self.queue)
        self.assertEqual(self.queue["queue_content_sha256"],
                         content_hash(self.queue, "queue_content_sha256"))
        self.assertEqual({item["reason_code"] for item in self.queue["entries"]}, {
            "unknown_or_new_label", "low_confidence",
            "vlm_sam_conflict", "qc_failed",
        })
        for item in self.queue["entries"]:
            self.assertEqual(item["state"], "human_review_required")
            self.assertTrue(item["candidate_only"])
            self.assertFalse(item["truth_eligible"])
            self.assertFalse(item["automatic_truth_promotion"])
            self.assertTrue(item["synthetic_test_only"])
            self.assertEqual(item["candidate_ref"]["artifact_state"],
                             "contract_fixture_not_materialized")
            self.assertEqual(item["provenance"]["split"], "train72")

    def test_review_queue_rejects_missing_provenance_and_truth_promotion(self):
        missing = copy.deepcopy(self.queue)
        del missing["entries"][0]["provenance"]["source_image_sha256"]
        missing["queue_content_sha256"] = content_hash(
            missing, "queue_content_sha256")
        self.assert_schema_invalid(
            "labelagent-human-review-queue.schema.json", missing)

        unsafe = copy.deepcopy(self.queue)
        unsafe["entries"][0]["truth_eligible"] = True
        unsafe["queue_content_sha256"] = content_hash(
            unsafe, "queue_content_sha256")
        self.assert_schema_invalid(
            "labelagent-human-review-queue.schema.json", unsafe)

    def test_review_queue_rejects_unknown_reason_or_missing_candidate_reference(self):
        bad_reason = copy.deepcopy(self.queue)
        bad_reason["entries"][0]["reason_code"] = "auto_accept"
        bad_reason["queue_content_sha256"] = content_hash(
            bad_reason, "queue_content_sha256")
        self.assert_schema_invalid(
            "labelagent-human-review-queue.schema.json", bad_reason)

        no_candidate = copy.deepcopy(self.queue)
        del no_candidate["entries"][0]["candidate_ref"]
        no_candidate["queue_content_sha256"] = content_hash(
            no_candidate, "queue_content_sha256")
        self.assert_schema_invalid(
            "labelagent-human-review-queue.schema.json", no_candidate)

    def test_blocked_stage_requires_failure_reason(self):
        unsafe = copy.deepcopy(self.plan)
        unsafe["records"][0]["stages"][0]["failure_reason"] = None
        unsafe["plan_content_sha256"] = content_hash(
            unsafe, "plan_content_sha256")
        self.assert_schema_invalid(
            "labelagent-batch-execution-plan.schema.json", unsafe)

    def test_config_is_static_and_contains_all_fail_closed_routes(self):
        config = load(ROOT / "configs/labelagent_batch_execution_contract.json")
        self.assertFalse(config["execution_enabled"])
        self.assertEqual(config["execution_mode"], "static_cpu_plan_only")
        self.assertEqual(config["allowed_candidate_input_split"], "train72")
        self.assertFalse(config["test24_policy"]["candidate_generation_allowed"])
        self.assertTrue(config["test24_policy"]["evaluation_only"])
        self.assertEqual(set(config["human_review_routing"]), {
            "unknown_or_new_label", "low_confidence",
            "vlm_sam_conflict", "qc_failed",
        })
        self.assertFalse(config["output_policy"]["truth_eligible"])
        self.assertFalse(config["output_policy"]["overwrite_human_truth"])

    def test_static_validation_receipt_schema_and_artifact_hashes(self):
        receipt = load(
            RESULTS / "labelagent-batch-contract-static-validation-20261111.json"
        )
        self.assert_schema_valid(
            "labelagent-batch-contract-static-validation.schema.json", receipt)
        for artifact in receipt["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertTrue(path.is_file(), artifact["path"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                artifact["sha256"],
                artifact["path"],
            )
        self.assertEqual(receipt["execution_boundary"]["gpu_model_calls"], 0)
        self.assertEqual(
            receipt["execution_boundary"]["inference_service_calls"], 0)
        self.assertFalse(receipt["validation_passed"] is not True)


if __name__ == "__main__":
    unittest.main()
