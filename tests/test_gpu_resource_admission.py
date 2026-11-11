import copy
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator, FormatChecker

from tools.gpu_resource_admission import evaluate


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "docs/test-results/gpu-resource-snapshot-20261111T171049+0800.json"
DECISION = ROOT / "docs/test-results/gpu-resource-admission-20261111T171049+0800.json"


class GpuResourceAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        cls.decision = json.loads(DECISION.read_text(encoding="utf-8"))

    def test_exact_live_decision_is_reproducible(self):
        self.assertEqual(evaluate(self.snapshot), self.decision)

    def test_exact_live_decision_is_schema_valid(self):
        schema = json.loads(
            (ROOT / "schemas/gpu-resource-admission-decision.schema.json")
            .read_text(encoding="utf-8")
        )
        errors = list(Draft202012Validator(
            schema, format_checker=FormatChecker()).iter_errors(self.decision))
        self.assertEqual(errors, [], [error.message for error in errors])

    def test_4090_is_always_protected(self):
        result = evaluate(self.snapshot)
        self.assertFalse(result["decisions"]["new_work_on_factory_4090"]["allowed"])

    def test_foreign_ddp_blocks_both_single_card_batches_and_training(self):
        result = evaluate(self.snapshot)
        decisions = result["decisions"]
        self.assertFalse(decisions["sam3_candidate_batch_via_gpu0_service"]["allowed"])
        self.assertFalse(decisions["labelagent_batch_via_gpu1_service"]["allowed"])
        self.assertFalse(decisions["dual_5090_training"]["allowed"])

    def test_candidate_services_can_be_admitted_after_foreign_processes_exit(self):
        snapshot = copy.deepcopy(self.snapshot)
        expected = {service["pid"] for service in snapshot["dual_5090"]["services"]}
        snapshot["dual_5090"]["processes"] = [
            item for item in snapshot["dual_5090"]["processes"]
            if item["pid"] in expected
        ]
        result = evaluate(snapshot)
        self.assertTrue(result["decisions"]["sam3_candidate_batch_via_gpu0_service"]["allowed"])
        self.assertTrue(result["decisions"]["labelagent_batch_via_gpu1_service"]["allowed"])
        self.assertFalse(result["decisions"]["sam3_finetune_gpu0"]["allowed"])

    def test_spoofed_owned_pid_is_still_foreign(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["dual_5090"]["processes"][0]["command"] = "/tmp/not-the-sam-service"
        result = evaluate(snapshot)
        self.assertFalse(result["decisions"]["sam3_candidate_batch_via_gpu0_service"]["allowed"])


if __name__ == "__main__":
    unittest.main()
