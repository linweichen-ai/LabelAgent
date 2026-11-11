import copy
import hashlib
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator, FormatChecker

from tools.formal_ddp_preflight import evaluate


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "docs/test-results/formal-ddp-preflight-input-20261111.json"
DECISION = ROOT / "docs/test-results/formal-ddp-preflight-decision-20261111.json"


class FormalDdpPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(INPUT.read_text(encoding="utf-8"))
        cls.decision = json.loads(DECISION.read_text(encoding="utf-8"))

    def validate(self, instance, schema_name):
        schema = json.loads((ROOT / "schemas" / schema_name).read_text(encoding="utf-8"))
        return list(Draft202012Validator(
            schema, format_checker=FormatChecker()).iter_errors(instance))

    def test_exact_live_preflight_is_schema_valid_reproducible_and_rejected(self):
        self.assertEqual(self.validate(
            self.payload, "formal-ddp-preflight-input.schema.json"), [])
        self.assertEqual(evaluate(self.payload), self.decision)
        self.assertEqual(self.validate(
            self.decision, "formal-ddp-preflight-decision.schema.json"), [])
        self.assertFalse(self.decision["preflight_passed"])
        self.assertFalse(self.decision["process_created"])
        self.assertFalse(self.decision["automatic_launch_performed"])
        config = self.payload["stage0_gate"]
        self.assertEqual(
            hashlib.sha256((ROOT / config["config_path"]).read_bytes()).hexdigest(),
            config["config_sha256"],
        )
        names_payload = (
            "\n".join(config["exact_parameter_names"]) + "\n"
        ).encode("utf-8")
        self.assertEqual(hashlib.sha256(names_payload).hexdigest(),
                         config["exact_names_sha256"])

    def test_gpu_idle_alone_never_authorizes_or_starts_training(self):
        payload = copy.deepcopy(self.payload)
        payload["resource_gate"]["other_user_processes"] = []
        decision = evaluate(payload)
        self.assertFalse(decision["preflight_passed"])
        self.assertIn("genuine_human_pixel_truth_version_missing", decision["blockers"])
        self.assertIn("fixed_independent_holdout_baseline_missing", decision["blockers"])
        self.assertFalse(decision["process_created"])

    def test_wrong_stage0_scope_is_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["stage0_gate"]["exact_parameter_names"] = payload[
            "stage0_gate"]["exact_parameter_names"][:-1]
        decision = evaluate(payload)
        self.assertIn("stage0_exact_six_parameter_scope_not_verified",
                      decision["blockers"])

    def test_output_and_rollback_are_both_hard_gates(self):
        payload = copy.deepcopy(self.payload)
        payload["path_gate"]["rollback_checkpoint_path"] = None
        payload["path_gate"]["rollback_checkpoint_sha256"] = None
        payload["path_gate"]["rollback_checkpoint_verified"] = False
        decision = evaluate(payload)
        self.assertIn("training_output_path_missing", decision["blockers"])
        self.assertIn("rollback_checkpoint_path_or_sha256_not_verified",
                      decision["blockers"])

    def test_all_conditions_pass_still_requires_separate_explicit_launch(self):
        payload = copy.deepcopy(self.payload)
        payload["truth_gate"] = {
            "truth_version_available": True,
            "truth_version_id": "truth-test-only",
            "truth_manifest_sha256": "1" * 64,
            "truth_manifest_sha256_verified": True,
        }
        payload["holdout_gate"] = {
            "fixed_independent_baseline_available": True,
            "baseline_receipt_path": "test-only/holdout.json",
            "protected_from_training": True,
        }
        payload["stage0_gate"]["training_config_executable"] = True
        payload["resource_gate"]["other_user_processes"] = []
        payload["path_gate"]["output_path"] = "/test-only/output"
        payload["formal_training_authorized"] = True
        decision = evaluate(payload)
        self.assertTrue(decision["preflight_passed"])
        self.assertEqual(decision["blockers"], [])
        self.assertEqual(decision["next_action"],
                         "explicit_separate_launch_required")
        self.assertFalse(decision["process_created"])
        self.assertFalse(decision["automatic_launch_performed"])

    def test_polling_or_idle_autostart_contract_is_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["execution_contract"]["auto_start_when_idle"] = True
        with self.assertRaisesRegex(ValueError, "no-autostart"):
            evaluate(payload)


if __name__ == "__main__":
    unittest.main()
