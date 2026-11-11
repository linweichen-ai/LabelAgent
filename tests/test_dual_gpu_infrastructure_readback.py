import hashlib
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = (
    ROOT / "docs/test-results/dual-gpu-infrastructure-live-readback-20261111.json"
)
SCHEMA_PATH = ROOT / "schemas/dual-gpu-infrastructure-live-readback.schema.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DualGpuInfrastructureReadbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def test_exact_receipt_is_schema_valid(self):
        validator = Draft202012Validator(
            self.schema, format_checker=FormatChecker()
        )
        errors = sorted(validator.iter_errors(self.receipt), key=lambda error: list(error.path))
        self.assertEqual(errors, [], [error.message for error in errors])

    def test_recorded_local_tool_hashes_match(self):
        smoke = self.receipt["bf16_smoke"]
        inventory = self.receipt["sam3_inventory"]
        self.assertEqual(sha256(ROOT / smoke["script_path"]), smoke["script_sha256"])
        self.assertEqual(
            sha256(ROOT / inventory["tool_path"]), inventory["tool_sha256"]
        )

    def test_stage0_exact_names_hash_and_scope_are_fixed(self):
        inventory = self.receipt["sam3_inventory"]
        stage0 = inventory["verified_stage0_scope"]
        payload = ("\n".join(stage0["exact_names"]) + "\n").encode("utf-8")
        self.assertEqual(hashlib.sha256(payload).hexdigest(), stage0["exact_names_sha256"])
        self.assertEqual(
            stage0["prefix"], "segmentation_head.mask_predictor.mask_embed."
        )
        self.assertFalse(inventory["conditional_expansion_authorized"])
        self.assertEqual(inventory["conditional_observations"]["adapter_lora_tensor_count"], 0)

    def test_backup_and_claim_boundaries_are_fail_closed(self):
        backup = self.receipt["git_backup"]
        self.assertEqual(
            {backup["local_head"], backup["github_head"], backup["cnb_head"]},
            {backup["local_head"]},
        )
        self.assertTrue(backup["all_equal"])
        boundary = self.receipt["claim_boundary"]
        for key in (
            "human_truth_available",
            "fixed_holdout_baseline_available",
            "formal_training_authorized",
            "training_started",
            "loss_available",
            "checkpoint_available",
            "full_model_training_authorized",
        ):
            self.assertFalse(boundary[key], key)


if __name__ == "__main__":
    unittest.main()
