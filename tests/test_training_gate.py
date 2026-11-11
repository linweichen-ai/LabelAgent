import copy
import json
from pathlib import Path
import unittest

from source.agent.training_gate import assert_training_authorized, validate_training_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/test-results/sam3-peft-training-readiness-20261111.json"
SCHEMA = ROOT / "schemas/sam3-peft-training-gate.schema.json"


class TrainingGateTests(unittest.TestCase):
    def setUp(self):
        self.blocked = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_current_project_decision_is_valid_but_training_is_blocked(self):
        self.assertEqual(validate_training_manifest(self.blocked), [])
        with self.assertRaisesRegex(PermissionError, "blocked by the truth gate"):
            assert_training_authorized(self.blocked)

    def test_machine_candidates_and_bev_boxes_can_never_be_truth(self):
        for index in range(3):
            manifest = copy.deepcopy(self.blocked)
            manifest["evidence_scope"] = "synthetic_test_only"
            manifest["data_sources"][index]["truth_eligible"] = True
            self.assertTrue(any("ineligible as truth" in error
                                for error in validate_training_manifest(manifest)))

    def test_full_model_or_unfrozen_backbone_is_rejected(self):
        manifest = copy.deepcopy(self.blocked)
        manifest["evidence_scope"] = "synthetic_test_only"
        manifest["model_strategy"]["full_model_training"] = True
        manifest["model_strategy"]["freeze_image_backbone"] = False
        errors = validate_training_manifest(manifest)
        self.assertIn("full-model training is forbidden by the current decision", errors)
        self.assertIn("image backbone must remain frozen", errors)

    def test_protected_holdouts_cannot_enter_training(self):
        manifest = copy.deepcopy(self.blocked)
        manifest["evidence_scope"] = "synthetic_test_only"
        manifest["isolation"]["protected_sets"][0]["included_in_training"] = True
        self.assertIn("test24 must be excluded from training",
                      validate_training_manifest(manifest))

    def test_authorization_requires_real_truth_version_and_all_isolation_gates(self):
        manifest = copy.deepcopy(self.blocked)
        manifest["evidence_scope"] = "synthetic_test_only"
        manifest["formal_training_authorized"] = True
        errors = validate_training_manifest(manifest)
        self.assertIn("formal training authorization requires all truth and isolation gates", errors)
        self.assertIn("formal training authorization requires non-empty human masks", errors)
        self.assertIn("formal training authorization requires an immutable truth version", errors)
        self.assertIn("formal training authorization requires a truth manifest hash", errors)

    def test_started_flag_cannot_bypass_authorization(self):
        manifest = copy.deepcopy(self.blocked)
        manifest["evidence_scope"] = "synthetic_test_only"
        manifest["formal_training_started"] = True
        self.assertIn("formal training cannot start before authorization",
                      validate_training_manifest(manifest))

    def test_current_manifest_validates_against_schema_unchanged(self):
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            self.skipTest("jsonschema development dependency is absent")
        validator = Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))
        errors = list(validator.iter_errors(self.blocked))
        self.assertEqual(errors, [], [error.message for error in errors])


if __name__ == "__main__":
    unittest.main()
