import hashlib
import json
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parents[1]
SLOPE = WORKSPACE / "outputs/sfl_cpd15y_stage2_20261110/slope_review"
HOLDOUT = WORKSPACE / "outputs/sfl_cpd15y_stage2_20261111/fixed_holdout_chain"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OrderedStageEvidenceTests(unittest.TestCase):
    def validator(self, schema_name):
        from jsonschema import Draft202012Validator, FormatChecker
        schema = json.loads((REPO / "schemas" / schema_name).read_text(encoding="utf-8"))
        return Draft202012Validator(schema, format_checker=FormatChecker())

    def test_policy_correction_supersedes_without_rewriting_old_sidecar(self):
        if not SLOPE.exists():
            self.skipTest("workspace slope evidence is not present in this checkout")
        old = SLOPE / "vlm_slope_candidate_review.json"
        event_path = SLOPE / "vlm_slope_candidate_review.policy-correction.json"
        receipt_path = SLOPE / "vlm_slope_candidate_review.policy-correction.receipt.json"
        event = json.loads(event_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(list(self.validator("slope-policy-correction-event.schema.json").iter_errors(event)), [])
        self.assertEqual(list(self.validator("slope-policy-correction-receipt.schema.json").iter_errors(receipt)), [])
        self.assertEqual(digest(old), event["supersedes"]["sha256"])
        self.assertEqual(digest(old), receipt["superseded_artifact_readback"]["actual_sha256"])
        self.assertEqual(digest(event_path), receipt["event"]["sha256"])
        correction = event["correction"]
        self.assertEqual(correction["traversability_status"], "passable")
        self.assertFalse(correction["human_pixel_truth_available"])
        self.assertFalse(correction["approved_for_training"])

    def test_fixed_holdout_chain_hashes_and_human_gate_are_read_back(self):
        if not HOLDOUT.exists():
            self.skipTest("workspace fixed-holdout evidence is not present in this checkout")
        readback = json.loads((REPO / "docs/test-results/fixed-holdout-chain-readback-20261111.json").read_text(encoding="utf-8"))
        for item in readback["steps"]:
            self.assertEqual(digest(HOLDOUT / item["artifact"]), item["sha256"])
        result = readback["result"]
        self.assertEqual(result["agent_action"], "no_op")
        self.assertEqual(result["second_vlm_action"], "no_op")
        self.assertEqual(result["workflow_state"], "human_review_required")
        self.assertFalse(result["human_pixel_truth_available"])
        self.assertFalse(result["truth_committed"])
        forbidden = [path for path in HOLDOUT.rglob("*") if path.is_file()
                     and any(token in path.name for token in
                             ("human_accepted", "truth_committed", "version_committed", "training_ready"))]
        self.assertEqual(forbidden, [])

    def test_training_preflight_schema_and_source_hashes(self):
        path = REPO / "docs/test-results/sam3-peft-preflight-20261111.json"
        preflight = json.loads(path.read_text(encoding="utf-8"))
        errors = list(self.validator("sam3-peft-preflight.schema.json").iter_errors(preflight))
        self.assertEqual(errors, [], [error.message for error in errors])
        for item in preflight["source_support"]:
            self.assertEqual(digest(REPO / item["path"]), item["sha256"])
        self.assertEqual(preflight["module_plan"]["trainable_prefix"],
                         "segmentation_head.mask_predictor.mask_embed.")
        self.assertFalse(preflight["formal_training"]["authorized"])
        self.assertFalse(preflight["formal_training"]["started"])
        self.assertFalse(preflight["ddp_check"]["process_started"])

    def test_training_plan_is_schema_valid_and_cannot_launch(self):
        path = REPO / "configs/sam3_sfl_cpd15y_peft_plan.json"
        plan = json.loads(path.read_text(encoding="utf-8"))
        errors = list(self.validator("sam3-peft-plan.schema.json").iter_errors(plan))
        self.assertEqual(errors, [], [error.message for error in errors])
        self.assertFalse(plan["execution_enabled"])
        self.assertFalse(plan["launch_gate"]["formal_training_authorized"])
        self.assertEqual(plan["model"]["trainable_prefixes"],
                         ["segmentation_head.mask_predictor.mask_embed."])
        self.assertIn("train/_preannotations.coco.json",
                      plan["data"]["forbidden_as_truth"])
        self.assertFalse(plan["data"]["test24_in_training"])
        self.assertFalse(plan["data"]["fixed_640_holdout_in_training"])


if __name__ == "__main__":
    unittest.main()
