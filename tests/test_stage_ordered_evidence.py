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
        self.assertTrue(plan["base_config_requires_freeze_override"])
        self.assertFalse(plan["launch_gate"]["formal_training_authorized"])
        self.assertEqual(plan["strategy"]["selected_route"],
                         "supervised_parameter_efficient_finetuning")
        self.assertEqual(plan["strategy"]["post_training_taxonomy"],
                         "peft_is_a_specific_supervised_post_training_subtype")
        self.assertEqual(plan["strategy"]["rejected_route"],
                         "ambiguous_or_full_model_post_training")
        self.assertEqual(plan["model"]["verified_trainable_scope"]["prefixes"],
                         ["segmentation_head.mask_predictor.mask_embed."])
        self.assertTrue(plan["model"]["stage0_is_minimal_baseline"])
        self.assertFalse(plan["model"]["stage0_is_final_upper_bound"])
        self.assertFalse(plan["model"]["conditional_expansion"]["enabled"])
        self.assertEqual(plan["model"]["conditional_expansion"][
            "predeclared_trainable_prefixes"], [])
        self.assertEqual(plan["model"]["conditional_expansion"][
            "predeclared_module_families"], [])
        self.assertFalse(plan["model"]["unverified_module_claims"][
            "mask_decoder_trainable"])
        self.assertFalse(plan["model"]["unverified_module_claims"][
            "adapter_exists_or_trainable"])
        self.assertFalse(plan["model"]["unverified_module_claims"][
            "lora_exists_or_trainable"])
        self.assertEqual(plan["staged_unfreeze_config"],
                         "configs/sam3_sfl_cpd15y_staged_unfreeze.json")
        self.assertIn("train/_preannotations.coco.json",
                      plan["data"]["forbidden_as_truth"])
        self.assertFalse(plan["data"]["test24_in_training"])
        self.assertFalse(plan["data"]["fixed_640_holdout_in_training"])
        self.assertEqual(plan["data"]["current_96_pack_classes"],
                         ["pallet", "person"])
        self.assertFalse(plan["data"]["slope_improvement_claim_allowed"])
        self.assertEqual(plan["data"]["training_labels"], ["floor", "slope"])
        self.assertIn("training_metrics",
                      plan["launch_gate"]["without_human_truth_forbidden"])

    def test_conditional_unfreeze_is_stage0_only_until_live_gates_pass(self):
        path = REPO / "configs/sam3_sfl_cpd15y_staged_unfreeze.json"
        plan = json.loads(path.read_text(encoding="utf-8"))
        errors = list(self.validator("sam3-staged-unfreeze.schema.json").iter_errors(plan))
        self.assertEqual(errors, [], [error.message for error in errors])
        current = plan["current_verified_stage"]
        self.assertEqual(current["trainable_prefixes"],
                         ["segmentation_head.mask_predictor.mask_embed."])
        self.assertEqual(current["runtime_exact_parameter_names_status"],
                         "verified_dual_5090_inventory_receipt")
        self.assertTrue(plan["runtime_inventory_gate"][
            "dual_5090_named_parameters_verified"])
        self.assertEqual(
            plan["runtime_inventory_gate"]["inventory_receipt"],
            "docs/test-results/dual-gpu-infrastructure-live-readback-20261111.json")
        self.assertFalse(plan["runtime_inventory_gate"][
            "genuine_human_truth_version_available"])
        self.assertFalse(plan["runtime_inventory_gate"][
            "fixed_holdout_baseline_available"])
        expansion = plan["conditional_expansion"]
        self.assertFalse(expansion["enabled"])
        self.assertEqual(expansion["predeclared_stages"], [])
        self.assertEqual(expansion["predeclared_trainable_prefixes"], [])
        self.assertEqual(expansion["predeclared_target_families"], [])
        self.assertIn("exact_trainable_parameter_names",
                      expansion["required_stage_record_fields"])
        self.assertIn("trainable_parameter_count",
                      expansion["required_stage_record_fields"])
        self.assertIn("training_config_sha256",
                      expansion["required_stage_record_fields"])
        self.assertIn("independent_holdout_gain",
                      expansion["required_stage_record_fields"])
        self.assertIn("rollback_checkpoint_sha256",
                      expansion["required_stage_record_fields"])
        self.assertEqual(plan["holdout_gate"]["required_metrics"],
                         ["pixel_recall", "mask_iou", "boundary_f1"])
        self.assertEqual(plan["holdout_gate"]["threshold_status"],
                         "pending_human_truth_calibration")
        self.assertFalse(plan["unverified_module_policy"][
            "mask_decoder_exists_or_is_verified_trainable"])
        self.assertFalse(plan["unverified_module_policy"][
            "adapter_exists_or_is_verified_trainable"])
        self.assertFalse(plan["unverified_module_policy"][
            "lora_exists_or_is_verified_trainable"])
        self.assertFalse(plan["full_model_policy"]["enabled"])

    def test_future_stage_record_requires_parameters_hashes_holdout_and_rollback(self):
        record = {
            "schema_version": "labelagent.sam3-unfreeze-stage-record.v1",
            "stage_id": "example-test-only",
            "scope_type": "stage0_mask_embed",
            "authorization": {
                "dual_5090_named_parameters_verified": True,
                "genuine_human_truth_version_available": True,
                "fixed_holdout_baseline_available": True,
            },
            "exact_trainable_parameter_names": [
                "segmentation_head.mask_predictor.mask_embed.example.weight"
            ],
            "trainable_parameter_count": 1,
            "training_config_sha256": "1" * 64,
            "base_checkpoint_sha256": "2" * 64,
            "independent_holdout": {
                "dataset_version": "test-only",
                "manifest_sha256": "3" * 64,
                "baseline_metrics": {"pixel_recall": 0.1, "mask_iou": 0.1,
                                     "boundary_f1": 0.1},
                "candidate_metrics": {"pixel_recall": 0.2, "mask_iou": 0.2,
                                      "boundary_f1": 0.2},
                "gain": {"pixel_recall": 0.1, "mask_iou": 0.1,
                         "boundary_f1": 0.1},
            },
            "rollback": {
                "checkpoint_path": "test-only/original.pt",
                "checkpoint_sha256": "4" * 64,
            },
            "full_model_training": False,
        }
        validator = self.validator("sam3-unfreeze-stage-record.schema.json")
        self.assertEqual(list(validator.iter_errors(record)), [])
        del record["trainable_parameter_count"]
        self.assertNotEqual(list(validator.iter_errors(record)), [])

    def test_training_strategy_decision_is_strict_and_claim_bounded(self):
        path = REPO / "docs/test-results/sam3-peft-strategy-decision-20261111.json"
        decision = json.loads(path.read_text(encoding="utf-8"))
        errors = list(self.validator(
            "sam3-peft-strategy-decision.schema.json").iter_errors(decision))
        self.assertEqual(errors, [], [error.message for error in errors])
        self.assertEqual(decision["taxonomy"]["relationship_to_post_training"],
                         "specific_subtype")
        self.assertFalse(decision["taxonomy"]["full_model_post_training_selected"])
        self.assertFalse(decision["data_scope"]["machine_preannotations_truth_eligible"])
        self.assertFalse(decision["data_scope"]["slope_improvement_claim_allowed"])
        self.assertEqual(decision["data_scope"]["training_labels"], ["floor", "slope"])
        self.assertFalse(decision["claim_boundary"]["training_started"])
        self.assertFalse(decision["claim_boundary"]["checkpoint_available"])


if __name__ == "__main__":
    unittest.main()
