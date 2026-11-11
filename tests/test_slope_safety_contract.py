import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BACKLOG = ROOT / "docs" / "eval" / "slope-hard-set-backlog.json"

EXPECTED_STRATA = {"near", "mid", "far", "weak_texture", "shadow", "backlight",
                    "occlusion", "slope_foot_boundary", "slope_top_boundary"}
EXPECTED_METRICS = {"slope_iou", "slope_dice", "slope_boundary_f1",
    "slope_boundary_tolerance_px", "slope_pixel_precision", "slope_pixel_recall",
    "slope_instance_precision", "slope_instance_recall",
    "slope_to_floor_confusion_pixels", "floor_to_slope_confusion_pixels",
    "slope_to_step_curb_confusion_pixels", "step_curb_to_slope_confusion_pixels",
    "slope_false_traversable_count"}


class SlopeSafetyContractTests(unittest.TestCase):
    def setUp(self):
        self.backlog = json.loads(BACKLOG.read_text(encoding="utf-8"))

    def test_hard_set_and_single_class_metrics_are_complete(self):
        self.assertEqual({item["name"] for item in self.backlog["hard_set_strata"]},
                         EXPECTED_STRATA)
        self.assertEqual(set(self.backlog["required_metrics"]), EXPECTED_METRICS)
        self.assertEqual(set(self.backlog["current_metrics"]), EXPECTED_METRICS)

    def test_unverified_real_candidate_keeps_truth_gate_but_uses_passable_policy(self):
        boundary = self.backlog["claim_boundary"]
        self.assertEqual(boundary["slope_visible"], "candidate_supported")
        self.assertFalse(boundary["human_pixel_truth_available"])
        self.assertFalse(boundary["approved_for_training"])
        self.assertFalse(boundary["approved_for_deployment"])
        self.assertEqual(boundary["business_mapping"], "traversable_area")
        mapping = boundary["downstream_mapping"]
        self.assertEqual(mapping["target_label"], "traversable_area")
        self.assertEqual(mapping["operation"], "union")
        self.assertEqual(mapping["source_labels"], ["floor", "slope"])
        self.assertTrue(mapping["preserve_source_labels"])
        self.assertEqual(boundary["traversability_status"], "passable")
        self.assertEqual(boundary["policy_default"], "passable")
        self.assertEqual(boundary["policy_basis"], "user_defined_project_semantics")
        self.assertEqual(boundary["policy_override"]["status"], "none")
        self.assertTrue(all(value is None
                            for value in self.backlog["current_metrics"].values()))

    def test_non_passable_override_is_explicit_and_never_model_inferred(self):
        policy = self.backlog["non_passable_override_policy"]
        self.assertEqual(set(policy["allowed_deciders"]), {"human", "system_policy"})
        self.assertFalse(policy["model_inference_allowed"])
        self.assertFalse(policy["geometry_required"])
        self.assertTrue(policy["requires_rule_id"])
        self.assertTrue(policy["requires_reason"])


if __name__ == "__main__":
    unittest.main()
