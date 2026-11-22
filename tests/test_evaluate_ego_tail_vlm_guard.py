import unittest

from tools.evaluate_ego_tail_vlm_guard import (
    select_camera1_tail, validate_vlm_decision)


class EgoTailVlmGuardTests(unittest.TestCase):
    def test_validates_conservative_fill_ok(self):
        value = validate_vlm_decision({
            "decision": "likely_fill_ok",
            "visual_evidence_clear": True,
            "same_solid_surface": True,
            "crosses_real_opening_or_background": False,
            "confidence": 0.9,
            "reason": "solid surface",
        })
        self.assertEqual(value["decision"], "likely_fill_ok")

    def test_rejects_unsafe_fill_ok_combination(self):
        with self.assertRaisesRegex(ValueError, "unsafe"):
            validate_vlm_decision({
                "decision": "likely_fill_ok",
                "visual_evidence_clear": False,
                "same_solid_surface": True,
                "crosses_real_opening_or_background": False,
                "confidence": 0.9,
                "reason": "unclear",
            })

    def test_reject_requires_opening_or_background_evidence(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_vlm_decision({
                "decision": "likely_reject",
                "visual_evidence_clear": True,
                "same_solid_surface": False,
                "crosses_real_opening_or_background": False,
                "confidence": 0.9,
                "reason": "no support",
            })

    def test_selects_complete_camera1_tail(self):
        policy = {
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False, "production_deployed": False,
            "default_enabled": False, "automatic_decisions_enabled": False,
            "policy": {"camera1_bottom_tail_evidence_coverage": "complete",
                       "camera1_defer_after_rank": 2},
            "counts": {"camera1_bottom_total": 1},
            "rows": [{
                "component_id": "c3", "camera": "camera1",
                "camera_review_priority_rank": 3,
                "manual_review_queue": "deferred_manual_review",
                "policy_reason": "reviewed",
            }],
        }
        review = {
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False, "production_deployed": False,
            "default_enabled": False,
            "rows": [{"component_id": "c3", "review_image": "c3.jpg",
                      "review_image_sha256": "a" * 64}],
        }
        selected = select_camera1_tail(policy, review)
        self.assertEqual([row["component_id"] for row in selected], ["c3"])

    def test_selection_fails_closed_on_incomplete_tail(self):
        policy = {
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False, "production_deployed": False,
            "default_enabled": False, "automatic_decisions_enabled": False,
            "policy": {"camera1_bottom_tail_evidence_coverage": "complete",
                       "camera1_defer_after_rank": 2},
            "counts": {"camera1_bottom_total": 2},
            "rows": [{
                "component_id": "c3", "camera": "camera1",
                "camera_review_priority_rank": 3,
                "manual_review_queue": "deferred_manual_review",
                "policy_reason": "reviewed",
            }],
        }
        review = {
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False, "production_deployed": False,
            "default_enabled": False,
            "rows": [{"component_id": "c3", "review_image": "c3.jpg",
                      "review_image_sha256": "a" * 64}],
        }
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            select_camera1_tail(policy, review)


if __name__ == "__main__":
    unittest.main()
