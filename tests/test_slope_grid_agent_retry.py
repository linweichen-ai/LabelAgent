import unittest

from tools.evaluate_slope_grid_agent_retry import (
    SIBLING_INSTRUCTION, VERIFY_INSTRUCTION, frame_group, summarize,
    validate_verification,
)


class SlopeGridAgentRetryTests(unittest.TestCase):
    def test_prompts_do_not_depend_on_ramp_color(self):
        for prompt in (SIBLING_INSTRUCTION, VERIFY_INSTRUCTION):
            self.assertNotIn("yellow", prompt.lower())
            self.assertNotIn("orange", prompt.lower())

    def test_verification_contract(self):
        accepted = validate_verification({
            "assessment": "accept_slope", "alternate_cell_id": "NONE",
            "confidence": 0.9, "reason": "mask follows inclined surface"})
        self.assertEqual(accepted["assessment"], "accept_slope")
        rejected = validate_verification({
            "assessment": "reject_not_slope", "alternate_cell_id": "D4",
            "confidence": 0.9, "reason": "mask follows level floor"})
        self.assertEqual(rejected["alternate_cell_id"], "D4")
        with self.assertRaises(ValueError):
            validate_verification({
                "assessment": "accept_slope", "alternate_cell_id": "D4",
                "confidence": 0.9, "reason": "invalid alternate"})

    def test_frame_group_and_summary(self):
        self.assertEqual(frame_group("008_123_camera_2"), "008_123")
        rows = [{"assistant_visible_slope_candidate": index < 4,
                 "final_candidate_accepted": index in {0, 1, 4},
                 "retry_mask_generated": False,
                 "vlm_elapsed_seconds": 1.0, "sam_elapsed_seconds": 0.0}
                for index in range(8)]
        result = summarize(rows)
        self.assertEqual(result["true_positive_images"], 2)
        self.assertEqual(result["false_positive_images"], 1)


if __name__ == "__main__":
    unittest.main()
