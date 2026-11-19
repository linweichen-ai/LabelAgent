import unittest

from tools.evaluate_other_robot_prompt_fallback import candidate_geometry, sam_prompts


def rectangle(left, top, right, bottom):
    return {"id": 1, "label": "other_robot", "points": [
        [left, top], [right, top], [right, bottom], [left, bottom]
    ]}


class OtherRobotPromptFallbackTests(unittest.TestCase):
    def test_sam_contract_compares_baseline_and_fallback(self):
        self.assertEqual(sam_prompts("robotic pallet truck"), {
            "baseline": "robot", "fallback": "robotic pallet truck"})
        self.assertEqual(sam_prompts("autonomous pallet truck")["fallback"],
                         "autonomous pallet truck")

    def test_large_bottom_candidate_is_warning_not_rejection(self):
        result = candidate_geometry(rectangle(0, 1200, 1600, 1535), 1920, 1536)
        self.assertTrue(result["touches_bottom"])
        self.assertTrue(result["bottom_large_ego_risk"])
        self.assertTrue(result["warning_only"])

    def test_small_distinct_robot_does_not_trigger_ego_warning(self):
        result = candidate_geometry(rectangle(600, 480, 770, 610), 1920, 1536)
        self.assertFalse(result["touches_bottom"])
        self.assertFalse(result["bottom_large_ego_risk"])


if __name__ == "__main__":
    unittest.main()
