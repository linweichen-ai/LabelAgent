import unittest

from tools.evaluate_slope_vlm_point_cascade import (
    POINT_INSTRUCTION, summarize, validate_decision,
)


class SlopeVlmPointCascadeTests(unittest.TestCase):
    def test_prompt_does_not_depend_on_ramp_color(self):
        self.assertNotIn("yellow", POINT_INSTRUCTION.lower())
        self.assertNotIn("orange", POINT_INSTRUCTION.lower())

    def test_decision_coordinate_contract(self):
        value = validate_decision({"slope_visible": True, "point_x": 100,
                                   "point_y": 200, "confidence": 0.8,
                                   "reason": "visible inclined surface"})
        self.assertTrue(value["slope_visible"])
        with self.assertRaises(ValueError):
            validate_decision({"slope_visible": False, "point_x": 100,
                               "point_y": 200, "confidence": 0.8,
                               "reason": "no ramp"})

    def test_summary_exposes_false_positives(self):
        rows = []
        for index in range(8):
            rows.append({
                "assistant_visible_slope_candidate": index < 4,
                "vlm_decision": {"slope_visible": index in {0, 1, 4}},
                "sam_candidate": None, "sam_error": None,
                "vlm_elapsed_seconds": 1.0, "sam_elapsed_seconds": 0.0,
            })
        result = summarize(rows)
        self.assertEqual(result["true_positive_images"], 2)
        self.assertEqual(result["false_positive_images"], 1)
        self.assertEqual(result["false_negative_images"], 2)


if __name__ == "__main__":
    unittest.main()
