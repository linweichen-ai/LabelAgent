import unittest

from tools.evaluate_other_robot_vlm_guard import (
    crop_data_url, expected_evidence, validate_output)


class OtherRobotVlmGuardTests(unittest.TestCase):
    def test_scene_class_maps_to_conservative_evidence(self):
        self.assertEqual(expected_evidence("separate_unmanned_robot"), "present")
        self.assertEqual(expected_evidence("ego_robot_part"), "absent")
        self.assertEqual(expected_evidence("conventional_manned_forklift"), "absent")
        self.assertEqual(expected_evidence("no_robot_evidence"), "absent")
        self.assertEqual(expected_evidence("uncertain"), "uncertain")

    def test_contradictory_guard_output_is_rejected(self):
        value = {"scene_class": "conventional_manned_forklift",
                 "other_robot_evidence": "present", "confidence": 0.9,
                 "reason": "forklift"}
        with self.assertRaises(ValueError):
            validate_output(value)

    def test_valid_guard_output_passes(self):
        value = {"scene_class": "separate_unmanned_robot",
                 "other_robot_evidence": "present", "confidence": 0.9,
                 "reason": "separate robot body"}
        self.assertIs(validate_output(value), value)

    def test_prompt_prioritizes_separate_robot_over_visible_ego(self):
        from tools.evaluate_other_robot_vlm_guard import PROMPT
        self.assertIn("ANY physically separate robot", PROMPT)
        self.assertIn("ego vehicle may be visible", PROMPT)

    def test_candidate_roi_is_bounds_checked(self):
        from PIL import Image
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (20, 10), (1, 2, 3)).save(path)
            self.assertTrue(crop_data_url(path, [0, 0, 10, 10]).startswith(
                "data:image/jpeg;base64,"))
            with self.assertRaises(ValueError):
                crop_data_url(path, [0, 0, 21, 10])


if __name__ == "__main__":
    unittest.main()
