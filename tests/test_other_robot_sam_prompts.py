import unittest

from tools.diagnose_other_robot_sam_prompts import PROMPTS


class OtherRobotSamPromptDiagnosticTests(unittest.TestCase):
    def test_prompt_set_is_distinct_and_color_independent(self):
        self.assertEqual(PROMPTS["baseline_robot"], "robot")
        self.assertGreaterEqual(len(PROMPTS), 6)
        self.assertEqual(len(PROMPTS), len(set(PROMPTS.values())))
        forbidden = {"red", "green", "blue", "yellow", "orange", "black",
                     "white", "gray", "grey"}
        for prompt in PROMPTS.values():
            self.assertTrue(forbidden.isdisjoint(prompt.casefold().split()))


if __name__ == "__main__":
    unittest.main()
