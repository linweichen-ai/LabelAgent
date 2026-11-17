import unittest

from tools.diagnose_slope_sam_prompts import PROMPTS, summarize


class SlopeSamPromptDiagnosticTests(unittest.TestCase):
    def test_prompts_are_color_independent(self):
        self.assertGreaterEqual(len(PROMPTS), 2)
        for prompt in PROMPTS.values():
            self.assertNotIn("yellow", prompt.lower())
            self.assertNotIn("orange", prompt.lower())

    def test_summary_keeps_false_positives_visible(self):
        rows = []
        for index in range(8):
            rows.append({
                "assistant_visible_slope_candidate": index < 4,
                "prompts": {
                    key: {"candidate_count": int(index in {0, 1, 4})}
                    for key in PROMPTS
                },
            })
        summary = summarize(rows)
        for result in summary.values():
            self.assertEqual(result["true_positive_images"], 2)
            self.assertEqual(result["false_positive_images"], 1)
            self.assertEqual(result["false_negative_images"], 2)
            self.assertEqual(result["true_negative_images"], 3)


if __name__ == "__main__":
    unittest.main()
