import unittest

from tools.evaluate_ego_component_risk_blind_audit import (
    _summary, wilson_interval)


class EgoComponentRiskBlindEvaluationTests(unittest.TestCase):
    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(5, 10)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

    def test_summary_counts_review_risk(self):
        rows = [
            {"decision": "likely_fill_ok"},
            {"decision": "uncertain"},
            {"decision": "likely_reject"},
        ]
        summary = _summary(rows)
        self.assertEqual(summary["review_risk_count"], 2)
        self.assertEqual(summary["uncertain"], 1)
        self.assertEqual(summary["likely_reject"], 1)

    def test_wilson_rejects_empty_group(self):
        with self.assertRaisesRegex(ValueError, "invalid Wilson"):
            wilson_interval(0, 0)

    def test_summary_keeps_reject_separate_from_uncertain(self):
        rows = [{"decision": "uncertain"},
                {"decision": "likely_reject"}]
        summary = _summary(rows)
        self.assertEqual(summary["review_risk_count"], 2)
        self.assertEqual(summary["likely_reject"], 1)


if __name__ == "__main__":
    unittest.main()
