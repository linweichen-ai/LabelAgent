import unittest

from tools.evaluate_qwen3vl_lora_generation import score_output, summarize


class Qwen3VlLoraGenerationTests(unittest.TestCase):
    def test_evidence_scoring_is_label_exact(self):
        row = {"target": '{"assessments":{"floor":{"evidence":"present"},'
                         '"slope":{"evidence":"absent"}},"unknown_candidates":[]}'}
        parsed = {"assessments": {
            "floor": {"evidence": "present"}, "slope": {"evidence": "present"}},
            "unknown_candidates": []}
        score = score_output(row, parsed)
        self.assertEqual(score["matched_labels"], 1)
        self.assertEqual(score["label_count"], 2)
        self.assertEqual(score["evidence_pairs"]["slope"], {
            "expected": "absent", "actual": "present"})

    def test_registry_drift_is_rejected(self):
        row = {"target": '{"assessments":{"floor":{"evidence":"present"}},'
                         '"unknown_candidates":[]}'}
        with self.assertRaises(ValueError):
            score_output(row, {"assessments": {"ground": {"evidence": "present"}},
                               "unknown_candidates": []})

    def test_summary_counts_only_valid_outputs(self):
        rows = [
            {"status": "ok", "generation_seconds": 1.0, "completion_tokens": 4,
             "score": {"label_count": 2, "matched_labels": 1,
                       "unknown_count_matches": True}},
            {"status": "failed"},
        ]
        summary = summarize(rows)
        self.assertEqual(summary["one_pass_success_rate"], 0.5)
        self.assertEqual(summary["teacher_evidence_accuracy"], 0.5)

    def test_summary_reports_present_confusion(self):
        pairs = [
            ("present", "present"), ("absent", "present"),
            ("present", "absent"), ("absent", "absent"),
        ]
        rows = [{
            "status": "ok", "generation_seconds": 1.0, "completion_tokens": 4,
            "score": {
                "label_count": 1,
                "matched_labels": int(expected == actual),
                "unknown_count_matches": True,
                "evidence_pairs": {
                    "other_robot": {"expected": expected, "actual": actual}},
            },
        } for expected, actual in pairs]
        metrics = summarize(rows)["per_label"]["other_robot"]
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
