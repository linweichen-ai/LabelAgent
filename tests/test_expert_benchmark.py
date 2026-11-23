"""Tests for strict human-referenced external expert VLM A/B reports."""
import copy
import unittest

from source.agent.expert_benchmark import build_expert_ab_report


def payload():
    return {
        "schema_version": "labelagent.external-expert-ab-input.v1",
        "partition": "dev_difficult_v1", "test24": False,
        "synthetic_fixture": False,
        "human_reference": True, "reviewer_confirmed": True,
        "samples": [
            {"sample_id": "s1",
             "candidate_verdicts": {"c1": "supported", "c2": "boundary_issue"},
             "missing_labels": ["person"], "new_label_names": ["rack"]},
            {"sample_id": "s2",
             "candidate_verdicts": {"c3": "instance_merge"},
             "missing_labels": [], "new_label_names": []},
        ],
        "runs": [
            {"model": "model-a", "sample_id": "s1", "status": "passed",
             "latency_ms": 100, "estimated_cost_usd": 0.01,
             "candidate_only": True, "human_truth": False,
             "approved_for_training": False,
             "review": {"candidate_reviews": [
                 {"candidate_id": "c1", "verdict": "supported"},
                 {"candidate_id": "c2", "verdict": "uncertain"}],
                 "missing_labels": ["person"],
                 "new_label_candidates": [{"name": "rack"}]}},
            {"model": "model-a", "sample_id": "s2", "status": "fallback",
             "latency_ms": 300, "estimated_cost_usd": 0,
             "candidate_only": True, "human_truth": False,
             "approved_for_training": False},
            {"model": "model-b", "sample_id": "s1", "status": "passed",
             "latency_ms": 200, "estimated_cost_usd": 0.02,
             "candidate_only": True, "human_truth": False,
             "approved_for_training": False,
             "review": {"candidate_reviews": [
                 {"candidate_id": "c1", "verdict": "supported"},
                 {"candidate_id": "c2", "verdict": "boundary_issue"}],
                 "missing_labels": [], "new_label_candidates": []}},
        ],
    }


class ExpertBenchmarkTests(unittest.TestCase):
    def test_report_separates_quality_operations_cost_and_boundaries(self):
        report = build_expert_ab_report(payload())
        self.assertTrue(report["quality_metrics_available"])
        self.assertTrue(report["real_world_quality_evidence"])
        self.assertEqual(report["reference_mode"], "reviewer_confirmed_human")
        self.assertFalse(report["test24_used"])
        self.assertFalse(report["automatic_winner_selected"])
        self.assertEqual(report["deployment_decision"], "not_evaluated")
        self.assertFalse(report["human_truth_written"])
        a, b = report["models"]
        self.assertEqual(a["model"], "model-a")
        self.assertEqual(a["structured_review_success_rate"], 0.5)
        self.assertEqual(a["candidate_verdict_exact_match"], 0.5)
        self.assertEqual(a["missing_label_precision"], 1.0)
        self.assertEqual(a["new_label_name_exact_recall"], 1.0)
        self.assertEqual(a["latency_ms_median"], 200.0)
        self.assertEqual(a["latency_ms_p95"], 300.0)
        self.assertEqual(b["candidate_verdict_exact_match"], 1.0)
        self.assertIsNone(b["missing_label_precision"])
        self.assertEqual(b["missing_label_recall"], 0.0)

    def test_rejects_test24_and_unconfirmed_or_missing_human_reference(self):
        for mutate in (
                lambda item: item.update(partition="test24"),
                lambda item: item.update(test24=True),
                lambda item: item.update(human_reference=False),
                lambda item: item.update(reviewer_confirmed=False)):
            item = payload(); mutate(item)
            with self.assertRaises(ValueError):
                build_expert_ab_report(item)

    def test_synthetic_fixture_cannot_impersonate_human_reference(self):
        item = payload()
        item.update(synthetic_fixture=True, human_reference=False,
                    reviewer_confirmed=False)
        report = build_expert_ab_report(item)
        self.assertEqual(report["reference_mode"], "synthetic_fixture")
        self.assertFalse(report["human_reference"])
        self.assertFalse(report["reviewer_confirmed"])
        self.assertFalse(report["real_world_quality_evidence"])
        item["human_reference"] = True
        with self.assertRaisesRegex(ValueError, "cannot claim"):
            build_expert_ab_report(item)

    def test_rejects_candidate_boundary_violations(self):
        for field, value in (("candidate_only", False), ("human_truth", True),
                             ("approved_for_training", True)):
            item = payload(); item["runs"][0][field] = value
            with self.assertRaisesRegex(ValueError, "candidate-only"):
                build_expert_ab_report(item)

    def test_rejects_duplicate_runs_and_candidate_binding_drift(self):
        duplicate = payload(); duplicate["runs"].append(copy.deepcopy(duplicate["runs"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_expert_ab_report(duplicate)
        drift = payload()
        drift["runs"][0]["review"]["candidate_reviews"][0]["candidate_id"] = "outside"
        with self.assertRaisesRegex(ValueError, "binding"):
            build_expert_ab_report(drift)

    def test_failed_or_fallback_runs_cannot_smuggle_reviews(self):
        item = payload()
        item["runs"][1]["review"] = copy.deepcopy(item["runs"][0]["review"])
        with self.assertRaisesRegex(ValueError, "cannot contain"):
            build_expert_ab_report(item)

    def test_quality_metrics_are_not_inflated_by_failed_samples(self):
        item = payload()
        item["runs"].append({
            "model": "model-b", "sample_id": "s2", "status": "failed",
            "latency_ms": 500, "estimated_cost_usd": 0.03,
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False})
        report = build_expert_ab_report(item)
        b = report["models"][1]
        self.assertEqual(b["structured_review_success_rate"], 0.5)
        self.assertEqual(b["candidate_verdict_total"], 2)
        self.assertEqual(b["failed_runs"], 1)
        self.assertEqual(b["estimated_cost_usd_total"], 0.05)


if __name__ == "__main__":
    unittest.main()
