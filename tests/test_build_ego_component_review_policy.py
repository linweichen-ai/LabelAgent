import copy
import unittest

from tools.build_ego_component_review_policy import build_policy


def fixtures():
    rows = []
    for camera in ("camera1", "camera3"):
        for rank in range(1, 6):
            rows.append({
                "component_id": f"{camera}-{rank}",
                "sample_id": f"sample-{camera}-{rank}",
                "camera": camera,
                "camera_review_priority_rank": rank,
                "review_priority_score": 1.0 / rank,
                "automatic_decision": None,
            })
    ranking = {
        "candidate_only": True,
        "human_truth": False,
        "automatic_decisions_enabled": False,
        "component_count": len(rows),
        "camera_component_counts": {"camera1": 5, "camera3": 5},
        "rows": rows,
    }
    evaluation = {
        "assistant_review_only": True,
        "human_truth": False,
        "reviewer_blinded_to_risk_band": True,
        "hard_reject_enrichment_observed": True,
        "top_plus_middle_likely_reject_capture_rate": 1.0,
    }
    return ranking, evaluation


class ReviewPolicyTests(unittest.TestCase):
    def test_camera1_tail_is_deferred_but_never_auto_accepted(self):
        ranking, evaluation = fixtures()
        payload = build_policy(ranking, evaluation,
                               ranking_sha256="a" * 64,
                               evaluation_sha256="b" * 64)
        self.assertEqual(payload["counts"]["priority_review"], 9)
        self.assertEqual(payload["counts"]["deferred_manual_review"], 1)
        tail = next(row for row in payload["rows"]
                    if row["component_id"] == "camera1-5")
        self.assertEqual(tail["manual_review_queue"], "deferred_manual_review")
        self.assertIsNone(tail["automatic_decision"])
        self.assertFalse(tail["training_eligible"])

    def test_camera3_is_always_priority_review(self):
        ranking, evaluation = fixtures()
        payload = build_policy(ranking, evaluation,
                               ranking_sha256="a" * 64,
                               evaluation_sha256="b" * 64)
        self.assertTrue(all(
            row["manual_review_queue"] == "priority_review"
            for row in payload["rows"] if row["camera"] == "camera3"))

    def test_fails_closed_without_blind_enrichment_gate(self):
        ranking, evaluation = fixtures()
        evaluation["hard_reject_enrichment_observed"] = False
        with self.assertRaisesRegex(ValueError, "enrichment"):
            build_policy(ranking, evaluation,
                         ranking_sha256="a" * 64,
                         evaluation_sha256="b" * 64)

    def test_fails_closed_on_noncontiguous_camera_ranks(self):
        ranking, evaluation = fixtures()
        ranking["rows"][4]["camera_review_priority_rank"] = 6
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            build_policy(ranking, evaluation,
                         ranking_sha256="a" * 64,
                         evaluation_sha256="b" * 64)

    def test_fails_closed_on_automatic_decision(self):
        ranking, evaluation = fixtures()
        ranking["rows"][0]["automatic_decision"] = "accept"
        with self.assertRaisesRegex(ValueError, "automatic decision"):
            build_policy(ranking, evaluation,
                         ranking_sha256="a" * 64,
                         evaluation_sha256="b" * 64)

    def test_fails_closed_on_camera_count_mismatch(self):
        ranking, evaluation = fixtures()
        ranking["camera_component_counts"]["camera1"] = 4
        with self.assertRaisesRegex(ValueError, "counts mismatch"):
            build_policy(ranking, evaluation,
                         ranking_sha256="a" * 64,
                         evaluation_sha256="b" * 64)


if __name__ == "__main__":
    unittest.main()
