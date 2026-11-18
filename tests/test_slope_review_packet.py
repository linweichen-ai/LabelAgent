import unittest

from tools.build_slope_review_packet import candidate_row


class SlopeReviewPacketTests(unittest.TestCase):
    def test_rows_are_always_candidate_only_and_training_disabled(self):
        row = candidate_row(
            frame=7, cohort="positive_proxy", source_frame_sha256="a" * 64,
            raw_path="raw/7.jpg", review_path="review/7.jpg", route="full_frame",
            candidate_count=1, target_provenance="legacy_auto_preannotation")
        self.assertTrue(row["candidate_only"])
        self.assertFalse(row["human_truth"])
        self.assertFalse(row["approved_for_training"])
        self.assertFalse(row["training_return_allowed"])
        self.assertIsNone(row["human_decision"])
        self.assertEqual(row["review_status"], "pending_human_review")

    def test_candidate_count_and_frame_are_normalized_to_integer(self):
        row = candidate_row(
            frame="8", cohort="negative_cohort", source_frame_sha256="b" * 64,
            raw_path="raw/8.jpg", review_path="review/8.jpg", route="production_path",
            candidate_count="0", target_provenance="unverified_saved_annotations")
        self.assertEqual(row["frame"], 8)
        self.assertEqual(row["candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
