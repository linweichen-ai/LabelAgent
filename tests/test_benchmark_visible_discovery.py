import unittest

from tools.benchmark_visible_discovery import (
    bounded_assessments, bounded_consistency_flags, bounded_unknown, median,
    request_metadata)


class VisibleDiscoveryBenchmarkTests(unittest.TestCase):
    def test_bounded_semantics_exclude_model_reasons(self):
        result = {"assessments": [{"label": "person", "evidence": "present",
            "confidence": 0.9, "reason": "bounded prose"}],
            "unknown_candidates": [{"candidate_name": "rack", "confidence": 0.7,
                "route": "human_review_required", "evidence": "bounded prose"}]}
        self.assertEqual(bounded_assessments(result), [{"label": "person",
            "evidence": "present", "confidence": 0.9}])
        self.assertEqual(bounded_unknown(result), [{"candidate_name": "rack",
            "confidence": 0.7, "route": "human_review_required"}])

    def test_request_metadata_uses_last_successful_gpt_call(self):
        trace = [{"service": "GPT", "status": "succeeded", "elapsed_ms": 10,
                  "output": {"raw_text": "{}", "local_vlm": {"quantization": "none"}}},
                 {"service": "SAM 3", "status": "succeeded", "elapsed_ms": 1,
                  "output": {}}]
        self.assertEqual(request_metadata(trace), {"elapsed_ms": 10,
            "raw_text_chars": 2, "local_vlm": {"quantization": "none"}})

    def test_consistency_flags_exclude_model_prose(self):
        result = {"consistency_flags": [{"label": "车辆",
            "normalized_evidence": "uncertain",
            "reason_code": "vlm_cross_label_entity_conflict",
            "route": "human_review_required",
            "conflicting_labels": ["车辆", "other_robot"],
            "shared_anchor": [700, 450], "reason": "long prose"}]}
        self.assertEqual(bounded_consistency_flags(result), [{
            "label": "车辆", "normalized_evidence": "uncertain",
            "reason_code": "vlm_cross_label_entity_conflict",
            "route": "human_review_required",
            "conflicting_labels": ["车辆", "other_robot"],
            "shared_anchor": [700, 450]}])

    def test_median_averages_even_middle_pair(self):
        self.assertEqual(median([4, 1, 3, 2]), 2.5)
        self.assertEqual(median([3, 1, 2]), 2.0)
        self.assertIsNone(median([]))


if __name__ == "__main__":
    unittest.main()
