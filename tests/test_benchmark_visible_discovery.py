import unittest
from pathlib import Path

from tools.benchmark_visible_discovery import (
    bounded_assessments, bounded_consistency_flags, bounded_image_name,
    bounded_unknown, median, model_config, progress_payload, request_metadata)


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
            "model_route": None, "fast_fallback_reason": None,
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

    def test_progress_payload_reports_incremental_eta_without_paths(self):
        rows = [{"failure": None, "model_route": "small_image_fast"},
                {"failure": {"type": "TimeoutError"}, "model_route": None}]
        self.assertEqual(progress_payload(2, 5, rows, 20.0, "frame-2.png"), {
            "event": "progress", "processed": 2, "total": 5,
            "current": "frame-2.png", "success_count": 1,
            "failure_count": 1, "model_route": None,
            "elapsed_seconds": 20.0, "eta_seconds": 30.0})

    def test_bounded_image_name_disambiguates_sequence_and_camera(self):
        path = Path("dataset/010/bevannotation_data/cameras_640x512/camera_3/frame.png")
        self.assertEqual(bounded_image_name(path), "010/camera_3/frame.png")
        self.assertEqual(bounded_image_name(Path("standalone.png")),
                         "standalone.png")

    def test_model_override_is_in_memory_and_rejects_credentials(self):
        source = {"llm_url": "http://old/v1", "llm_model": "old"}
        updated = model_config(source, "http://127.0.0.1:38868/v1", "new",
                               "http://127.0.0.1:38869/v1", "fast")
        self.assertEqual(updated["llm_url"], "http://127.0.0.1:38868/v1")
        self.assertEqual(updated["llm_model"], "new")
        self.assertEqual(updated["llm_fast_url"], "http://127.0.0.1:38869/v1")
        self.assertEqual(updated["llm_fast_model"], "fast")
        self.assertEqual(source, {"llm_url": "http://old/v1", "llm_model": "old"})
        with self.assertRaises(ValueError):
            model_config(source, "http://user:secret@127.0.0.1/v1", None)
        with self.assertRaises(ValueError):
            model_config(source, None, None,
                         "http://user:secret@127.0.0.1/v1", None)

    def test_primary_only_benchmark_strips_fast_route_in_memory(self):
        source = {"llm_url": "http://primary/v1", "llm_model": "primary",
                  "llm_fast_url": "http://fast/v1", "llm_fast_model": "fast",
                  "llm_fast_key": "private", "llm_fast_timeout_seconds": 30}
        selected = model_config(source, disable_fast_route=True)
        self.assertEqual(selected, {"llm_url": "http://primary/v1",
                                    "llm_model": "primary"})
        self.assertIn("llm_fast_url", source)
        self.assertIn("llm_fast_key", source)


if __name__ == "__main__":
    unittest.main()
