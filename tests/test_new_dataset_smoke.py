import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.run_new_dataset_smoke import (
    cache_stat_delta,
    grouped_audit_timing,
    model_config,
    read_model_trace,
    summarize_model_trace,
    vlm_health_url,
)


class NewDatasetSmokeTests(unittest.TestCase):
    def test_vlm_health_url_and_monotonic_content_free_delta(self):
        self.assertEqual(vlm_health_url("http://127.0.0.1:28868/v1"),
                         "http://127.0.0.1:28868/health")
        self.assertEqual(cache_stat_delta(
            {"response_cache_stats": {"hits": 4, "misses": 5}},
            {"response_cache_stats": {"hits": 6, "misses": 6,
                                        "evictions": 1}}),
            {"evictions": 1, "hits": 2, "misses": 1})
        with self.assertRaises(RuntimeError):
            cache_stat_delta(
                {"response_cache_stats": {"hits": 4}},
                {"response_cache_stats": {"hits": 3}})

    def test_model_override_is_in_memory_and_rejects_credentials(self):
        source = {"llm_url": "http://old/v1", "llm_model": "old",
                  "sam3_url": "http://sam"}
        updated = model_config(source, "http://127.0.0.1:38868/v1", "new",
                               "http://127.0.0.1:38869/v1", "fast")
        self.assertEqual(updated["llm_url"], "http://127.0.0.1:38868/v1")
        self.assertEqual(updated["llm_model"], "new")
        self.assertEqual(updated["llm_fast_url"], "http://127.0.0.1:38869/v1")
        self.assertEqual(updated["llm_fast_model"], "fast")
        self.assertEqual(source["llm_model"], "old")
        with self.assertRaises(ValueError):
            model_config(source, "http://user:secret@127.0.0.1/v1", None)
        with self.assertRaises(ValueError):
            model_config(source, None, None,
                         "http://user:secret@127.0.0.1/v1", None)

    def test_trace_summary_excludes_prompt_and_model_prose(self):
        trace = [{"service": "GPT", "status": "succeeded", "elapsed_ms": 10,
                  "input": {"instruction": "private prompt", "images": [{}]},
                  "output": {"raw_text": "private model prose",
                             "parsed": {"status": "done"},
                             "model_route": "small_image_fast",
                             "fast_fallback_reason": None,
                             "local_vlm": {"quantization": "nf4"}}}]
        summary = summarize_model_trace(trace)
        self.assertEqual(summary[0]["raw_text_chars"], 19)
        self.assertNotIn("instruction", summary[0])
        self.assertNotIn("raw_text", summary[0])
        self.assertEqual(summary[0]["model_route"], "small_image_fast")

    def test_read_model_trace_adds_stage_without_exposing_content(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model_debug.json"
            path.write_text(json.dumps({"trace": [{
                "service": "GPT", "status": "succeeded", "elapsed_ms": 12,
                "input": {"instruction": "private prompt", "images": [{}]},
                "output": {"raw_text": "private prose", "parsed": {"ok": True}},
            }]}), encoding="utf-8")
            rows = read_model_trace(path, stage="background_agent_refinement")
        self.assertEqual(rows[0]["pipeline_stage"],
                         "background_agent_refinement")
        self.assertNotIn("instruction", rows[0])
        self.assertNotIn("raw_text", rows[0])

    def test_grouped_audit_timing_is_content_free_and_counts_requests(self):
        row = grouped_audit_timing({"agent": {"candidate_class_audits": [
            {"execution_mode": "sequential", "decision": {"reason": "private"}},
            {"execution_mode": "sequential", "decision": {"reason": "private"}},
        ]}}, 123.5)
        self.assertEqual(row["request_count"], 2)
        self.assertEqual(row["elapsed_ms"], 123.5)
        self.assertEqual(row["execution_modes"], ["sequential"])
        self.assertNotIn("decision", row)


if __name__ == "__main__":
    unittest.main()
