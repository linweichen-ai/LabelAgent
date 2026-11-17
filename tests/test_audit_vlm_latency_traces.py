import json
import tempfile
import unittest
from pathlib import Path

from tools.audit_vlm_latency_traces import audit_paths, bounded_record, group_rows, summary


def gpt_record(call_id="call-1", generation_seconds=12.0, cache_hit=False):
    return {
        "id": call_id,
        "service": "GPT",
        "status": "succeeded",
        "purpose": "后台 Agent 候选精修",
        "frame": 7,
        "review_attempt": 1,
        "elapsed_ms": 12500,
        "input": {
            "max_tokens": 256,
            "images": ["data:image/png;base64,SECRET_IMAGE"],
            "prompt": "SECRET_PROMPT",
            "api_key": "SECRET_KEY",
        },
        "output": {
            "finish_reason": "stop",
            "model_route": "primary",
            "raw_text": "SECRET_RAW_TEXT",
            "parsed": {
                "status": "continue",
                "operations": [{"type": "fill_gap"}],
                "hole_assessment": [{"hole_id": 1, "semantic": "ground"}],
            },
            "local_vlm": {
                "cache_hit": cache_hit,
                "coalesced": False,
                "image_count": 1,
                "actual_image_token_counts": [238],
                "inference_stats": {
                    "generation_seconds": generation_seconds,
                    "preprocess_seconds": 0.2,
                    "constraint_setup_seconds": 0.01,
                    "prompt_tokens": 100,
                    "completion_tokens": 24,
                    "generation_cache_implementation": "static",
                    "generation_cache_fallback_reason": None,
                },
            },
        },
    }


class VlmLatencyTraceAuditTests(unittest.TestCase):
    def test_bounded_record_never_copies_payload_content(self):
        row = bounded_record(gpt_record(), Path("model_debug.json"))
        serialized = json.dumps(row, ensure_ascii=False)
        for secret in ("SECRET_IMAGE", "SECRET_PROMPT", "SECRET_KEY", "SECRET_RAW_TEXT"):
            self.assertNotIn(secret, serialized)
        self.assertEqual(row["raw_text_chars"], len("SECRET_RAW_TEXT"))
        self.assertEqual(row["operation_count"], 1)
        self.assertEqual(row["hole_assessment_count"], 1)

    def test_summary_and_groups_ignore_non_numeric_values(self):
        self.assertEqual(
            summary([1, 2, 3, True, None, "4"]),
            {"count": 3, "min": 1.0, "median": 2.0, "p90": 2.8, "p95": 2.9, "max": 3.0},
        )
        groups = group_rows([
            {"image_count": 6, "generation_seconds": 10, "completion_tokens": 20},
            {"image_count": 6, "generation_seconds": 30, "completion_tokens": 40},
            {"image_count": 1, "generation_seconds": 5, "completion_tokens": 10},
        ], "image_count")
        self.assertEqual(groups["6"]["generation_seconds"]["median"], 20.0)
        self.assertEqual(groups["6"]["count"], 2)

    def test_audit_deduplicates_calls_and_reports_invalid_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            duplicate = root / "duplicate.json"
            invalid = root / "invalid.json"
            first.write_text(json.dumps({"trace": [gpt_record("same", 55.0)]}), encoding="utf-8")
            duplicate.write_text(json.dumps({"trace": [gpt_record("same", 55.0)]}), encoding="utf-8")
            invalid.write_text("{not json", encoding="utf-8")

            report = audit_paths([duplicate, invalid, first], long_seconds=50.0)

        self.assertEqual(report["scope"]["files_scanned"], 3)
        self.assertEqual(report["scope"]["unique_calls_with_local_metadata"], 1)
        self.assertEqual(report["scope"]["duplicate_copies_dropped"], 1)
        self.assertEqual(report["scope"]["uncached_calls"], 1)
        self.assertEqual(report["scope"]["long_uncached_calls"], 1)
        self.assertEqual(len(report["scope"]["invalid_files"]), 1)
        self.assertEqual(report["uncached_overall"]["generation_seconds"]["median"], 55.0)
        serialized = json.dumps(report, ensure_ascii=False)
        for secret in ("SECRET_IMAGE", "SECRET_PROMPT", "SECRET_KEY", "SECRET_RAW_TEXT"):
            self.assertNotIn(secret, serialized)


if __name__ == "__main__":
    unittest.main()
