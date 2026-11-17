import importlib.util
from pathlib import Path
import unittest


PATH = (Path(__file__).resolve().parents[1] / "tools"
        / "verify_vlm_exact_replay.py")
SPEC = importlib.util.spec_from_file_location("verify_vlm_exact_replay", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExactReplayVerifierTests(unittest.TestCase):
    def test_health_url_is_bound_to_chat_endpoint(self):
        self.assertEqual(MODULE.health_url(
            "http://127.0.0.1:28868/v1/chat/completions"),
            "http://127.0.0.1:28868/health")
        with self.assertRaises(ValueError):
            MODULE.health_url("http://127.0.0.1:28868/completions")

    def test_record_preserves_exact_cache_and_content_evidence(self):
        response = {
            "choices": [{"message": {"content": '{"ok":true}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            "local_vlm": {"cache_hit": True, "coalesced": False,
                          "response_cache_entries": 9},
        }

        record = MODULE.record(response, 12.5)

        self.assertTrue(record["cache_hit"])
        self.assertFalse(record["coalesced"])
        self.assertEqual(record["cache_entries"], 9)
        self.assertEqual(record["usage"]["completion_tokens"], 4)

    def test_cache_stat_delta_is_explicit_and_tolerates_new_counters(self):
        before = {"response_cache_stats": {
            "hits": 2, "misses": 3, "stores": 3}}
        after = {"response_cache_stats": {
            "hits": 3, "misses": 4, "stores": 4, "evictions": 1}}

        self.assertEqual(MODULE.cache_stat_delta(before, after), {
            "evictions": 1, "hits": 1, "misses": 1, "stores": 1,
        })


if __name__ == "__main__":
    unittest.main()
