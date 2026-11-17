import importlib.util
from pathlib import Path
import unittest


PATH = (Path(__file__).resolve().parents[1] / "tools"
        / "verify_vlm_concurrent_coalescing.py")
SPEC = importlib.util.spec_from_file_location(
    "verify_vlm_concurrent_coalescing", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ConcurrentCoalescingVerifierTests(unittest.TestCase):
    def test_acceptance_requires_one_leader_one_follower_and_exact_counters(self):
        records = [
            {"cache_hit": False, "coalesced": False,
             "content_sha256": "same", "usage": {"total_tokens": 12}},
            {"cache_hit": True, "coalesced": True,
             "content_sha256": "same", "usage": {"total_tokens": 12}},
        ]
        stats = {"misses": 2, "stores": 1, "coalesced": 1, "hits": 0,
                 "busy_rejections": 0, "coalesce_timeouts": 0,
                 "coalesce_reuse_failures": 0}
        self.assertTrue(all(MODULE.coalescing_acceptance(
            records, stats).values()))
        stats["coalesce_reuse_failures"] = 1
        self.assertFalse(MODULE.coalescing_acceptance(
            records, stats)["expected_cache_stat_delta"])


if __name__ == "__main__":
    unittest.main()
