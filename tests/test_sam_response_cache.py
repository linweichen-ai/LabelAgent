import threading
import time
import unittest

from source.sam3_service.response_cache import ExactResponseCoordinator


class ExactResponseCoordinatorTests(unittest.TestCase):
    def test_exact_hit_returns_independent_json_value(self):
        cache = ExactResponseCoordinator(max_entries=2, max_bytes=1024)
        claim = cache.claim("a")
        self.assertEqual(claim.role, "owner")
        cache.complete("a", claim.flight, {"items": [1, 2]})

        first = cache.claim("a")
        self.assertEqual(first.role, "hit")
        first.payload["items"].append(3)
        second = cache.claim("a")
        self.assertEqual(second.payload, {"items": [1, 2]})
        self.assertEqual(cache.stats()["storage"], "cpu_json")

    def test_lru_and_byte_limits_are_bounded(self):
        cache = ExactResponseCoordinator(max_entries=2, max_bytes=80)
        for key in ("a", "b", "c"):
            claim = cache.claim(key)
            cache.complete(key, claim.flight, {"key": key, "text": "x" * 10})
        self.assertLessEqual(cache.stats()["entries"], 2)
        self.assertLessEqual(cache.stats()["bytes"], 80)
        self.assertEqual(cache.claim("a").role, "owner")

    def test_identical_inflight_call_waits_for_one_owner(self):
        cache = ExactResponseCoordinator()
        owner = cache.claim("same")
        waiter = cache.claim("same")
        self.assertEqual(owner.role, "owner")
        self.assertEqual(waiter.role, "wait")

        def complete():
            time.sleep(0.02)
            cache.complete("same", owner.flight, {"result": 7})

        thread = threading.Thread(target=complete)
        thread.start()
        payload, waited = cache.wait(waiter.flight, timeout=1)
        thread.join()
        self.assertEqual(payload, {"result": 7})
        self.assertGreater(waited, 0)
        self.assertEqual(cache.stats()["misses"], 1)
        self.assertEqual(cache.stats()["coalesced_requests"], 1)

    def test_failure_reaches_waiter_and_next_call_can_retry(self):
        cache = ExactResponseCoordinator()
        owner = cache.claim("same")
        waiter = cache.claim("same")
        cache.fail("same", owner.flight, RuntimeError("boom"))
        with self.assertRaisesRegex(RuntimeError, "boom"):
            cache.wait(waiter.flight, timeout=1)
        self.assertEqual(cache.claim("same").role, "owner")

    def test_oversized_response_is_shared_but_not_retained(self):
        cache = ExactResponseCoordinator(max_entries=2, max_bytes=10)
        owner = cache.claim("large")
        waiter = cache.claim("large")
        cache.complete("large", owner.flight, {"payload": "too large"})
        payload, _ = cache.wait(waiter.flight, timeout=1)
        self.assertEqual(payload["payload"], "too large")
        self.assertEqual(cache.stats()["entries"], 0)
        self.assertEqual(cache.claim("large").role, "owner")


if __name__ == "__main__":
    unittest.main()
