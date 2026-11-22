import unittest

from tools.browser_idle_polling_acceptance import request_budget, validate_loopback_url


class BrowserIdlePollingAcceptanceTests(unittest.TestCase):
    def test_only_explicit_loopback_http_url_is_allowed(self):
        self.assertEqual(
            validate_loopback_url("http://127.0.0.1:8765/"),
            "http://127.0.0.1:8765",
        )
        for value in (
            "https://127.0.0.1:8765",
            "http://example.com:8765",
            "http://localhost",
        ):
            with self.assertRaises(ValueError):
                validate_loopback_url(value)

    def test_budget_preserves_one_startup_fallback_and_recovery_cadence(self):
        self.assertEqual(request_budget(32), {"run_status": 4, "state_summary": 4})
        self.assertEqual(request_budget(60), {"run_status": 6, "state_summary": 5})

    def test_too_short_observation_is_rejected(self):
        with self.assertRaises(ValueError):
            request_budget(14.9)


if __name__ == "__main__":
    unittest.main()
