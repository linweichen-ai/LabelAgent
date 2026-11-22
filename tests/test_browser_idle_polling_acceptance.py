import unittest

from tools.browser_idle_polling_acceptance import (
    request_budget,
    sanitize_initiators,
    validate_loopback_url,
)


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

    def test_steady_state_budget_preserves_one_interval_boundary(self):
        self.assertEqual(request_budget(32), {"run_status": 3, "state_summary": 2})
        self.assertEqual(request_budget(60), {"run_status": 5, "state_summary": 3})

    def test_initiators_keep_only_loopback_paths_and_function_names(self):
        result = sanitize_initiators([{
            "target": "http://127.0.0.1:8766/api/run/status?secret=value",
            "at": 12.3456,
            "stack": (
                "Error: fetch initiator\n"
                " at fetchAgentStatus (http://127.0.0.1:8766/app.js?v=206:1:2)\n"
                " at leaked (file:///C:/Users/private/script.js:1:2)\n"
                " at external (https://example.com/remote.js:1:2)"
            ),
        }], "http://127.0.0.1:8766")
        self.assertEqual(result, [{
            "path": "/api/run/status",
            "at_ms": 12.346,
            "frames": ["fetchAgentStatus@/app.js"],
        }])

    def test_too_short_observation_is_rejected(self):
        with self.assertRaises(ValueError):
            request_budget(14.9)


if __name__ == "__main__":
    unittest.main()
