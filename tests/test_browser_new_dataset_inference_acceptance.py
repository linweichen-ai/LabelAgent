import unittest

from tools.browser_new_dataset_inference_acceptance import (
    CAPTURE_TOPICS,
    expected_status_poll_abort,
)


class BrowserNewDatasetInferenceAcceptanceTests(unittest.TestCase):
    def test_cross_view_topics_are_explicit_and_bounded(self):
        self.assertEqual(CAPTURE_TOPICS, tuple(
            f"/capture/camera_{index}/image" for index in range(4)
        ))

    def test_only_intentional_run_status_abort_is_expected(self):
        self.assertTrue(expected_status_poll_abort(
            "GET", "http://127.0.0.1:18795/api/run/status", "net::ERR_ABORTED"))
        self.assertFalse(expected_status_poll_abort(
            "POST", "http://127.0.0.1:18795/api/run/status", "net::ERR_ABORTED"))
        self.assertFalse(expected_status_poll_abort(
            "GET", "http://127.0.0.1:18795/api/state", "net::ERR_ABORTED"))
        self.assertFalse(expected_status_poll_abort(
            "GET", "http://127.0.0.1:18795/api/run/status", "net::ERR_CONNECTION_RESET"))


if __name__ == "__main__":
    unittest.main()
