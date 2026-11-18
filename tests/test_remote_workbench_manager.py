import importlib.util
from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock


PATH = Path(__file__).resolve().parents[1] / "tools" / "manage_remote_workbench.py"
SPEC = importlib.util.spec_from_file_location("manage_remote_workbench", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RemoteWorkbenchManagerTests(unittest.TestCase):
    def test_primary_prewarm_uses_long_timeout_and_accepts_busy_status(self):
        ready = {"prewarmed": True, "elapsed_ms": 15000}
        with mock.patch.object(MODULE, "post_json", return_value=ready) as post:
            self.assertEqual(MODULE.prewarm_primary_vlm(), ready)
        post.assert_called_once_with(
            MODULE.WEB_PORT, "/api/model_prewarm", {}, timeout=90,
            accepted_statuses=(200, 202))

    def test_primary_prewarm_retries_busy_without_blocking_forever(self):
        ready = {"prewarmed": True, "elapsed_ms": 15000}
        with (mock.patch.object(MODULE, "post_json", side_effect=[
                  {"prewarmed": False, "code": "vlm_busy"}, ready]) as post,
              mock.patch.object(MODULE.time, "sleep") as sleep):
            self.assertEqual(MODULE.prewarm_primary_vlm(), ready)
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(2)

    def test_optional_fast_route_uses_a_short_default_wait(self):
        self.assertEqual(MODULE.optional_fast_wait_seconds({}), 3.0)

    def test_optional_fast_route_wait_is_bounded_and_invalid_values_are_safe(self):
        self.assertEqual(MODULE.optional_fast_wait_seconds({
            "LABELAGENT_FAST_ROUTE_WAIT_SECONDS": "0"}), 1.0)
        self.assertEqual(MODULE.optional_fast_wait_seconds({
            "LABELAGENT_FAST_ROUTE_WAIT_SECONDS": "60"}), 20.0)
        self.assertEqual(MODULE.optional_fast_wait_seconds({
            "LABELAGENT_FAST_ROUTE_WAIT_SECONDS": "7.5"}), 7.5)
        self.assertEqual(MODULE.optional_fast_wait_seconds({
            "LABELAGENT_FAST_ROUTE_WAIT_SECONDS": "invalid"}), 3.0)

    def test_startup_uses_the_bounded_wait_for_the_optional_route(self):
        source = PATH.read_text(encoding="utf-8")
        self.assertIn(
            'optional_fast_wait_seconds(), "5090 可选加速通道")', source)
        self.assertNotIn('20, "5090 可选加速通道")', source)

    def test_unready_remote_skips_optional_tunnel_without_spawning(self):
        items = {}
        newly_started = []
        with (mock.patch.object(MODULE, "remote_fast_ready", return_value=False),
              mock.patch.object(MODULE.subprocess, "Popen") as popen):
            with self.assertRaisesRegex(RuntimeError, "尚未就绪"):
                MODULE.start_optional_fast_tunnel(items, {}, newly_started)
        popen.assert_not_called()
        self.assertEqual(items, {})
        self.assertEqual(newly_started, [])

    def test_ready_remote_starts_and_verifies_optional_tunnel(self):
        items = {}
        newly_started = []
        process = mock.Mock(pid=321)
        process_record = mock.Mock()
        process_record.create_time.return_value = 123.0
        with (tempfile.TemporaryDirectory() as temporary,
              mock.patch.object(MODULE, "LOCAL", Path(temporary)),
              mock.patch.object(MODULE, "remote_fast_ready", return_value=True),
              mock.patch.object(MODULE.subprocess, "Popen", return_value=process) as popen,
              mock.patch.object(MODULE.psutil, "Process", return_value=process_record),
              mock.patch.object(MODULE, "save") as save,
              mock.patch.object(MODULE, "wait_for") as wait_for):
            (Path(temporary) / "logs").mkdir()
            MODULE.start_optional_fast_tunnel(items, {}, newly_started)
        popen.assert_called_once()
        save.assert_called_once_with(items)
        wait_for.assert_called_once_with(
            items["fast_tunnel"], ((MODULE.FAST_VLM_PORT, "/v1/models"),),
            MODULE.optional_fast_wait_seconds(), "5090 可选加速通道")
        self.assertEqual(newly_started, ["fast_tunnel"])
        self.assertEqual(items["fast_tunnel"]["pid"], 321)

    def test_isolated_acceptance_uses_the_verified_primary_tunnels(self):
        self.assertEqual(MODULE.isolate_model_config(), {
            "llm_url": f"http://127.0.0.1:{MODULE.VLM_PORT}/v1",
            "llm_model": "labelagent-vlm8b",
            "sam3_url": f"http://127.0.0.1:{MODULE.SAM_PORT}",
        })
        source = PATH.read_text(encoding="utf-8")
        self.assertNotIn(
            'SAM3_URL="http://127.0.0.1:18766"', source)
        self.assertNotIn(
            '"llm_url": "http://127.0.0.1:18878/v1"', source)

    def test_isolated_acceptance_copies_only_a_valid_label_registry(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state.json"
            state.write_text(json.dumps({
                "labels": ["floor", "slope", "person"],
                "truth": {"0": ["must not be copied"]},
            }), encoding="utf-8")
            self.assertEqual(MODULE.isolate_labels(state),
                             ["floor", "slope", "person"])
            for labels in ([], ["floor", "floor"], ["floor", ""]):
                state.write_text(json.dumps({"labels": labels}), encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    MODULE.isolate_labels(state)


if __name__ == "__main__":
    unittest.main()
