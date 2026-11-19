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
    def test_config_and_label_writes_use_fresh_server_bindings(self):
        bootstrap = {
            "path": "D:/data/example.mcap", "topic": "camera2",
            "file_activation_generation": 17,
            "config_state_sha256": "a" * 64,
            "label_state_sha256": "b" * 64,
        }
        with mock.patch.object(MODULE, "get_json", return_value=bootstrap):
            config = MODULE.config_write_payload(8765, {"llm_model": "model"})
            labels = MODULE.label_write_payload(8765, ["floor", "slope"])
        self.assertEqual(config["expected_path"], bootstrap["path"])
        self.assertEqual(config["expected_topic"], bootstrap["topic"])
        self.assertEqual(config["expected_file_activation_generation"], 17)
        self.assertEqual(config["expected_config_state_sha256"], "a" * 64)
        self.assertEqual(labels["path"], bootstrap["path"])
        self.assertEqual(labels["topic"], bootstrap["topic"])
        self.assertEqual(labels["file_activation_generation"], 17)
        self.assertEqual(labels["expected_label_state_sha256"], "b" * 64)

    def test_bound_config_reloads_binding_after_one_conflict(self):
        payloads = [
            {"expected_config_state_sha256": "a" * 64},
            {"expected_config_state_sha256": "b" * 64},
        ]
        conflict = MODULE.urllib.error.HTTPError(
            "http://127.0.0.1:8765/api/config", 409, "Conflict", {}, None)
        with (mock.patch.object(MODULE, "config_write_payload",
                                side_effect=payloads) as bind,
              mock.patch.object(MODULE, "post_json",
                                side_effect=[conflict, {"saved": True}]) as post):
            result = MODULE.post_bound_config(8765, {"llm_model": "model"})
        self.assertEqual(result, {"saved": True})
        self.assertEqual(bind.call_count, 2)
        self.assertEqual(post.call_count, 2)

    def test_runtime_temp_redirects_from_low_system_volume(self):
        with (tempfile.TemporaryDirectory() as temporary,
              mock.patch.object(MODULE, "storage_free_bytes", side_effect=[
                  512 * 1024 ** 2, 12 * 1024 ** 3, 12 * 1024 ** 3])):
            runtime_temp = Path(temporary) / "runtime-temp"
            environment, report = MODULE.prepare_runtime_environment(
                {"TEMP": "C:/system-temp", "KEEP": "yes"}, runtime_temp)
        self.assertEqual(environment["TEMP"], str(runtime_temp.resolve()))
        self.assertEqual(environment["TMP"], str(runtime_temp.resolve()))
        self.assertEqual(environment["TMPDIR"], str(runtime_temp.resolve()))
        self.assertEqual(environment["LABELAGENT_RUNTIME_TEMP"],
                         str(runtime_temp.resolve()))
        self.assertEqual(environment["KEEP"], "yes")
        self.assertTrue(report["redirected"])
        self.assertEqual(report["system_volume_free_bytes"], 512 * 1024 ** 2)
        self.assertEqual(report["system_temp_free_bytes"], 12 * 1024 ** 3)
        self.assertEqual(report["runtime_free_bytes"], 12 * 1024 ** 3)

    def test_runtime_temp_refuses_critically_low_target_volume(self):
        with (tempfile.TemporaryDirectory() as temporary,
              mock.patch.object(MODULE, "storage_free_bytes", side_effect=[
                  8 * 1024 ** 3, 8 * 1024 ** 3, 128 * 1024 ** 2])):
            with self.assertRaisesRegex(RuntimeError, "不足 256 MiB"):
                MODULE.prepare_runtime_environment(
                    {"TEMP": temporary}, Path(temporary) / "runtime-temp")

    def test_workbench_temp_falls_back_to_local_app_data_when_default_is_full(self):
        environment = {"LOCALAPPDATA": "C:/Users/test/AppData/Local"}
        with mock.patch.object(MODULE, "storage_free_bytes",
                               side_effect=[128 * 1024 ** 2, 3 * 1024 ** 3]):
            selected = MODULE.select_workbench_runtime_temp(
                environment, "D:/LabelAgent/runtime-temp/workbench-8765")
        self.assertEqual(
            selected,
            (Path(environment["LOCALAPPDATA"]) / "LabelAgent" / "runtime-temp"
             / "workbench-8765").resolve())

    def test_workbench_temp_override_is_explicit_and_port_scoped(self):
        selected = MODULE.select_workbench_runtime_temp(
            {"LABELAGENT_RUNTIME_TEMP_ROOT": "C:/LabelAgentTemp"},
            "D:/LabelAgent/runtime-temp/workbench-8765")
        self.assertEqual(selected,
                         (Path("C:/LabelAgentTemp") / "workbench-8765").resolve())

    def test_low_space_advisory_reports_redirect_without_blocking(self):
        report = {
            "system_volume": "C:\\",
            "system_volume_free_bytes": 512 * 1024 ** 2,
            "system_temp": "C:/system-temp",
            "system_temp_free_bytes": 512 * 1024 ** 2,
            "runtime_temp": "D:/runtime-temp",
            "runtime_free_bytes": 12 * 1024 ** 3,
            "redirected": True,
        }
        with mock.patch("builtins.print") as output:
            MODULE.print_storage_advisory(report)
        message = " ".join(str(call.args[0]) for call in output.call_args_list)
        self.assertIn("Windows 系统盘", message)
        self.assertIn("工作台临时文件已固定", message)
        self.assertNotIn("数据盘可用空间低于", message)

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

    def test_ready_5090_sam_starts_and_verifies_dedicated_tunnel(self):
        items = {}
        newly_started = []
        process = mock.Mock(pid=654)
        process_record = mock.Mock()
        process_record.create_time.return_value = 456.0
        with (tempfile.TemporaryDirectory() as temporary,
              mock.patch.object(MODULE, "LOCAL", Path(temporary)),
              mock.patch.object(MODULE, "remote_5090_sam_ready", return_value=True),
              mock.patch.object(MODULE.subprocess, "Popen", return_value=process),
              mock.patch.object(MODULE.psutil, "Process", return_value=process_record),
              mock.patch.object(MODULE, "save") as save,
              mock.patch.object(MODULE, "wait_for") as wait_for):
            (Path(temporary) / "logs").mkdir()
            MODULE.start_5090_sam_tunnel(items, {}, newly_started)
        save.assert_called_once_with(items)
        wait_for.assert_called_once_with(
            items["sam_5090_tunnel"], ((MODULE.FAST_SAM_PORT, "/health"),),
            MODULE.optional_fast_wait_seconds(), "5090 SAM 通道")
        self.assertEqual(newly_started, ["sam_5090_tunnel"])
        self.assertEqual(items["sam_5090_tunnel"]["pid"], 654)

    def test_standard_start_persists_an_explicit_sam_route(self):
        source = PATH.read_text(encoding="utf-8")
        self.assertIn('"sam3_url": f"http://127.0.0.1:{sam_endpoint_port}"', source)
        self.assertNotIn("sam_endpoint_port = FAST_SAM_PORT", source)
        self.assertIn("多提示契约验收完成前仅作备用", source)

    def test_5090_sam_contract_requires_complete_reused_multi_prompt_output(self):
        accepted = {
            "items": [
                {"prompt": "warehouse floor", "results": []},
                {"prompt": "traversable ramp surface", "results": []},
            ],
            "diagnostics": {"feature_reuse": True, "image_feature_passes": 1},
        }
        with mock.patch.object(MODULE, "post_json", return_value=accepted) as post:
            report = MODULE.verify_5090_sam_multi_prompt_contract()
        self.assertTrue(report["accepted"])
        self.assertTrue(report["image_encoded_once"])
        self.assertEqual(report["prompt_count"], 2)
        self.assertEqual(post.call_args.args[:2],
                         (MODULE.FAST_SAM_PORT, "/segment_text_multi"))

        incomplete = {"items": accepted["items"][:1],
                      "diagnostics": {"feature_reuse": True}}
        with mock.patch.object(MODULE, "post_json", return_value=incomplete):
            self.assertFalse(
                MODULE.verify_5090_sam_multi_prompt_contract()["accepted"])

    def test_standard_sam_route_falls_back_unless_contract_is_accepted(self):
        with mock.patch.object(MODULE, "verify_5090_sam_multi_prompt_contract",
                               return_value={"accepted": True}):
            self.assertEqual(MODULE.select_standard_sam_port(), MODULE.FAST_SAM_PORT)
        with mock.patch.object(MODULE, "verify_5090_sam_multi_prompt_contract",
                               return_value={"accepted": False}):
            self.assertEqual(MODULE.select_standard_sam_port(), MODULE.SAM_PORT)
        with mock.patch.object(MODULE, "verify_5090_sam_multi_prompt_contract",
                               side_effect=RuntimeError("offline")):
            self.assertEqual(MODULE.select_standard_sam_port(), MODULE.SAM_PORT)

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
