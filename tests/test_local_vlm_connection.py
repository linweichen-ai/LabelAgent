import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("connect_local_vlm", Path(__file__).resolve().parents[1] / "tools/connect_local_vlm.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LocalVLMConnectionTests(unittest.TestCase):
    def test_private_tunnel_url_rejects_external_hosts_and_credentials(self):
        self.assertEqual(MODULE.loopback_url("http://127.0.0.1:18868/v1/"), "http://127.0.0.1:18868/v1")
        for url in ("https://example.test/v1", "http://user:secret@localhost/v1", "http://192.168.1.2/v1", "http://localhost/v1?token=x"):
            with self.assertRaises(ValueError):
                MODULE.loopback_url(url)

    def test_contract_test_double_cannot_be_applied_as_real_model(self):
        with patch.object(MODULE, "request_json", return_value={"model_loaded": True, "backend": "contract_test_double", "model": "test"}):
            with self.assertRaises(ValueError):
                MODULE.require_real_service("http://localhost:18868/v1", "test")

    def test_busy_workbench_is_not_reconfigured(self):
        with patch.object(MODULE, "require_real_service"), patch.object(MODULE, "request_json", return_value={"status": "running"}) as call:
            with self.assertRaises(ValueError):
                MODULE.apply_configuration("http://localhost:18868/v1", "test", "http://localhost:8765")
        self.assertEqual(call.call_count, 1)

    def test_cloud_key_is_replaced_by_explicit_nonsecret_local_value(self):
        with patch.object(MODULE, "require_real_service"), patch.object(MODULE, "request_json", side_effect=[{"status": "idle"}, {"saved": True}]) as call:
            self.assertEqual(MODULE.apply_configuration("http://localhost:18868/v1", "test", "http://localhost:8765"), {"saved": True})
        self.assertEqual(call.call_args.args[1]["llm_key"], "local")


if __name__ == "__main__":
    unittest.main()
