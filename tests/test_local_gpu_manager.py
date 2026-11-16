import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import psutil

SPEC = importlib.util.spec_from_file_location("local_gpu_manager", Path(__file__).resolve().parents[1] / "tools/local_gpu_manager.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SingleGPUManagerTests(unittest.TestCase):
    def test_switch_waits_for_owned_process_tree_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, "-c", "import time;time.sleep(60)"]
            specs = {kind: {"command": command, "url": "http://127.0.0.1:1"} for kind in ("sam3", "qwen")}
            manager = MODULE.WorkerManager(specs, directory, directory, Path(directory)/"events.json")
            try:
                with patch.object(MODULE, "http", return_value=(200, b'{"status":"ok"}', "application/json")):
                    manager.activate("sam3")
                    time.sleep(0.3)
                    first = psutil.Process(manager.child.pid)
                    descendants = first.children(recursive=True)
                    manager.activate("qwen")
                    self.assertFalse(first.is_running())
                    self.assertTrue(all(not process.is_running() for process in descendants))
                    events = [item["event"] for item in manager.events]
                    self.assertLess(events.index("worker_exited"), len(events)-2)
                    self.assertEqual(manager.status()["live_owned_workers"], 1)
            finally:
                manager.stop()

    def test_busy_manager_does_not_start_another_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = MODULE.WorkerManager({}, directory, directory, Path(directory)/"events.json")
            with manager.lock, patch.object(manager, "activate") as activate:
                with self.assertRaises(RuntimeError):
                    manager.call("qwen", "/v1/chat/completions", b"{}")
                activate.assert_not_called()

    def test_public_health_and_models_do_not_load_a_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = MODULE.WorkerManager({}, directory, directory, Path(directory)/"events.json")
            sam, vlm = MODULE.create_apps(manager, "local-test")
            self.assertFalse(sam.test_client().get("/health").json["model_loaded"])
            self.assertTrue(vlm.test_client().get("/health").json["on_demand"])
            self.assertEqual(vlm.test_client().get("/v1/models").json["data"][0]["id"], "local-test")
            self.assertEqual(manager.started, 0)


if __name__ == "__main__":
    unittest.main()
