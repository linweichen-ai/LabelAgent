import os
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from source.agent import server


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ROOT / "tools" / "setup_linux.sh",
    ROOT / "tools" / "start_linux.sh",
    ROOT / "tools" / "stop_linux.sh",
    ROOT / "tools" / "status_linux.sh",
]


class LinuxCompatibilityTest(unittest.TestCase):
    def test_default_bind_is_loopback_and_override_is_explicit(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUTOLABEL_HOST", None)
            os.environ.pop("AUTOLABEL_PORT", None)
            self.assertEqual(server.server_bind_address(), ("127.0.0.1", 8765))
        with mock.patch.dict(os.environ, {"AUTOLABEL_HOST": "0.0.0.0",
                                          "AUTOLABEL_PORT": "9876"}):
            self.assertEqual(server.server_bind_address(), ("0.0.0.0", 9876))

    def test_linux_scripts_are_lf_bash_and_parse_cleanly(self):
        bash = shutil.which("bash")
        for path in SCRIPTS:
            data = path.read_bytes()
            self.assertTrue(data.startswith(b"#!/usr/bin/env bash\n"), path)
            self.assertNotIn(b"\r\n", data, path)
            self.assertIn(b"set -Eeuo pipefail", data, path)
            if bash and os.name != "nt":
                result = subprocess.run([bash, "-n", str(path)], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))

    def test_startup_is_loopback_by_default_and_stop_checks_identity(self):
        start = SCRIPTS[1].read_text(encoding="utf-8")
        status = SCRIPTS[3].read_text(encoding="utf-8")
        stop = SCRIPTS[2].read_text(encoding="utf-8")
        self.assertIn('HOST="127.0.0.1"', start)
        self.assertIn("AUTOLABEL_HOST", start)
        self.assertIn('AUTOLABEL_PORT="$PORT"', start)
        self.assertIn("LABELAGENT_VLM_URL", start)
        self.assertIn("LABELAGENT_VLM_MODEL", start)
        self.assertIn('"workbench_port": int(port)', start)
        self.assertIn("record.get('workbench_port', 8765)", status)
        self.assertNotIn("http://127.0.0.1:8765/api/run/status", status)
        self.assertIn("source.agent.server", start)
        self.assertIn("source.sam3_service.server", start)
        self.assertEqual(start.count("9>&- &"), 2)
        self.assertIn('cmdline" != *"$module"*', stop)
        self.assertNotIn("pkill", stop)
        self.assertNotIn("killall", stop)

    def test_environment_model_defaults_do_not_override_persisted_config(self):
        original_state = server.state
        original_work = server.WORK
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as directory:
                server.WORK = Path(directory)
                server.state = {
                    "video": None, "topic": None, "labels": [], "truth": {},
                    "reference_frames": {}, "config": {
                        "llm_url": "http://persisted.test/v1",
                        "llm_model": "persisted-model",
                    }, "candidates": {}, "candidates_by_topic": {},
                    "agent_runs": {}, "agent_runs_by_topic": {},
                }
                with mock.patch.dict(os.environ, {
                    "LABELAGENT_VLM_URL": "http://server.test/v1",
                    "LABELAGENT_VLM_MODEL": "server-model",
                    "LABELAGENT_FAST_VLM_URL": "http://fast.test/v1",
                    "LABELAGENT_FAST_VLM_MODEL": "fast-model",
                }):
                    server.load()
                self.assertEqual(server.state["config"]["llm_url"],
                                 "http://persisted.test/v1")
                self.assertEqual(server.state["config"]["llm_model"],
                                 "persisted-model")
                self.assertEqual(server.state["config"]["llm_fast_url"],
                                 "http://fast.test/v1")
                self.assertEqual(server.state["config"]["llm_fast_model"],
                                 "fast-model")
        finally:
            server.state = original_state
            server.WORK = original_work

    def test_windows_remote_workbench_tunnel_is_bounded_and_identity_checked(self):
        connect = (ROOT / "tools" / "connect_remote_workbench.ps1").read_text(
            encoding="utf-8")
        disconnect = (ROOT / "tools" / "disconnect_remote_workbench.ps1").read_text(
            encoding="utf-8")
        self.assertIn('RemotePort = 18765', connect)
        self.assertIn('LocalPort = 18765', connect)
        self.assertIn('"-L", $marker', connect)
        self.assertIn("ExitOnForwardFailure=yes", connect)
        self.assertIn("ServerAliveInterval=30", connect)
        self.assertIn("-WindowStyle Hidden", connect)
        self.assertIn("/api/run/status", connect)
        self.assertIn("/api/model_readiness", connect)
        self.assertIn("Get-WorkbenchReadiness", connect)
        self.assertIn("models are not ready", connect)
        self.assertIn("Stop-Process -Id $existing.pid", connect)
        self.assertIn("identity-checked managed tunnel", connect)
        self.assertNotIn("password", connect.lower())
        self.assertIn("process identity does not match", disconnect)
        self.assertNotIn("Get-Process ssh", disconnect)


if __name__ == "__main__":
    unittest.main()
