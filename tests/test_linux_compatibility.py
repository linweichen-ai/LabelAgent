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
        stop = SCRIPTS[2].read_text(encoding="utf-8")
        self.assertIn('HOST="127.0.0.1"', start)
        self.assertIn("AUTOLABEL_HOST", start)
        self.assertIn("source.agent.server", start)
        self.assertIn("source.sam3_service.server", start)
        self.assertIn('cmdline" != *"$module"*', stop)
        self.assertNotIn("pkill", stop)
        self.assertNotIn("killall", stop)


if __name__ == "__main__":
    unittest.main()
