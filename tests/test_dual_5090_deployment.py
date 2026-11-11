import os
from pathlib import Path
import re
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


def bash_executable():
    candidates = []
    if os.name == "nt":
        candidates.append(Path(os.environ.get("ProgramFiles", "C:/Program Files")) /
                          "Git/bin/bash.exe")
    else:
        candidates.append(shutil.which("bash"))
    return next((str(path) for path in candidates if path and Path(path).is_file()), None)


class Dual5090DeploymentTests(unittest.TestCase):
    def test_all_shell_scripts_use_lf_line_endings(self):
        scripts = sorted((ROOT / "tools").rglob("*.sh"))
        self.assertTrue(scripts, "no shell scripts found")
        for path in scripts:
            relative = path.relative_to(ROOT).as_posix()
            self.assertNotIn(b"\r", path.read_bytes(), relative)

    def test_linux_environment_template_uses_lf_line_endings(self):
        relative = "tools/dual_5090.env.example"
        self.assertNotIn(b"\r", (ROOT / relative).read_bytes(), relative)

    def test_health_url_is_built_after_port_assignment(self):
        script = (ROOT / "tools/dual_5090_services.sh").read_text(encoding="utf-8")
        self.assertIn('local port="$1" limit="$2" pid="$3"\n'
                      '  local url="http://127.0.0.1:$port/health"', script)

    def test_wait_health_runs_with_nounset_enabled(self):
        bash = bash_executable()
        if bash is None:
            self.skipTest("bash is not installed")
        script = (ROOT / "tools/dual_5090_services.sh").read_text(encoding="utf-8")
        match = re.search(r"(?ms)^wait_health\(\) \{\n.*?^\}\n", script)
        self.assertIsNotNone(match, "wait_health function not found")
        program = "\n".join((
            "set -u",
            "curl() { printf '%s' '{\"status\":\"ok\"}'; return 0; }",
            "kill() { return 0; }",
            "sleep() { return 0; }",
            match.group(0),
            "wait_health 18766 1 123",
        ))
        result = subprocess.run([bash, "-c", program], capture_output=True,
                                text=True, timeout=10, check=False)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('{"status":"ok"}', result.stdout)

    def test_workbench_supports_an_isolated_work_directory(self):
        server = (ROOT / "source/agent/server.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.get("LABELAGENT_WORK", ROOT / "work")', server)
        self.assertIn("WORK.mkdir(parents=True, exist_ok=True)", server)


if __name__ == "__main__":
    unittest.main()
