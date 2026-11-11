from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


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

    def test_workbench_supports_an_isolated_work_directory(self):
        server = (ROOT / "source/agent/server.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.get("LABELAGENT_WORK", ROOT / "work")', server)
        self.assertIn("WORK.mkdir(parents=True, exist_ok=True)", server)


if __name__ == "__main__":
    unittest.main()
