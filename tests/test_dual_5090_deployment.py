from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Dual5090DeploymentTests(unittest.TestCase):
    def test_linux_deployment_files_use_lf_line_endings(self):
        for relative in ("tools/dual_5090_services.sh",
                         "tools/dual_5090.env.example"):
            content = (ROOT / relative).read_bytes()
            self.assertNotIn(b"\r\n", content, relative)

    def test_health_url_is_built_after_port_assignment(self):
        script = (ROOT / "tools/dual_5090_services.sh").read_text(encoding="utf-8")
        self.assertIn('local port="$1" limit="$2" pid="$3"\n'
                      '  local url="http://127.0.0.1:$port/health"', script)


if __name__ == "__main__":
    unittest.main()
