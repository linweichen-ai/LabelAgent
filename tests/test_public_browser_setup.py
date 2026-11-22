import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicBrowserSetupTests(unittest.TestCase):
    def test_setup_isolated_from_the_full_gpu_environment(self):
        source = (ROOT / "tools" / "setup_public_browser_check.ps1").read_text(encoding="utf-8")
        self.assertIn('.repro\\public-browser-venv', source)
        self.assertIn("resolve_python312.ps1", source)
        self.assertIn("requirements-public-browser-qa.txt", source)
        self.assertNotIn("requirements.txt\n", source)
        self.assertNotIn("torch", source.lower())
        self.assertNotIn("Get-Command node", source)
        self.assertNotIn("& node", source)
        self.assertNotIn("require-cuda", source.lower())

    def test_setup_runs_dependency_and_real_browser_readback(self):
        source = (ROOT / "tools" / "setup_public_browser_check.ps1").read_text(encoding="utf-8")
        self.assertIn("-m pip check", source)
        self.assertIn("tools\\verify_synthetic_signoff_browser.py --ephemeral", source)
        self.assertIn("No model weights, data, Node.js or GPU were used", source)

    def test_minimal_requirements_are_exactly_pinned(self):
        requirements = (ROOT / "tools" / "requirements-public-browser-qa.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(requirements, ["Pillow==12.3.0", "playwright==1.65.2"])

    def test_python_resolver_supports_standard_and_registered_runtimes(self):
        source = (ROOT / "tools" / "resolve_python312.ps1").read_text(encoding="utf-8")
        self.assertIn('py -3.12 -c "import sys; print(sys.executable)"', source)
        self.assertIn("py -0p", source)
        self.assertIn("Resolve-LabelAgentPython312", source)
        self.assertIn("Python 3.12 was not found", source)


if __name__ == "__main__":
    unittest.main()
