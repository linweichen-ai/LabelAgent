import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReproCheckoutBrowserAcceptanceTests(unittest.TestCase):
    def test_setup_installs_pinned_browser_qa_dependencies(self):
        source = (ROOT / "tools" / "setup_windows.ps1").read_text(encoding="utf-8")
        self.assertIn("pip install -r tools\\requirements-browser-qa.txt", source)
        self.assertIn("Browser acceptance dependency installation failed", source)

    def test_full_setup_announces_size_and_guards_free_space(self):
        source = (ROOT / "tools" / "setup_windows.ps1").read_text(encoding="utf-8")
        self.assertIn("$free -lt 8GB", source)
        self.assertIn("about 2.9 GB will be downloaded", source)
        self.assertIn("final venv is about 5.1 GiB", source)
        self.assertIn("setup_public_browser_check.ps1 instead", source)

    def test_startup_requires_exact_ports_before_spawning(self):
        source = (ROOT / "tools" / "start_local.ps1").read_text(encoding="utf-8")
        doctor = source.index('$doctorArgs = @("tools\\repro_doctor.py", "--require-web-port")')
        sam_start = source.index("$sam = Start-Process")
        web_start = source.index("$web = Start-Process")
        self.assertLess(doctor, sam_start)
        self.assertLess(doctor, web_start)
        self.assertIn('"--require-sam-port"', source)

    def test_startup_rolls_back_every_spawn_failure(self):
        source = (ROOT / "tools" / "start_local.ps1").read_text(encoding="utf-8")
        spawn = source.index("$sam = Start-Process")
        cleanup = source.index("} catch {\n    foreach ($processId in $pids)", spawn)
        browser = source.index("if (-not $NoBrowser)")
        self.assertLess(spawn, cleanup)
        self.assertLess(cleanup, browser)
        self.assertIn("$web.HasExited", source[spawn:cleanup])
        self.assertIn("$sam.HasExited", source[spawn:cleanup])
        self.assertIn("foreach ($processId in $pids)", source[cleanup:browser])
        self.assertIn("Remove-Item -LiteralPath $Record", source[cleanup:browser])

    def test_checkout_runs_logic_and_real_synthetic_browser_acceptance(self):
        source = (ROOT / "tools" / "verify_checkout.ps1").read_text(encoding="utf-8")
        self.assertIn("$VenvScripts = Split-Path -Parent $Python", source)
        self.assertIn("$env:PATH = $VenvScripts + [IO.Path]::PathSeparator + $env:PATH", source)
        self.assertIn("tests\\test_browser_cvat_human_signoff_draft_acceptance.cjs", source)
        self.assertIn("tests\\test_cvat_human_signoff_drafts.js", source)
        self.assertIn("tools\\verify_synthetic_signoff_browser.py --ephemeral", source)
        self.assertIn("Synthetic offline signoff browser preflight failed", source)

    def test_browser_dependencies_fail_before_the_long_python_suite(self):
        source = (ROOT / "tools" / "verify_checkout.ps1").read_text(encoding="utf-8")
        node_check = source.index("Get-Command node")
        browser_check = source.index("tools\\verify_synthetic_signoff_browser.py --ephemeral")
        python_suite = source.index('unittest discover -s tests -p "test_*.py"')
        self.assertLess(node_check, python_suite)
        self.assertLess(browser_check, python_suite)

    def test_ephemeral_mode_uses_guarded_repo_test_temp(self):
        source = (ROOT / "tools" / "verify_synthetic_signoff_browser.py").read_text(encoding="utf-8")
        self.assertIn('ROOT / ".repro" / "test-temp"', source)
        self.assertIn('TemporaryDirectory(prefix="synthetic-signoff-browser-", dir=temp_parent)', source)
        self.assertIn('Path(temp_name) / "result"', source)


if __name__ == "__main__":
    unittest.main()
