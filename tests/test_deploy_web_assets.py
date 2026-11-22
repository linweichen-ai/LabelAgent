import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "deploy_web_assets.ps1"
BASE = "d5f60466adcb8973dea6dbc4cab3dc3253556c2f"
ASSETS = (
    "source/web/app.js",
    "source/web/guide.js",
    "source/web/index.html",
)


def current_source_ref():
    clean = all(
        subprocess.check_output(
            ["git", "-C", str(ROOT), "hash-object", str(ROOT / relative)],
            text=True,
        ).strip()
        == subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", f"HEAD:{relative}"],
            text=True,
        ).strip()
        for relative in ASSETS
    )
    if clean:
        return "HEAD"
    ref = subprocess.check_output(
        ["git", "-C", str(ROOT), "stash", "create", "deploy-web-assets-test"],
        text=True,
    ).strip()
    if not ref:
        raise RuntimeError("could not snapshot current tracked web assets")
    return ref


SOURCE_REF = current_source_ref()


def git_blob(ref, relative):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{ref}:{relative}"]
    )


def run_deploy(target):
    return subprocess.run(
        [
            "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(SCRIPT), "-TargetRoot", str(target),
            "-SourceRoot", str(ROOT), "-SourceRef", SOURCE_REF,
            "-ExpectedTargetRef", BASE,
        ],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=30,
    )


class DeployWebAssetsTests(unittest.TestCase):
    def make_target(self, directory, ref):
        root = Path(directory)
        for relative in ASSETS:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(git_blob(ref, relative))
        return root

    def test_exact_previous_assets_are_backed_up_and_atomically_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            target = self.make_target(directory, BASE)
            completed = run_deploy(target)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            receipt = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertEqual(receipt["status"], "deployed")
            self.assertEqual(len(receipt["deployed_files"]), len(ASSETS))
            backup = Path(receipt["backup_root"])
            for relative in ASSETS:
                self.assertEqual((target / relative).read_bytes(), (ROOT / relative).read_bytes())
                self.assertEqual((backup / relative).read_bytes(), git_blob(BASE, relative))

    def test_unrecognized_target_change_is_rejected_before_any_write(self):
        with tempfile.TemporaryDirectory() as directory:
            target = self.make_target(directory, BASE)
            changed = target / "source/web/guide.js"
            changed.write_bytes(changed.read_bytes() + b"\nlocal divergence\n")
            before = {relative: (target / relative).read_bytes() for relative in ASSETS}
            completed = run_deploy(target)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("unrecognized local changes", completed.stderr)
            self.assertEqual(
                before,
                {relative: (target / relative).read_bytes() for relative in ASSETS},
            )

    def test_current_assets_are_a_verified_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            target = self.make_target(directory, SOURCE_REF)
            completed = run_deploy(target)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            receipt = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertEqual(receipt["status"], "already_current")
            self.assertEqual(receipt["deployed_files"], [])


if __name__ == "__main__":
    unittest.main()
