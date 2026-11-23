import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from tools import deploy_formal_minimal_overlay as deployer


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "deploy_formal_minimal_overlay.py"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StatusHandler(BaseHTTPRequestHandler):
    status = "idle"

    def do_GET(self):
        body = json.dumps({"status": self.status}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


class FormalMinimalOverlayDeployTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.candidate = self.root / "candidate"
        self.target = self.root / "formal"
        source = self.candidate / "files/source/web/asset.js"
        target = self.target / "source/web/asset.js"
        source.parent.mkdir(parents=True)
        target.parent.mkdir(parents=True)
        source.write_text("new\n", encoding="utf-8")
        target.write_text("old\n", encoding="utf-8")
        manifest = {
            "schema_version": "labelagent.formal-minimal-overlay-candidate.v1",
            "candidate_validated": True,
            "deployment_performed": False,
            "formal_files_written": 0,
            "files": [{
                "relative": "source/web/asset.js",
                "formal_sha256": digest(target),
                "candidate_sha256": digest(source),
            }],
        }
        (self.candidate / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), StatusHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.status_url = f"http://127.0.0.1:{self.server.server_port}/api/run/status"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.temp.cleanup()

    def run_tool(self, *extra):
        return subprocess.run(
            ["python", str(SCRIPT), "--candidate-root", str(self.candidate),
             "--target-root", str(self.target), *extra],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
        )

    def test_default_is_read_only_dry_run(self):
        before = (self.target / "source/web/asset.js").read_bytes()
        result = self.run_tool()
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt["status"], "dry_run")
        self.assertFalse(receipt["explicit_execute"])
        self.assertEqual((self.target / "source/web/asset.js").read_bytes(), before)

    def test_execute_requires_idle_and_creates_verified_backup(self):
        old = (self.target / "source/web/asset.js").read_bytes()
        result = self.run_tool("--execute", "--status-url", self.status_url)
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt["status"], "deployed")
        self.assertEqual(receipt["idle_probe"], {"status": "idle"})
        self.assertEqual((self.target / "source/web/asset.js").read_text(), "new\n")
        backup = Path(receipt["deployed_files"][0]["backup"])
        self.assertEqual(backup.read_bytes(), old)
        self.assertFalse(receipt["human_truth"])
        self.assertFalse(receipt["approved_for_training"])

    def test_busy_service_blocks_all_writes(self):
        StatusHandler.status = "running"
        before = (self.target / "source/web/asset.js").read_bytes()
        result = self.run_tool("--execute", "--status-url", self.status_url)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not confirmed idle", result.stderr)
        self.assertEqual((self.target / "source/web/asset.js").read_bytes(), before)
        StatusHandler.status = "idle"

    def test_target_hash_drift_blocks_before_write(self):
        target = self.target / "source/web/asset.js"
        target.write_text("unexpected\n", encoding="utf-8")
        before = target.read_bytes()
        result = self.run_tool("--execute", "--status-url", self.status_url)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("formal target hash mismatch", result.stderr)
        self.assertEqual(target.read_bytes(), before)

    def test_second_replace_failure_rolls_back_first_file(self):
        rows = []
        for name in ("one.js", "two.js"):
            source = self.candidate / "files/source/web" / name
            target = self.target / "source/web" / name
            source.write_text(f"new-{name}\n", encoding="utf-8")
            target.write_text(f"old-{name}\n", encoding="utf-8")
            rows.append({
                "relative": f"source/web/{name}",
                "source": source,
                "target": target,
                "old_sha256": digest(target),
                "new_sha256": digest(source),
                "existed_before": True,
                "needs_update": True,
            })
        before = {row["relative"]: row["target"].read_bytes() for row in rows}
        real_replace = deployer.os.replace
        calls = 0

        def fail_second(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected second replace failure")
            return real_replace(source, target)

        with patch.object(deployer.os, "replace", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "injected second replace failure"):
                deployer.deploy(rows, self.target, self.root / "rollback-backup")
        self.assertEqual(
            before,
            {row["relative"]: row["target"].read_bytes() for row in rows},
        )


if __name__ == "__main__":
    unittest.main()
