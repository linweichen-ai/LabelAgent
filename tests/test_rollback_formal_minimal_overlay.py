import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "rollback_formal_minimal_overlay.py"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class IdleHandler(BaseHTTPRequestHandler):
    status = "idle"
    def do_GET(self):
        body = json.dumps({"status": self.status}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *_):
        pass


class FormalOverlayRollbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.target = self.root / "formal"
        self.backup = self.target / ".local/deploy-backups/run"
        old = self.backup / "source/web/existing.js"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"old\n")
        existing = self.target / "source/web/existing.js"
        created = self.target / "source/web/created.js"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_bytes(b"new-existing\n")
        created.write_bytes(b"new-created\n")
        receipt = {
            "schema_version": "labelagent.formal-minimal-overlay-deploy-receipt.v1",
            "status": "deployed", "explicit_execute": True,
            "backup_root": str(self.backup.resolve()),
            "deployed_files": [
                {"relative": "source/web/existing.js", "old_sha256": digest(old),
                 "new_sha256": digest(existing), "backup": str(old.resolve())},
                {"relative": "source/web/created.js", "old_sha256": None,
                 "new_sha256": digest(created), "backup": None},
            ],
        }
        self.receipt = self.backup / "deploy-receipt.json"
        self.receipt.write_text(json.dumps(receipt), encoding="utf-8")
        IdleHandler.status = "idle"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), IdleHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/api/run/status"

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.temp.cleanup()

    def run_tool(self, *extra):
        return subprocess.run(["python", str(SCRIPT), "--receipt", str(self.receipt),
                               "--target-root", str(self.target), *extra],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=20)

    def test_default_dry_run_does_not_restore(self):
        before = (self.target / "source/web/existing.js").read_bytes()
        result = self.run_tool()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "dry_run")
        self.assertEqual((self.target / "source/web/existing.js").read_bytes(), before)

    def test_execute_restores_old_and_removes_new_file(self):
        result = self.run_tool("--execute", "--status-url", self.url)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "rolled_back")
        self.assertEqual((self.target / "source/web/existing.js").read_bytes(), b"old\n")
        self.assertFalse((self.target / "source/web/created.js").exists())
        self.assertTrue((self.backup / "rollback-receipt.json").is_file())

    def test_busy_service_blocks_all_restore_writes(self):
        IdleHandler.status = "running"
        before = (self.target / "source/web/existing.js").read_bytes()
        result = self.run_tool("--execute", "--status-url", self.url)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not confirmed idle", result.stderr)
        self.assertEqual((self.target / "source/web/existing.js").read_bytes(), before)

    def test_postdeploy_drift_is_rejected(self):
        target = self.target / "source/web/existing.js"
        target.write_bytes(b"unexpected\n")
        result = self.run_tool("--execute", "--status-url", self.url)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("drifted after deployment", result.stderr)
        self.assertEqual(target.read_bytes(), b"unexpected\n")


if __name__ == "__main__":
    unittest.main()
