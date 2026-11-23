import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "verify_formal_minimal_overlay.py"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FixtureHandler(BaseHTTPRequestHandler):
    mode = "ok"

    def do_GET(self):
        if self.path == "/api/run/status":
            payload = {"status": "idle"}
        elif self.path == "/api/pallet-candidate-review":
            payload = {"available": False, "read_only": True, "production_deployed": False,
                       "human_truth_writes": 0, "training_data_writes": 0}
        elif self.path == "/api/review_packets/slope-round160":
            rows = []
            for frame, rank in ((95, 1), (140, 2)):
                rows.append({"frame": frame, "review_priority": {
                    "priority_rank": rank, "assistant_only": True, "human_truth": False,
                    "approved_for_training": False, "auto_submit_allowed": False}})
            payload = {"rows": rows, "priority_reviews_loaded": 2,
                       "read_only_packet": True, "submits_human_truth": False,
                       "summary": {"human_truth_frames": 0, "training_approved_frames": 0}}
            if self.mode == "bad_boundary":
                payload["rows"][0]["review_priority"]["human_truth"] = True
        else:
            self.send_error(404)
            return
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


class FormalOverlayVerificationTests(unittest.TestCase):
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
        target.write_bytes(source.read_bytes())
        manifest = {
            "schema_version": "labelagent.formal-minimal-overlay-candidate.v1",
            "candidate_validated": True, "deployment_performed": False,
            "formal_files_written": 0,
            "files": [{"relative": "source/web/asset.js",
                       "formal_sha256": hashlib.sha256(b"old\n").hexdigest(),
                       "candidate_sha256": digest(source)}],
        }
        (self.candidate / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        FixtureHandler.mode = "ok"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.temp.cleanup()

    def run_tool(self):
        return subprocess.run(
            ["python", str(SCRIPT), "--candidate-root", str(self.candidate),
             "--target-root", str(self.target), "--base-url", self.base],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)

    def test_verified_postdeploy_state_passes_without_writes(self):
        before = (self.target / "source/web/asset.js").read_bytes()
        result = self.run_tool()
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads(result.stdout)
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["priority_frames"], [95, 140])
        self.assertEqual(receipt["write_requests"], 0)
        self.assertEqual((self.target / "source/web/asset.js").read_bytes(), before)

    def test_old_target_is_rejected_before_endpoint_acceptance(self):
        (self.target / "source/web/asset.js").write_bytes(b"old\n")
        result = self.run_tool()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not fully deployed", result.stderr)

    def test_truth_boundary_drift_is_rejected(self):
        FixtureHandler.mode = "bad_boundary"
        result = self.run_tool()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("candidate-only boundaries", result.stderr)


if __name__ == "__main__":
    unittest.main()
