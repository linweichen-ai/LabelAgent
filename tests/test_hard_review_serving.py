import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from source.agent import server


class HardReviewServingTest(unittest.TestCase):
    def setUp(self):
        self.previous_work = server.WORK
        self.temporary = tempfile.TemporaryDirectory()
        server.WORK = Path(self.temporary.name)
        self.workspace = server.WORK / "hard_review" / "round-test"
        (self.workspace / "review").mkdir(parents=True)
        files = {
            "reviewer.html": b"<html><script>localStorage.setItem('x','1')</script></html>",
            "review/example.jpg": b"not-a-real-jpeg-but-receipt-bound",
        }
        for relative, data in files.items():
            path = self.workspace / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        receipt = {
            "schema_version": "labelagent.mixed-hard-adjudication-workspace-receipt.v2",
            "files": {relative: hashlib.sha256(data).hexdigest()
                      for relative, data in files.items()},
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "automatic_approval": False,
        }
        (self.workspace / "receipt.json").write_text(
            json.dumps(receipt), encoding="utf-8")
        self.client = server.app.test_client()

    def tearDown(self):
        server.WORK = self.previous_work
        self.temporary.cleanup()

    def test_serves_only_receipt_bound_workspace_files(self):
        page = self.client.get("/hard-review/round-test/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"localStorage", page.data)
        self.assertIn("private, no-store", page.headers["Cache-Control"])
        self.assertIn("script-src 'unsafe-inline'", page.headers["Content-Security-Policy"])
        self.assertIn("connect-src 'none'", page.headers["Content-Security-Policy"])
        image = self.client.get("/hard-review/round-test/review/example.jpg")
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.data, b"not-a-real-jpeg-but-receipt-bound")
        self.assertEqual(image.headers["X-Content-Type-Options"], "nosniff")

    def test_rejects_unlisted_traversal_and_hash_drift(self):
        (self.workspace / "secret.txt").write_text("secret", encoding="utf-8")
        self.assertEqual(self.client.get(
            "/hard-review/round-test/secret.txt").status_code, 404)
        self.assertEqual(self.client.get(
            "/hard-review/../round-test/reviewer.html").status_code, 404)
        (self.workspace / "reviewer.html").write_text("changed", encoding="utf-8")
        self.assertEqual(self.client.get(
            "/hard-review/round-test/").status_code, 404)

    def test_rejects_receipt_that_changes_safety_boundaries(self):
        receipt_path = self.workspace / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["approved_for_training"] = True
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        self.assertEqual(self.client.get(
            "/hard-review/round-test/").status_code, 404)


if __name__ == "__main__":
    unittest.main()
