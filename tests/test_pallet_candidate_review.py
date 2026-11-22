import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from source.agent.pallet_candidate_review import (CandidateReviewUnavailable,
                                                   load_review_page)
from source.agent.server import app


class PalletCandidateReviewTest(unittest.TestCase):
    def _fixture(self, root: Path):
        preview = root / "previews" / "scene.png"
        mask = root / "masks" / "candidate.png"
        preview.parent.mkdir(); mask.parent.mkdir()
        Image.new("RGB", (8, 6), "green").save(preview)
        Image.new("L", (8, 6), 255).save(mask)
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        bundle = {
            "schema_version": "labelagent.pallet_visibility_candidate_bundle.v1",
            "production_deployed": False, "human_truth_writes": 0,
            "training_data_writes": 0, "workbench_writes": 0,
            "candidate_count": 2, "preview_count": 1,
            "previews": [{"path": "previews/scene.png", "sha256": sha(preview)}],
            "rows": [
                {"mask_path": "masks/candidate.png", "mask_sha256": sha(mask),
                 "mask_pixels": 48},
                {"mask_path": None, "mask_pixels": 0},
            ],
        }
        (root / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
        (root / "index.html").write_text(
            "<!doctype html><style>body{color:#123}</style>"
            "<p>离线候选 · 未部署 · 不是真值</p>"
            '<img src="previews/scene.png">', encoding="utf-8")

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(CandidateReviewUnavailable):
                load_review_page()
            client = app.test_client()
            self.assertEqual(client.get(
                "/api/pallet-candidate-review").status_code, 404)
            shell = client.get("/")
            self.assertIn(b'id="palletCandidateReviewEntry"', shell.data)
            self.assertIn(b"hidden", shell.data)
            self.assertIn(b"pallet-candidate-review-entry.js", shell.data)

    def test_verified_bundle_is_served_read_only(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); self._fixture(root)
            with patch.dict(os.environ, {
                    "LABELAGENT_PALLET_CANDIDATE_REVIEW_DIR": raw}):
                client = app.test_client()
                status = client.get("/api/pallet-candidate-review")
                self.assertEqual(status.status_code, 200)
                self.assertEqual(status.json["candidate_count"], 2)
                self.assertEqual(status.json["mask_count"], 1)
                page = client.get("/pallet-candidate-review/")
                self.assertEqual(page.status_code, 200)
                self.assertIn("default-src 'none'",
                              page.headers["Content-Security-Policy"])
                self.assertEqual(page.headers["Cache-Control"], "no-store")
                asset = client.get("/pallet-candidate-review/previews/scene.png")
                self.assertEqual(asset.status_code, 200)
                self.assertTrue(asset.data.startswith(b"\x89PNG"))
                self.assertEqual(client.get(
                    "/pallet-candidate-review/unknown.png").status_code, 404)

    def test_tampered_asset_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); self._fixture(root)
            (root / "previews" / "scene.png").write_bytes(b"tampered")
            with patch.dict(os.environ, {
                    "LABELAGENT_PALLET_CANDIDATE_REVIEW_DIR": raw}):
                self.assertEqual(app.test_client().get(
                    "/api/pallet-candidate-review").status_code, 404)


if __name__ == "__main__":
    unittest.main()
