import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.build_mixed_hard_adjudication_workspace import (
    SUBMISSION_SCHEMA, build_workspace, sha256, validate_packet,
    validate_submission,
)


class MixedHardAdjudicationWorkspaceTest(unittest.TestCase):
    def packet(self, root):
        root = Path(root)
        (root / "raw").mkdir(parents=True)
        (root / "review").mkdir()
        rows, files = [], []
        for index in range(24):
            raw = root / "raw" / f"{index}.jpg"
            review = root / "review" / f"{index}.jpg"
            Image.new("RGB", (8, 8), (index, 0, 0)).save(raw)
            Image.new("RGB", (8, 8), (0, index, 0)).save(review)
            for path in (raw, review):
                files.append({"path": path.relative_to(root).as_posix(),
                              "sha256": sha256(path), "bytes": path.stat().st_size})
            rows.append({"sample_id": f"sample-{index}", "cohort": "slope_negative_candidate",
                         "source_image_sha256": f"{index + 1:064x}",
                         "raw_image": f"raw/{index}.jpg", "review_image": f"review/{index}.jpg",
                         "review_note": "fixture", "candidate_only": True,
                         "human_truth": False, "approved_for_training": False})
        manifest = {"schema_version": "labelagent.mixed-hard-review-pack.v1",
                    "summary": {"test24_intersection_count": 0}, "rows": rows,
                    "candidate_only": True, "human_truth": False,
                    "approved_for_training": False, "training_authorized": False,
                    "test24_used": False, "production_deployed": False}
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (root / "receipt.json").write_text(json.dumps({
            "manifest_sha256": sha256(manifest_path), "files": files}), encoding="utf-8")
        return root

    def submission(self, manifest, digest):
        return {"schema_version": SUBMISSION_SCHEMA,
                "review_manifest_sha256": digest, "reviewer_declared_id": "reviewer-1",
                "rows": [{"sample_id": row["sample_id"],
                          "source_image_sha256": row["source_image_sha256"],
                          "candidate_decision": "uncertain", "slope_presence": "absent",
                          "class_conflict": "none", "pallet_quality": "not_applicable"}
                         for row in manifest["rows"]],
                "candidate_only": True, "human_truth": False,
                "approved_for_training": False, "training_authorized": False,
                "automatic_approval": False}

    def test_builds_hash_bound_local_reviewer_without_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            packet = self.packet(Path(directory) / "packet")
            output = Path(directory) / "workspace"
            receipt = build_workspace(packet, output)
            self.assertFalse(receipt["human_truth"])
            page = (output / "reviewer.html").read_text(encoding="utf-8")
            self.assertIn("localStorage", page)
            self.assertIn("mixed-hard-adjudication.json", page)
            self.assertIn("human_truth:false", page)
            self.assertNotIn("approved_for_training:true", page)
            self.assertEqual(receipt["copied_image_count"], 48)
            for relative in ("raw/0.jpg", "review/0.jpg", "raw/23.jpg", "review/23.jpg"):
                copied = output / relative
                self.assertTrue(copied.is_file())
                self.assertEqual(sha256(copied), sha256(packet / relative))
                self.assertEqual(receipt["files"][relative], sha256(copied))

    def test_reviewer_image_links_resolve_inside_portable_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            packet = self.packet(Path(directory) / "packet")
            output = Path(directory) / "portable-workspace"
            build_workspace(packet, output)
            page = (output / "reviewer.html").read_text(encoding="utf-8")
            decoded = page.replace("&quot;", '"')
            self.assertIn('"review_image": "review/0.jpg"', decoded)
            self.assertIn('"raw_image": "raw/0.jpg"', decoded)
            self.assertTrue((output / "review" / "0.jpg").is_file())
            self.assertTrue((output / "raw" / "0.jpg").is_file())

    def test_validates_complete_bound_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            packet = self.packet(Path(directory) / "packet")
            manifest, digest = validate_packet(packet)
            result = validate_submission(manifest, digest, self.submission(manifest, digest))
            self.assertEqual(result["sample_count"], 24)
            self.assertFalse(result["human_truth"])

    def test_rejects_training_approval_and_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            packet = self.packet(Path(directory) / "packet")
            manifest, digest = validate_packet(packet)
            submission = self.submission(manifest, digest)
            submission["approved_for_training"] = True
            with self.assertRaisesRegex(ValueError, "approved_for_training"):
                validate_submission(manifest, digest, submission)
            submission = self.submission(manifest, digest)
            submission["rows"][0]["source_image_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "source image binding differs"):
                validate_submission(manifest, digest, submission)


if __name__ == "__main__":
    unittest.main()
