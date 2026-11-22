import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from io import BytesIO

from PIL import Image
from tools.build_cvat_human_signoff_reviewer import build as build_real


def build(review, template, review_root, output, *args):
    if len(args) == 1:
        return build_real(review, template, review_root, output,
                          sha(review.read_bytes()), sha(template.read_bytes()), args[0])
    return build_real(review, template, review_root, output, *args)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class OfflineHumanSignoffReviewerTests(unittest.TestCase):
    def fixture(self, root: Path):
        review_root = root / "review"
        items_dir = review_root / "items"
        items_dir.mkdir(parents=True)
        review_items = []
        rows = []
        for index in range(81):
            cid = f"synthetic-{index:03d}"
            buffer = BytesIO()
            Image.new("RGB", (24, 18), (index, 90, 120)).save(buffer, format="JPEG")
            image = buffer.getvalue()
            image_name = f"items/evidence-{index:03d}.jpg"
            (review_root / image_name).write_bytes(image)
            binding = {
                "candidate_id": cid,
                "frame_id": index,
                "source_image_sha256": sha(f"source-{index}".encode()),
                "canonical_geometry_sha256": sha(f"geometry-{index}".encode()),
                "focused_image_sha256": sha(image),
            }
            review_items.append({**binding, "focused_image_file": image_name})
            rows.append({
                **binding,
                "decision": None,
                "reviewer": None,
                "reviewed_at": None,
                "note": None,
            })
        review = {
            "schema_version": "labelagent.cvat-manual-instance-review-pack.v1",
            "status": "pending_independent_review",
            "test24_intersection_count": 0,
            "training_authorized": False,
            "candidate_count": 81,
            "review_items": review_items,
        }
        review_path = root / "review_manifest.json"
        review_path.write_text(json.dumps(review), encoding="utf-8")
        template = {
            "schema_version": "labelagent.cvat-human-instance-signoff.v1",
            "status": "pending_human_review",
            "review_manifest_sha256": sha(review_path.read_bytes()),
            "candidate_count": 81,
            "allowed_decisions": ["pass", "reject", "uncertain"],
            "rows": rows,
            "human_signoff_count": 0,
            "training_authorized": False,
            "automatic_approval": False,
        }
        template_path = root / "signoff_template.json"
        template_path.write_text(json.dumps(template), encoding="utf-8")
        return review_path, template_path, review_root, review, template

    def test_builds_complete_offline_reviewer_without_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, template, review_root, _, _ = self.fixture(root)
            review_sha, template_sha = sha(review.read_bytes()), sha(template.read_bytes())
            repository = root / "repository"
            repository.mkdir()
            output = repository / ".repro" / "review-session"
            result = build(review, template, review_root, output, review_sha, template_sha, repository)
            page = (output / "index.html").read_text(encoding="utf-8")
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(result["candidate_count"], 81)
            self.assertEqual(result["pending_count"], 81)
            self.assertEqual(len(manifest["items"]), 81)
            self.assertIn('type="radio"', page)
            self.assertIn('value="pass"', page)
            self.assertNotRegex(page, r'<input[^>]+checked(?:\\s|>)')
            self.assertNotIn("localStorage", page)
            self.assertNotIn("fetch(", page)
            self.assertNotIn("XMLHttpRequest", page)
            self.assertNotIn("<script src=", page)
            self.assertFalse(result["training_authorized"])
            self.assertFalse(result["human_pass_is_training_authorization"])

    def test_manifest_hash_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, template, review_root, _, value = self.fixture(root)
            value["review_manifest_sha256"] = "0" * 64
            template.write_text(json.dumps(value), encoding="utf-8")
            repository = root / "repo"
            repository.mkdir()
            with self.assertRaisesRegex(ValueError, "manifest hash differs"):
                build(review, template, review_root, repository / ".repro" / "out", repository)

    def test_missing_image_and_hash_drift_fail_closed(self):
        for mutation in ("missing", "drift"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                review, template, review_root, data, _ = self.fixture(root)
                target = review_root / data["review_items"][0]["focused_image_file"]
                if mutation == "missing":
                    target.unlink()
                else:
                    target.write_bytes(b"changed")
                repository = root / "repo"
                repository.mkdir()
                with self.assertRaises(ValueError):
                    build(review, template, review_root, repository / ".repro" / "out", repository)

    def test_path_escape_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, template, review_root, data, _ = self.fixture(root)
            data["review_items"][0]["focused_image_file"] = "../outside.jpg"
            review.write_text(json.dumps(data), encoding="utf-8")
            template_data = json.loads(template.read_text(encoding="utf-8"))
            template_data["review_manifest_sha256"] = sha(review.read_bytes())
            template.write_text(json.dumps(template_data), encoding="utf-8")
            repository = root / "repo"
            repository.mkdir()
            with self.assertRaisesRegex(ValueError, "escapes"):
                build(review, template, review_root, repository / ".repro" / "out", repository)

    def test_id_or_binding_mismatch_fails_closed(self):
        for field in ("candidate_id", "frame_id", "canonical_geometry_sha256", "source_image_sha256", "focused_image_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                review, template, review_root, _, value = self.fixture(root)
                value["rows"][0][field] = "unknown" if field == "candidate_id" else (999 if field == "frame_id" else "0" * 64)
                template.write_text(json.dumps(value), encoding="utf-8")
                repository = root / "repo"
                repository.mkdir()
                with self.assertRaises(ValueError):
                    build(review, template, review_root, repository / ".repro" / "out", repository)

    def test_output_escape_and_existing_destination_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, template, review_root, _, _ = self.fixture(root)
            repository = root / "repo"
            repository.mkdir()
            with self.assertRaisesRegex(ValueError, "child of repository .repro"):
                build(review, template, review_root, root / "outside", repository)
            destination = repository / ".repro" / "existing"
            destination.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "fresh"):
                build(review, template, review_root, destination, repository)

    def test_template_sha_is_frozen_in_output_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, template, review_root, _, _ = self.fixture(root)
            repository = root / "repo"
            repository.mkdir()
            output = repository / ".repro" / "out"
            manifest = build(review, template, review_root, output, repository)
            self.assertEqual(manifest["signoff_template_sha256"], sha(template.read_bytes()))

    def test_template_row_mutation_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, template, review_root, _, value = self.fixture(root)
            value["rows"][0]["reviewed_at"] = "2026-11-22T10:00:00+08:00"
            template.write_text(json.dumps(value), encoding="utf-8")
            repository = root / "repo"
            repository.mkdir()
            with self.assertRaisesRegex(ValueError, "already contains human decisions"):
                build(review, template, review_root, repository / ".repro" / "out", repository)


if __name__ == "__main__":
    unittest.main()
