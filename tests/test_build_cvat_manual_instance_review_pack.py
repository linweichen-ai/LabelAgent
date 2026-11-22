import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.build_cvat_manual_instance_review_pack import build_pack


class ManualInstanceReviewPackTests(unittest.TestCase):
    def fixture(self, root: Path, source: str = "manual", count: int = 81):
        image_root = root / "images"
        image_root.mkdir()
        manifest = root / "candidates.json"
        candidates = []
        for index in range(count):
            image_path = image_root / f"frame_{index:03d}.png"
            Image.new("RGB", (64, 48), (index % 251, 130, 140)).save(image_path)
            image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
            candidates.append({
                "candidate_id": f"{index}:{10 + index}",
                "frame_id": index,
                "image": {"file": image_path.name, "sha256": image_sha},
                "shape": {
                    "id": 10 + index, "label": "pallet",
                    "points": [[10, 10], [40, 10], [40, 35], [10, 35]],
                    "parts": [{"points": [[10, 10], [40, 10], [40, 35], [10, 35]], "holes": []}],
                    "cvat_metadata": {"source": source},
                },
                "supervision": {"outside_instance": "ignore"},
            })
        manifest.write_text(json.dumps({
            "schema_version": "labelagent.cvat-manual-instance-candidates.v1",
            "status": "prepared_not_training_authorized", "errors": [],
            "policy": {
                "training_authorized": False,
                "test24_included": False,
                "auto_shapes_included": False,
                "outside_manual_instance": "ignore",
                "full_frame_semantic_loss_allowed": False,
                "negative_sampling_allowed": False,
            },
            "candidates": candidates,
        }), encoding="utf-8")
        receipt = root / "receipt.json"
        receipt.write_text(json.dumps({
            "schema_version": "labelagent.cvat-manual-instance-candidate-receipt.v1",
            "status": "prepared_not_training_authorized", "errors": [],
            "private_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "summary": {"candidate_count": count, "frame_count": 81,
                        "verified_image_count": 81, "label_counts": {"pallet": 81}},
        }), encoding="utf-8")
        evaluation = root / "test24.jsonl"
        evaluation.write_text(json.dumps({
            "split": "test24", "test24": True, "evaluation_only": True,
            "training_return_allowed": False, "image_sha256": "f" * 64,
        }) + "\n", encoding="utf-8")
        return manifest, receipt, evaluation, image_root

    def test_builds_private_pending_review_with_thin_outline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, candidate_receipt, evaluation, image_root = self.fixture(root)
            review, receipt = build_pack(manifest, candidate_receipt, evaluation, image_root, root / "review")
            page = root / "review" / review["pages"][0]["file"]
            self.assertTrue(page.is_file())
        self.assertEqual(review["line_width_pixels"], 2)
        self.assertFalse(review["training_authorized"])
        self.assertFalse(receipt["private_review_images_published"])
        self.assertEqual(receipt["test24_intersection_count"], 0)
        self.assertEqual(len(review["review_items"]), 81)

    def test_rejects_non_manual_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, candidate_receipt, evaluation, image_root = self.fixture(root, source="auto")
            with self.assertRaisesRegex(ValueError, "source is not manual"):
                build_pack(manifest, candidate_receipt, evaluation, image_root, root / "review")

    def test_rejects_policy_that_allows_negative_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, candidate_receipt, evaluation, image_root = self.fixture(root)
            value = json.loads(manifest.read_text())
            value["policy"]["negative_sampling_allowed"] = True
            manifest.write_text(json.dumps(value))
            receipt_value = json.loads(candidate_receipt.read_text())
            receipt_value["private_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            candidate_receipt.write_text(json.dumps(receipt_value))
            with self.assertRaisesRegex(ValueError, "negative_sampling_allowed"):
                build_pack(manifest, candidate_receipt, evaluation, image_root, root / "review")

    def test_rejects_candidate_manifest_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, candidate_receipt, evaluation, image_root = self.fixture(root)
            manifest.write_text(manifest.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                build_pack(manifest, candidate_receipt, evaluation, image_root, root / "review")


if __name__ == "__main__":
    unittest.main()
