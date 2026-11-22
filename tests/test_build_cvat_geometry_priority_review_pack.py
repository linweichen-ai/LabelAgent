import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.build_cvat_geometry_priority_review_pack import build


class GeometryPriorityReviewPackTests(unittest.TestCase):
    def fixture(self, root: Path):
        review_root = root / "review"
        (review_root / "items").mkdir(parents=True)
        image = review_root / "items" / "item.jpg"
        Image.new("RGB", (1080, 312), "gray").save(image)
        image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
        review = root / "review.json"
        review.write_text(json.dumps({
            "schema_version": "labelagent.cvat-manual-instance-review-pack.v1",
            "status": "pending_independent_review", "candidate_manifest_sha256": "a" * 64,
            "review_items": [{"candidate_id": "0:1", "focused_image_file": "items/item.jpg",
                              "focused_image_sha256": image_sha}],
        }))
        priority = root / "priority.json"
        priority.write_text(json.dumps({
            "schema_version": "labelagent.cvat-manual-instance-geometry-priority.v1",
            "candidate_manifest_sha256": "a" * 64, "automatic_decisions": False,
            "training_authorized": False,
            "rows": [{"candidate_id": "0:1", "risk_score": 5,
                      "risk_flags": ["touches_image_edge"]}],
        }))
        return review, priority, review_root

    def test_builds_high_priority_page_without_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, receipt = build(*self.fixture(root), root / "output")
            self.assertTrue((root / "output" / manifest["pages"][0]["file"]).is_file())
        self.assertEqual(receipt["priority_counts"]["high"], 1)
        self.assertFalse(receipt["automatic_decisions"])
        self.assertFalse(receipt["training_authorized"])

    def test_hash_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, priority, review_root = self.fixture(root)
            (review_root / "items" / "item.jpg").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                build(review, priority, review_root, root / "output")


if __name__ == "__main__":
    unittest.main()
