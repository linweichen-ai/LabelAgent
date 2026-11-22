import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.build_sam3_assistant_instance_crop_dataset import build


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AssistantInstanceCropDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        Image.new("RGB", (20, 16), (40, 80, 120)).save(self.images / "frame.jpg")
        candidate = {
            "schema_version": "labelagent.cvat-manual-instance-candidates.v1",
            "status": "prepared_not_training_authorized",
            "policy": {
                "human_truth_claimed": False,
                "training_authorized": False,
                "test24_included": False,
                "auto_shapes_included": False,
                "outside_manual_instance": "ignore",
                "full_frame_semantic_loss_allowed": False,
                "negative_sampling_allowed": False,
            },
            "candidates": [
                {
                    "candidate_id": "1:2",
                    "image": {"file": "frame.jpg", "sha256": digest(self.images / "frame.jpg")},
                    "shape": {
                        "label": "pallet",
                        "points": [[5, 4], [12, 4], [12, 10], [5, 10]],
                        "holes": [],
                        "parts": [],
                    },
                    "supervision": {
                        "positive_instance_only": True,
                        "outside_instance": "ignore",
                        "full_frame_semantic_loss_allowed": False,
                        "negative_sampling_allowed": False,
                    },
                }
            ],
        }
        self.candidates = self.root / "candidates.json"
        self.candidates.write_text(json.dumps(candidate), encoding="utf-8")
        review = {
            "schema_version": "labelagent.assistant-delegated-instance-review.v1",
            "candidate_manifest_sha256": digest(self.candidates),
            "reviewer_kind": "independent_assistant_visual_review",
            "human_truth": False,
            "training_authorized": False,
            "automatic_approval": False,
            "assistant_pass_count": 1,
            "decisions": [{"candidate_id": "1:2", "decision": "pass"}],
        }
        self.review = self.root / "review.json"
        self.review.write_text(json.dumps(review), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_builds_hash_bound_positive_crop(self):
        manifest, receipt = build(
            self.candidates, self.review, self.images, self.root / "out"
        )
        self.assertEqual(manifest["crop_count"], 1)
        self.assertFalse(manifest["human_truth"])
        self.assertFalse(manifest["test24_included"])
        self.assertFalse(receipt["training_authorized"])
        coco = json.loads((self.root / "out/_assistant_delegated_pallet_crops.coco.json").read_text())
        self.assertEqual(coco["categories"][0]["name"], "pallet")
        self.assertGreater(coco["annotations"][0]["area"], 0)
        rle = coco["annotations"][0]["segmentation"]
        self.assertEqual(sum(rle["counts"]), rle["size"][0] * rle["size"][1])

    def test_rejects_review_hash_drift(self):
        review = json.loads(self.review.read_text())
        review["candidate_manifest_sha256"] = "0" * 64
        self.review.write_text(json.dumps(review), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not bind"):
            build(self.candidates, self.review, self.images, self.root / "out")

    def test_rejects_human_truth_claim(self):
        review = json.loads(self.review.read_text())
        review["human_truth"] = True
        self.review.write_text(json.dumps(review), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "human_truth"):
            build(self.candidates, self.review, self.images, self.root / "out")


if __name__ == "__main__":
    unittest.main()


