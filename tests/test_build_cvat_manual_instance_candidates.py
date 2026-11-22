import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.build_cvat_manual_instance_candidates import build


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shape(source: str, shape_id: int) -> dict:
    return {
        "id": shape_id,
        "label": "pallet",
        "points": [[0, 0], [1, 0], [1, 1]],
        "holes": [],
        "parts": [{"points": [[0, 0], [1, 0], [1, 1]], "holes": []}],
        "cvat_metadata": {"source": source},
    }


class ManualInstanceCandidateBuilderTests(unittest.TestCase):
    def fixture(self, root: Path):
        image_root = root / "images"
        image_root.mkdir()
        payload = b"small-image-fixture"
        (image_root / "frame.jpg").write_bytes(payload)
        truth = root / "truth.json"
        truth.write_text(
            json.dumps({"0": [shape("manual", 10), shape("auto", 11)]}),
            encoding="utf-8",
        )
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "frames": [
                        {
                            "frame": 0,
                            "file": "frame.jpg",
                            "sha256": digest(payload),
                            "source_name": "source.png",
                            "source_timestamp_ns": "1",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return truth, manifest, image_root

    def test_only_manual_shape_becomes_positive_instance_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            private, receipt = build(*self.fixture(Path(directory)))
        self.assertEqual(private["status"], "prepared_not_training_authorized")
        self.assertEqual(private["summary"]["candidate_count"], 1)
        self.assertEqual(private["candidates"][0]["shape"]["id"], 10)
        self.assertTrue(private["candidates"][0]["supervision"]["positive_instance_only"])
        self.assertEqual(private["candidates"][0]["supervision"]["outside_instance"], "ignore")
        self.assertFalse(receipt["policy"]["training_authorized"])
        self.assertFalse(receipt["private_geometry_published"])

    def test_image_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth, manifest, image_root = self.fixture(root)
            value = json.loads(manifest.read_text())
            value["frames"][0]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(value))
            private, _ = build(truth, manifest, image_root)
        self.assertEqual(private["status"], "failed_closed")
        self.assertIn("frame 0: image digest differs", private["errors"])

    def test_missing_frame_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth, manifest, image_root = self.fixture(root)
            manifest.write_text(json.dumps({"frames": []}))
            private, _ = build(truth, manifest, image_root)
        self.assertEqual(private["status"], "failed_closed")
        self.assertIn("frame 0: absent from import manifest", private["errors"])


if __name__ == "__main__":
    unittest.main()
