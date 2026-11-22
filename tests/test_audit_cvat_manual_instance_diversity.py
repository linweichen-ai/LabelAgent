import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from tools.audit_cvat_manual_instance_diversity import audit


class ManualInstanceDiversityAuditTests(unittest.TestCase):
    def fixture(self, root: Path):
        image_root = root / "images"
        image_root.mkdir()
        candidates = []
        for index, color in enumerate(((100, 100, 100), (100, 100, 100), (10, 200, 30))):
            path = image_root / f"{index}.png"
            metadata = PngInfo()
            metadata.add_text("fixture-id", str(index))
            Image.new("RGB", (64, 48), color).save(path, pnginfo=metadata)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            candidates.append({
                "candidate_id": f"{index}:{index + 10}", "frame_id": index,
                "image": {"file": path.name, "sha256": digest},
                "shape": {"id": index + 10, "label": "pallet",
                          "cvat_metadata": {"source": "manual"}},
            })
        manifest = root / "candidates.json"
        manifest.write_text(json.dumps({
            "status": "prepared_not_training_authorized", "errors": [],
            "candidates": candidates,
        }), encoding="utf-8")
        receipt = root / "receipt.json"
        receipt.write_text(json.dumps({
            "private_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "errors": [], "summary": {"candidate_count": 3},
        }), encoding="utf-8")
        return manifest, receipt, image_root

    def test_clusters_near_identical_images_without_removing_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            private, public = audit(*self.fixture(Path(directory)))
        self.assertGreaterEqual(private["threshold_hit_count"], 1)
        self.assertLess(private["representative_count"], private["candidate_count"])
        self.assertEqual(private["samples_removed"], 0)
        self.assertFalse(public["training_authorized"])

    def test_hash_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, receipt, image_root = self.fixture(root)
            manifest.write_text(manifest.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                audit(manifest, receipt, image_root)


if __name__ == "__main__":
    unittest.main()
