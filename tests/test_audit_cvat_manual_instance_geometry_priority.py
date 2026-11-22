import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.audit_cvat_manual_instance_geometry_priority import audit


class GeometryPriorityAuditTests(unittest.TestCase):
    def fixture(self, root: Path):
        image_root = root / "images"
        image_root.mkdir()
        image = image_root / "frame.png"
        Image.new("RGB", (64, 48), "gray").save(image)
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        candidate = {
            "candidate_id": "0:10", "image": {"file": image.name, "sha256": digest},
            "shape": {"id": 10, "label": "pallet", "cvat_metadata": {"source": "manual"},
                      "points": [[0, 5], [20, 5], [20, 20], [0, 20]],
                      "parts": [{"points": [[0, 5], [20, 5], [20, 20], [0, 20]], "holes": []}]},
        }
        manifest = root / "candidates.json"
        manifest.write_text(json.dumps({"candidates": [candidate]}))
        manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        receipt = root / "receipt.json"
        receipt.write_text(json.dumps({"private_manifest_sha256": manifest_sha}))
        diversity = root / "diversity.json"
        diversity.write_text(json.dumps({"status": "assistant_triage_only",
                                         "candidate_manifest_sha256": manifest_sha,
                                         "representative_candidate_ids": ["0:10"]}))
        return manifest, receipt, diversity, image_root

    def test_edge_touching_shape_is_prioritized_without_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            private, public = audit(*self.fixture(Path(directory)))
        self.assertIn("touches_image_edge", private["rows"][0]["risk_flags"])
        self.assertGreaterEqual(private["rows"][0]["risk_score"], 2)
        self.assertFalse(public["automatic_decisions"])
        self.assertFalse(public["training_authorized"])

    def test_diversity_manifest_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, receipt, diversity, image_root = self.fixture(root)
            value = json.loads(diversity.read_text())
            value["candidate_manifest_sha256"] = "0" * 64
            diversity.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "different candidate manifest"):
                audit(manifest, receipt, diversity, image_root)


if __name__ == "__main__":
    unittest.main()
