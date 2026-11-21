import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.build_ego_component_union_review import file_sha256
from tools.build_ego_identity_manifest import build_manifest


class EgoIdentityManifestTests(unittest.TestCase):
    def _fixture(self, root: Path):
        existing = root / "existing.json"
        existing.write_text(json.dumps({"samples": {
            "000-camera1-100": {
                "package": "000", "camera": "camera1", "frame_id": "100",
                "image_sha256": "0" * 64,
            }}}), encoding="utf-8")
        source = root / "source"
        source.mkdir()
        image = source / "003-camera1-idx22-12345.png"
        Image.new("RGB", (10, 8), (70, 80, 90)).save(image)
        diagnostic = root / "diagnostic" / "003-camera1-idx22-12345"
        diagnostic.mkdir(parents=True)
        report = {
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "production_deployed": False,
            "source_state_modified": False,
            "image": str(image),
            "image_sha256": file_sha256(image),
        }
        (diagnostic / "results.json").write_text(
            json.dumps(report), encoding="utf-8")
        return existing, diagnostic.parent, source, report

    def test_merges_flat_diagnostic_identity_with_relative_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing, diagnostic, source, report = self._fixture(root)
            payload = build_manifest(
                existing, [(diagnostic, source)], root / "output.json", "dataset")
            self.assertEqual(payload["sample_count"], 2)
            entry = payload["samples"]["003-camera1-idx22-12345"]
            self.assertEqual(entry["relative_path"],
                             "003-camera1-idx22-12345.png")
            self.assertEqual(entry["image_sha256"], report["image_sha256"])
            self.assertFalse(payload["human_truth"])
            self.assertFalse(payload["approved_for_training"])

    def test_rejects_source_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing, diagnostic, source, _ = self._fixture(root)
            path = diagnostic / "003-camera1-idx22-12345" / "results.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["image_sha256"] = "f" * 64
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash-mismatched"):
                build_manifest(
                    existing, [(diagnostic, source)], root / "output.json", "dataset")

    def test_rejects_source_outside_allowed_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing, diagnostic, _, _ = self._fixture(root)
            allowed = root / "allowed"
            allowed.mkdir()
            with self.assertRaisesRegex(ValueError, "outside allowed root"):
                build_manifest(
                    existing, [(diagnostic, allowed)], root / "output.json", "dataset")

    def test_rejects_inconsistent_safety_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing, diagnostic, source, _ = self._fixture(root)
            path = diagnostic / "003-camera1-idx22-12345" / "results.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["approved_for_training"] = True
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "safety flags"):
                build_manifest(
                    existing, [(diagnostic, source)], root / "output.json", "dataset")

    def test_requires_at_least_one_new_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing, _, source, _ = self._fixture(root)
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(ValueError, "no new diagnostic samples"):
                build_manifest(
                    existing, [(empty, source)], root / "output.json", "dataset")


if __name__ == "__main__":
    unittest.main()
