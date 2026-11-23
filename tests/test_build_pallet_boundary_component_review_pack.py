import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.build_pallet_boundary_component_review_pack import build_pack


def png_bytes(color=(20, 30, 40)):
    stream = io.BytesIO()
    Image.new("RGB", (64, 32), color).save(stream, "PNG")
    return stream.getvalue()


class BuildPalletBoundaryComponentReviewPackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        image = png_bytes()
        (self.source / "ownership.png").write_bytes(image)
        self.value = {
            "schema_version": "labelagent.pallet-roi-shadow-review.v1",
            "candidate_only": True,
            "human_truth": False,
            "training_authorized": False,
            "deployment_authorized": False,
            "workbench_write": False,
            "component_context_ownership_included": True,
            "rows": [{
                "frame": 15,
                "candidate_id": 3,
                "component_sheets": [{
                    "component_index": 1,
                    "area": 9,
                    "context_ownership": True,
                    "context_locator": True,
                    "context_detail_sheet": "ownership.png",
                    "context_detail_sheet_sha256": hashlib.sha256(image).hexdigest(),
                }],
            }],
        }
        self.manifest = self.source / "manifest.json"
        self._write()

    def _write(self):
        self.manifest.write_text(json.dumps(self.value), encoding="utf-8")

    def test_builds_deterministic_read_only_generic_pack(self):
        first = self.root / "first"
        second = self.root / "second"
        result = build_pack([self.manifest], first)
        repeat = build_pack([self.manifest], second)
        self.assertEqual(result["manifest_sha256"], repeat["manifest_sha256"])
        self.assertEqual(result["review_count"], 1)
        packet = json.loads((first / "focused_review_manifest.json").read_text(
            encoding="utf-8"))
        self.assertEqual(packet["schema_version"],
                         "labelagent.component-review-pack.v2")
        self.assertFalse(packet["human_truth"])
        self.assertFalse(packet["approved_for_training"])
        self.assertFalse(packet["automatic_decisions_enabled"])
        self.assertEqual(packet["rows"][0]["review_rank"], 1)
        self.assertIsNone(packet["rows"][0]["automatic_decision"])
        self.assertEqual(
            (first / packet["rows"][0]["focused_image"]).read_bytes(),
            (second / packet["rows"][0]["focused_image"]).read_bytes())

    def test_rejects_promoted_source_and_hash_drift(self):
        self.value["training_authorized"] = True
        self._write()
        with self.assertRaisesRegex(ValueError, "safety gates"):
            build_pack([self.manifest], self.root / "unsafe")
        self.value["training_authorized"] = False
        self.value["rows"][0]["component_sheets"][0][
            "context_detail_sheet_sha256"] = "0" * 64
        self._write()
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            build_pack([self.manifest], self.root / "drift")

    def test_rejects_path_escape(self):
        outside = self.root / "outside.png"
        outside.write_bytes(png_bytes((1, 2, 3)))
        component = self.value["rows"][0]["component_sheets"][0]
        component["context_detail_sheet"] = "../outside.png"
        component["context_detail_sheet_sha256"] = hashlib.sha256(
            outside.read_bytes()).hexdigest()
        self._write()
        with self.assertRaisesRegex(ValueError, "binding"):
            build_pack([self.manifest], self.root / "escape")


if __name__ == "__main__":
    unittest.main()
