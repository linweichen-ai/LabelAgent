import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools.build_pallet_roi_shadow_review import build_review


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PalletRoiShadowReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.snapshot = self.root / "snapshot"
        self.scan = self.root / "scan"
        self.snapshot.mkdir(); self.scan.mkdir()
        source = self.snapshot / "frame-000002.png"
        Image.new("RGB", (100, 80), (80, 100, 120)).save(source)
        snapshot_manifest = {
            "schema_version": "labelagent.workbench-frame-snapshot.v1",
            "images": [{"frame": 2, "image": source.name,
                        "sha256": sha(source), "bytes": source.stat().st_size}],
        }
        snapshot_path = self.snapshot / "manifest.json"
        snapshot_path.write_text(json.dumps(snapshot_manifest))
        before = np.zeros((80, 100), dtype=np.uint8)
        before[20:70, 20:80] = 255
        after = before.copy(); after[18:20, 45:50] = 255
        before_path = self.scan / "before.png"
        after_path = self.scan / "after.png"
        Image.fromarray(before).save(before_path)
        Image.fromarray(after).save(after_path)
        scan_manifest = {
            "schema_version": "labelagent.pallet-evidence-roi-offline-scan.v2",
            "snapshot_manifest_sha256": sha(snapshot_path),
            "candidate_only": True, "human_truth": False,
            "training_authorized": False, "deployment_authorized": False,
            "rows": [{
                "frame": 2, "candidate_id": 9, "changed_pixels": 10,
                "before_mask": before_path.name,
                "before_mask_sha256": sha(before_path),
                "after_mask": after_path.name,
                "after_mask_sha256": sha(after_path),
            }],
        }
        (self.scan / "manifest.json").write_text(json.dumps(scan_manifest))

    def test_builds_hash_bound_candidate_only_contact_sheet(self):
        output = self.root / "review"
        result = build_review(self.snapshot, self.scan, output)
        manifest = json.loads((output / "manifest.json").read_text())

        self.assertEqual(result["changed_candidate_count"], 1)
        self.assertEqual(manifest["changed_candidate_count"], 1)
        self.assertFalse(manifest["human_truth"])
        self.assertFalse(manifest["training_authorized"])
        self.assertFalse(manifest["deployment_authorized"])
        self.assertEqual(manifest["geometry_policy"],
                         "appearance_evidence_roi")
        self.assertFalse(manifest["workbench_write"])
        self.assertEqual(sha(output / "contact-sheet.jpg"),
                         manifest["contact_sheet_sha256"])

    def test_mask_hash_drift_fails_without_output(self):
        (self.scan / "before.png").write_bytes(b"drift")
        output = self.root / "drift"

        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            build_review(self.snapshot, self.scan, output)
        self.assertFalse(output.exists())

    def test_builds_lossless_detail_sheet_around_added_pixels(self):
        output = self.root / "details"
        result = build_review(
            self.snapshot, self.scan, output, include_details=True,
            include_components=True)
        manifest = json.loads((output / "manifest.json").read_text())
        row = manifest["rows"][0]
        detail = output / row["detail_sheet"]

        self.assertEqual(result["detail_sheet_count"], 1)
        self.assertEqual(result["component_sheet_count"], 1)
        self.assertTrue(manifest["detail_sheets_included"])
        self.assertEqual(sha(detail), row["detail_sheet_sha256"])
        with Image.open(detail) as image:
            self.assertEqual(image.size, (2560, 480))
            self.assertEqual(image.format, "PNG")
        self.assertEqual(len(row["detail_crop_box"]), 4)
        component = row["component_sheets"][0]
        component_path = output / component["detail_sheet"]
        self.assertEqual(component["area"], 10)
        self.assertTrue(component["standalone_component_count_preserved"])
        self.assertGreater(component["contact_ratio"], 0)
        self.assertEqual(sha(component_path), component["detail_sheet_sha256"])
        with Image.open(component_path) as image:
            self.assertEqual(image.size, (2560, 480))
            self.assertEqual(image.format, "PNG")

    def test_builds_one_sheet_per_connected_added_component(self):
        after_path = self.scan / "after.png"
        after = np.asarray(Image.open(after_path).convert("L")).copy()
        after[40:43, 18:20] = 255
        Image.fromarray(after).save(after_path)
        manifest_path = self.scan / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["rows"][0]["changed_pixels"] = 16
        manifest["rows"][0]["after_mask_sha256"] = sha(after_path)
        manifest_path.write_text(json.dumps(manifest))

        output = self.root / "components"
        result = build_review(
            self.snapshot, self.scan, output, include_components=True)
        review = json.loads((output / "manifest.json").read_text())
        components = review["rows"][0]["component_sheets"]

        self.assertEqual(result["component_sheet_count"], 2)
        self.assertEqual([item["area"] for item in components], [10, 6])
        self.assertEqual(sum(item["area"] for item in components), 16)
        self.assertTrue(all(
            item["standalone_component_count_preserved"]
            for item in components))

    def test_disconnected_added_component_fails_closed(self):
        after_path = self.scan / "after.png"
        after = np.asarray(Image.open(after_path).convert("L")).copy()
        after[4:6, 4:6] = 255
        Image.fromarray(after).save(after_path)
        manifest_path = self.scan / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["rows"][0]["changed_pixels"] = 14
        manifest["rows"][0]["after_mask_sha256"] = sha(after_path)
        manifest_path.write_text(json.dumps(manifest))
        output = self.root / "disconnected"

        with self.assertRaisesRegex(
                ValueError, "foreground component count changed"):
            build_review(
                self.snapshot, self.scan, output, include_components=True)
        self.assertFalse(output.exists())

    def test_balanced_bridge_and_island_hits_independent_component_gate(self):
        before_path = self.scan / "before.png"
        after_path = self.scan / "after.png"
        before = np.zeros((80, 100), dtype=np.uint8)
        before[20:40, 20:35] = 255
        before[20:40, 40:55] = 255
        after = before.copy()
        after[28:32, 35:40] = 255
        after[60:64, 70:75] = 255
        Image.fromarray(before).save(before_path)
        Image.fromarray(after).save(after_path)
        manifest_path = self.scan / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["rows"][0]["changed_pixels"] = 40
        manifest["rows"][0]["before_mask_sha256"] = sha(before_path)
        manifest["rows"][0]["after_mask_sha256"] = sha(after_path)
        manifest_path.write_text(json.dumps(manifest))
        output = self.root / "balanced-bridge-and-island"

        with self.assertRaisesRegex(
                ValueError, "not independently connected"):
            build_review(
                self.snapshot, self.scan, output, include_components=True)
        self.assertFalse(output.exists())

    def test_deletion_and_symlink_escape_fail_closed(self):
        after_path = self.scan / "after.png"
        after = np.asarray(Image.open(after_path).convert("L")).copy()
        after[30, 30] = 0
        Image.fromarray(after).save(after_path)
        manifest_path = self.scan / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["rows"][0]["after_mask_sha256"] = sha(after_path)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "not add-only"):
            build_review(self.snapshot, self.scan, self.root / "deletion")

        before_path = self.scan / "before.png"
        outside = self.root / "outside.png"
        outside.write_bytes(before_path.read_bytes())
        before_path.unlink()
        try:
            before_path.symlink_to(outside)
        except OSError:
            original_resolve = Path.resolve

            def resolve_as_escape(path, *args, **kwargs):
                if path == before_path:
                    return outside
                return original_resolve(path, *args, **kwargs)

            with patch.object(Path, "resolve", new=resolve_as_escape):
                with self.assertRaisesRegex(ValueError, "path escape"):
                    build_review(
                        self.snapshot, self.scan, self.root / "simulated-link")
            return
        with self.assertRaisesRegex(ValueError, "path escape"):
            build_review(self.snapshot, self.scan, self.root / "link")

    def test_duplicate_changed_candidate_identity_fails_closed(self):
        manifest_path = self.scan / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["rows"].append(dict(manifest["rows"][0]))
        manifest_path.write_text(json.dumps(manifest))
        output = self.root / "duplicate"

        with self.assertRaisesRegex(ValueError, "duplicate changed candidate"):
            build_review(
                self.snapshot, self.scan, output, include_details=True)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
