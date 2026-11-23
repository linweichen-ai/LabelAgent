import hashlib
import json
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
