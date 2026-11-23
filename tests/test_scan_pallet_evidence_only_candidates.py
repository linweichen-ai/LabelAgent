import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from source.agent.edit_agent import _shape_from_mask
from tools.scan_pallet_evidence_only_candidates import scan


class PalletEvidenceOnlyOfflineScanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.snapshot = self.root / "snapshot"
        self.candidates = self.root / "candidates"
        self.snapshot.mkdir(); self.candidates.mkdir()
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[245:480, :] = 1
        cv2.circle(mask, (290, 280), 6, 0, thickness=cv2.FILLED)
        image = np.full((480, 640, 3), 24, dtype=np.uint8)
        image[245:480, :] = np.array([205, 190, 170], dtype=np.uint8)
        image[274:287, 284:297] = np.array([198, 183, 164], dtype=np.uint8)
        image_path = self.snapshot / "frame-000002.png"
        Image.fromarray(image).save(image_path)
        image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
        snapshot = {
            "schema_version": "labelagent.workbench-frame-snapshot.v1",
            "images": [{"frame": 2, "image": image_path.name,
                        "sha256": image_sha, "bytes": image_path.stat().st_size}],
        }
        (self.snapshot / "manifest.json").write_text(json.dumps(snapshot))
        # The isolated workflow hashes its canonical loaded frame, which may
        # legitimately differ from the frozen source-file bytes.
        result = {"input_sha256": "1" * 64, "candidate_only": True,
                  "human_truth": False, "approved_for_training": False,
                  "candidates": [_shape_from_mask(mask, "pallet", 4)]}
        candidate_path = self.candidates / "frame-000002-candidate.json"
        candidate_path.write_text(json.dumps(result))
        summary = {
            "schema_version": "labelagent.new_dataset_smoke.v1",
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False,
            "runs": [{"input": str(image_path), "input_sha256": image_sha,
                      "candidate_only": True, "human_truth": False,
                      "approved_for_training": False,
                      "result_json": str(candidate_path),
                      "result_sha256": hashlib.sha256(
                          candidate_path.read_bytes()).hexdigest()}],
        }
        (self.candidates / "summary.json").write_text(json.dumps(summary))

    def test_scans_bound_candidate_and_writes_only_changed_masks(self):
        output = self.root / "output"
        result = scan(self.snapshot, self.candidates, output)
        self.assertEqual(result["frame_count"], 1)
        self.assertEqual(result["pallet_candidate_count"], 1)
        self.assertEqual(result["changed_candidate_count"], 1)
        self.assertEqual(result["image_texture_scale_gate_passed_count"], 1)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertFalse(manifest["workbench_write"])
        self.assertFalse(manifest["training_authorized"])
        self.assertGreater(manifest["rows"][0]["changed_pixels"], 0)
        self.assertTrue((output / manifest["rows"][0]["after_mask"]).exists())

    def test_input_sha_drift_fails_without_output(self):
        (self.snapshot / "frame-000002.png").write_bytes(b"drift")
        output = self.root / "drift-output"
        with self.assertRaisesRegex(ValueError, "SHA-256 drift"):
            scan(self.snapshot, self.candidates, output)
        self.assertFalse(output.exists())

    def test_invalid_workflow_input_sha_fails_closed(self):
        candidate_path = self.candidates / "frame-000002-candidate.json"
        result = json.loads(candidate_path.read_text())
        result["input_sha256"] = "not-a-sha256"
        candidate_path.write_text(json.dumps(result))
        summary_path = self.candidates / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["runs"][0]["result_sha256"] = hashlib.sha256(
            candidate_path.read_bytes()).hexdigest()
        summary_path.write_text(json.dumps(summary))

        with self.assertRaisesRegex(ValueError, "binding or safety flags"):
            scan(self.snapshot, self.candidates, self.root / "input-mismatch")

    def test_extra_smoke_run_or_traversal_image_fails_closed(self):
        summary_path = self.candidates / "summary.json"
        summary = json.loads(summary_path.read_text())
        extra = dict(summary["runs"][0])
        extra["input"] = "frame-000003.png"
        summary["runs"].append(extra)
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "inputs differ"):
            scan(self.snapshot, self.candidates, self.root / "extra-run")

        summary["runs"] = summary["runs"][:1]
        summary_path.write_text(json.dumps(summary))
        manifest_path = self.snapshot / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["images"][0]["image"] = "../frame-000002.png"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "identity invalid"):
            scan(self.snapshot, self.candidates, self.root / "traversal")

    def test_roi_policy_uses_separate_schema_and_audit(self):
        output = self.root / "roi-output"
        result = scan(
            self.snapshot, self.candidates, output,
            geometry_policy="appearance_evidence_roi")
        manifest = json.loads((output / "manifest.json").read_text())

        self.assertEqual(
            result["schema_version"],
            "labelagent.pallet-evidence-roi-offline-scan.v2")
        self.assertEqual(result["geometry_policy"],
                         "appearance_evidence_roi")
        self.assertEqual(manifest["geometry_policy"],
                         "appearance_evidence_roi")
        self.assertTrue(
            manifest["rows"][0]["roi_evidence_scale_gate_passed"])
        self.assertEqual(
            manifest["roi_evidence_scale_gate_passed_count"], 1)
        self.assertGreaterEqual(
            manifest["rows"][0]["roi_image_texture_hole_count"], 1)


if __name__ == "__main__":
    unittest.main()
