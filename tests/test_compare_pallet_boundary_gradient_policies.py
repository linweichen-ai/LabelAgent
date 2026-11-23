import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools.compare_pallet_boundary_gradient_policies import compare


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


class PalletBoundaryGradientPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scan = self.root / "scan"
        self.scan.mkdir()
        pairs = [(10, 20), (25, 20), (35, 20), (70, 80)]
        rows = []
        for index, (notch, ring) in enumerate(pairs, start=1):
            before = np.zeros((8, 8), dtype=np.uint8)
            before[1:7, 1:7] = 255
            before[3, index] = 0
            after = before.copy()
            after[3, index] = 255
            before_name = f"before-{index}.png"
            after_name = f"after-{index}.png"
            Image.fromarray(before).save(self.scan / before_name)
            Image.fromarray(after).save(self.scan / after_name)
            before_payload = (self.scan / before_name).read_bytes()
            after_payload = (self.scan / after_name).read_bytes()
            rows.append({
                "frame": index,
                "candidate_id": index,
                "changed_pixels": 1,
                "roi_evidence_scale_gate_passed": True,
                "source_image_sha256": f"{index:x}" * 64,
                "candidate_file_sha256": f"{index + 4:x}" * 64,
                "before_mask": before_name,
                "after_mask": after_name,
                "before_mask_sha256": digest(before_payload),
                "after_mask_sha256": digest(after_payload),
                "audit": {
                    "geometry_cleanup_version": 8,
                    "geometry_cleanup_policy": "appearance_evidence_roi",
                    "add_only": True,
                    "component_merge_count": 0,
                    "required_structural_hole_retention": 1.0,
                    "changed_pixels": 1,
                    "filled_image_boundary_notch_count": 1,
                    "filled_image_boundary_notch_pixels": 1,
                    "filled_image_boundary_notch_evidence": [{
                        "area": 1,
                        "notch_gradient_mean": notch,
                        "ring_gradient_mean": ring,
                        "scale_evidence_route": "roi_absolute",
                    }],
                },
            })
        self.manifest = {
            "schema_version": "labelagent.pallet-evidence-roi-offline-scan.v2",
            "geometry_policy": "appearance_evidence_roi",
            "candidate_only": True,
            "human_truth": False,
            "training_authorized": False,
            "deployment_authorized": False,
            "workbench_write": False,
            "snapshot_manifest_sha256": "a" * 64,
            "smoke_summary_sha256": "b" * 64,
            "pallet_candidate_count": 4,
            "roi_evidence_scale_gate_passed_count": 4,
            "changed_candidate_count": 4,
            "rows": rows,
        }
        self.manifest_path = self.scan / "manifest.json"
        self._write_manifest()
        self.map_path = self.root / "input-map.json"
        self._write_map()

    def _write_manifest(self):
        self.manifest_path.write_text(
            json.dumps(self.manifest), encoding="utf-8")

    def _write_map(self):
        mapping = {
            "schema_version": "labelagent.pallet-gradient-policy-input-map.v1",
            "inputs": [{
                "input_id": "fixture",
                "manifest_path": "scan/manifest.json",
                "manifest_sha256": digest(self.manifest_path.read_bytes()),
            }],
        }
        self.map_path.write_text(json.dumps(mapping), encoding="utf-8")

    def _compare(self, name="output"):
        return compare(
            self.map_path, digest(self.map_path.read_bytes()), self.root / name)

    def test_compares_all_notches_with_frozen_policy_counts(self):
        report = self._compare()
        self.assertEqual(report["changed_candidate_count"], 4)
        self.assertEqual(report["policies"]["relative0"]["pass"], 2)
        self.assertEqual(report["policies"]["relative_plus_8"]["pass"], 3)
        self.assertEqual(report["policies"]["absolute64"]["pass"], 3)
        self.assertEqual(report["policies"]["combined"]["pass"], 2)
        self.assertFalse(report["human_truth"])
        self.assertFalse(report["mask_write"])
        payload = (self.root / "output" / "report.json").read_text()
        self.assertNotIn(str(self.root), payload)

    def test_safety_flag_and_nonfinite_gradient_fail_without_output(self):
        self.manifest["human_truth"] = True
        self._write_manifest(); self._write_map()
        with self.assertRaisesRegex(ValueError, "safety flag"):
            self._compare("unsafe")
        self.assertFalse((self.root / "unsafe").exists())

        self.manifest["human_truth"] = False
        self.manifest["rows"][0]["audit"][
            "filled_image_boundary_notch_evidence"][0][
                "notch_gradient_mean"] = float("nan")
        self._write_manifest(); self._write_map()
        with self.assertRaisesRegex(ValueError, "notch gradient invalid"):
            self._compare("nan")
        self.assertFalse((self.root / "nan").exists())

    def test_all_notches_must_pass_candidate_policy(self):
        row = self.manifest["rows"][0]
        before_path = self.scan / row["before_mask"]
        after_path = self.scan / row["after_mask"]
        before = np.asarray(Image.open(before_path).convert("L")).copy()
        after = np.asarray(Image.open(after_path).convert("L")).copy()
        before[4, 1] = 0
        after[4, 1] = 255
        Image.fromarray(before).save(before_path)
        Image.fromarray(after).save(after_path)
        row["before_mask_sha256"] = digest(before_path.read_bytes())
        row["after_mask_sha256"] = digest(after_path.read_bytes())
        row["changed_pixels"] = 2
        row["audit"]["changed_pixels"] = 2
        row["audit"]["filled_image_boundary_notch_count"] = 2
        row["audit"]["filled_image_boundary_notch_pixels"] = 2
        row["audit"]["filled_image_boundary_notch_evidence"].append({
            "area": 1,
            "notch_gradient_mean": 30,
            "ring_gradient_mean": 20,
            "scale_evidence_route": "roi_absolute",
        })
        self._write_manifest(); self._write_map()
        report = self._compare("multi-notch")
        policies = report["rows"][0]["policies"]
        self.assertFalse(policies["relative0"])
        self.assertFalse(policies["relative_plus_8"])
        self.assertTrue(policies["absolute64"])
        self.assertFalse(policies["combined"])

    def test_changed_candidate_without_gradient_is_insufficient(self):
        row = self.manifest["rows"][0]
        row["audit"]["filled_image_boundary_notch_count"] = 0
        row["audit"]["filled_image_boundary_notch_pixels"] = 0
        row["audit"]["filled_image_boundary_notch_evidence"] = []
        self._write_manifest(); self._write_map()
        report = self._compare("insufficient")
        self.assertEqual(report["insufficient_candidate_count"], 1)
        self.assertEqual(report["rows"][0]["status"], "insufficient")
        self.assertIsNone(report["rows"][0]["policies"])

    def test_mask_deletion_and_hash_drift_fail_closed(self):
        row = self.manifest["rows"][0]
        after_path = self.scan / row["after_mask"]
        after = np.zeros((8, 8), dtype=np.uint8)
        after[4, 1] = 255
        Image.fromarray(after).save(after_path)
        row["after_mask_sha256"] = digest(after_path.read_bytes())
        self._write_manifest(); self._write_map()
        with self.assertRaisesRegex(ValueError, "not add-only"):
            self._compare("deletion")

        row["after_mask_sha256"] = "f" * 64
        self._write_manifest(); self._write_map()
        with self.assertRaisesRegex(ValueError, "SHA-256 drift"):
            self._compare("drift")

    def test_component_bridge_and_non_boolean_roi_flag_fail_closed(self):
        row = self.manifest["rows"][0]
        before_path = self.scan / row["before_mask"]
        after_path = self.scan / row["after_mask"]
        before = np.zeros((8, 8), dtype=np.uint8)
        before[4, 2] = 255
        before[4, 4] = 255
        after = before.copy(); after[4, 3] = 255
        Image.fromarray(before).save(before_path)
        Image.fromarray(after).save(after_path)
        row["before_mask_sha256"] = digest(before_path.read_bytes())
        row["after_mask_sha256"] = digest(after_path.read_bytes())
        self._write_manifest(); self._write_map()
        with self.assertRaisesRegex(ValueError, "component count changed"):
            self._compare("bridge")

        row["roi_evidence_scale_gate_passed"] = "false"
        self._write_manifest(); self._write_map()
        with self.assertRaisesRegex(ValueError, "ROI scale flag invalid"):
            self._compare("string-flag")

    def test_mask_symlink_escape_fails_closed(self):
        row = self.manifest["rows"][0]
        before_path = self.scan / row["before_mask"]
        outside = self.root / "outside.png"
        outside.write_bytes(before_path.read_bytes())
        before_path.unlink()
        try:
            before_path.symlink_to(outside)
        except OSError as exc:
            original_resolve = Path.resolve

            def resolve_as_escape(path, *args, **kwargs):
                if path == before_path:
                    return outside
                return original_resolve(path, *args, **kwargs)

            self._write_manifest(); self._write_map()
            with patch.object(Path, "resolve", new=resolve_as_escape):
                with self.assertRaisesRegex(
                        ValueError, "path escapes input root"):
                    self._compare("simulated-symlink")
            return
        self._write_manifest(); self._write_map()
        with self.assertRaisesRegex(ValueError, "path escapes input root"):
            self._compare("symlink")

    def test_duplicate_candidate_identity_across_inputs_fails(self):
        second = self.root / "second"
        second.mkdir()
        for path in self.scan.iterdir():
            (second / path.name).write_bytes(path.read_bytes())
        second_manifest = second / "manifest.json"
        second_payload = json.loads(second_manifest.read_text())
        second_payload["smoke_summary_sha256"] = "c" * 64
        second_manifest.write_text(
            json.dumps(second_payload, indent=2),
            encoding="utf-8")
        mapping = json.loads(self.map_path.read_text())
        mapping["inputs"].append({
            "input_id": "fixture2",
            "manifest_path": "second/manifest.json",
            "manifest_sha256": digest(second_manifest.read_bytes()),
        })
        self.map_path.write_text(json.dumps(mapping), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate candidate identity"):
            self._compare("duplicate")

    def test_input_map_hash_and_existing_output_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "input map SHA-256 drift"):
            compare(self.map_path, "0" * 64, self.root / "bad-map")
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            compare(self.map_path, digest(self.map_path.read_bytes()), output)


if __name__ == "__main__":
    unittest.main()
