import json
from pathlib import Path
import unittest

from tools.evaluate_pallet_temporal_visibility import (
    freeze_temporal_inputs, projected_overlap, temporal_input_paths,
    temporal_identity_counts, temporal_occlusion_gate, temporal_presence_gate,
    verify_temporal_inputs,
    visible_point_median,
)


class PalletTemporalVisibilityTest(unittest.TestCase):
    def test_temporal_gate_requires_both_sides_and_visibility_recovery(self):
        self.assertTrue(temporal_occlusion_gate(0, 2, 1, 11, 11, 0, 16))
        self.assertFalse(temporal_occlusion_gate(0, 2, 0, 11, 11, 0, 16))
        self.assertFalse(temporal_occlusion_gate(0, 2, 1, 10, 11, 0, 16))
        self.assertFalse(temporal_occlusion_gate(0, 2, 1, 11, 11, 10, 12))

    def test_projected_overlap_handles_missing_projection(self):
        self.assertEqual(projected_overlap(None, None), 0.0)

    def test_visible_point_median_is_fail_closed_when_partition_empty(self):
        self.assertIsNone(visible_point_median([]))
        self.assertEqual(visible_point_median([
            {"target_visible_points": 2}, {"target_visible_points": 8},
        ]), 5.0)

    def test_missing_target_frame_fails_identity_continuity(self):
        present, total = temporal_identity_counts([
            {"target_present": True}, {"target_present": False},
            {"target_present": True},
        ])
        self.assertEqual((present, total), (2, 3))
        self.assertFalse(temporal_occlusion_gate(0, 1, 1, present, total, 0, 8))

    def test_presence_gate_is_review_support_not_boundary_truth(self):
        self.assertTrue(temporal_presence_gate(4, 2, 1, 5, 5, True, 0.0))
        self.assertFalse(temporal_presence_gate(4, 2, 0, 5, 5, True, 0.0))
        self.assertFalse(temporal_presence_gate(4, 2, 1, 4, 5, True, 0.0))
        self.assertFalse(temporal_presence_gate(4, 2, 1, 5, 5, False, 0.0))
        self.assertFalse(temporal_presence_gate(4, 2, 1, 5, 5, True, 0.01))

    def test_frozen_inputs_bind_manifest_calibration_labels_and_pcd(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            data_root = root / "data"
            calibration = root / "calib"
            (data_root / "labels").mkdir(parents=True)
            (data_root / "pcd").mkdir()
            calibration.mkdir()
            manifest = {"camera": 1, "frames": [{"frame_id": "100"}]}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (calibration / "lidar2camera.yaml").write_text("x", encoding="utf-8")
            (calibration / "camera1_calib.yaml").write_text("y", encoding="utf-8")
            (data_root / "labels" / "100.json").write_text("{}", encoding="utf-8")
            (data_root / "pcd" / "100.pcd").write_text("pcd", encoding="utf-8")
            paths = temporal_input_paths(
                manifest_path, manifest, data_root, calibration)
            self.assertEqual(len(paths), 5)
            frozen = freeze_temporal_inputs(
                manifest_path, manifest, data_root, calibration)
            self.assertEqual(len(frozen), 5)
            verify_temporal_inputs(frozen)
            (calibration / "camera1_calib.yaml").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed during evaluation"):
                verify_temporal_inputs(frozen)


if __name__ == "__main__":
    unittest.main()
