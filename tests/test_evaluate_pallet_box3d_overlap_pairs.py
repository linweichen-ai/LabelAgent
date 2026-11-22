import unittest

import numpy as np

from tools.evaluate_pallet_box3d_overlap_pairs import (
    depth_split_masks, index_manifest_files, mask_separation,
    projected_conflicts, select_source_rows,
)


class PalletBox3dOverlapPairsTest(unittest.TestCase):
    def test_manifest_index_preserves_multiple_frames_per_camera(self):
        rows = [
            {"sequence": "010", "camera": "camera_0", "frame_id": "100"},
            {"sequence": "010", "camera": "camera_0", "frame_id": "200"},
        ]
        indexed = index_manifest_files(rows)
        self.assertEqual(len(indexed), 2)
        self.assertEqual(indexed[("010", 0, "100")]["frame_id"], "100")
        self.assertEqual(indexed[("010", 0, "200")]["frame_id"], "200")

    def test_manifest_index_rejects_duplicate_frame_identity(self):
        row = {"sequence": "010", "camera": "camera_0", "frame_id": "100"}
        with self.assertRaisesRegex(ValueError, "duplicate manifest image identity"):
            index_manifest_files([row, dict(row)])

    def test_mask_separation_uses_smaller_instance(self):
        first = np.zeros((20, 20), dtype=bool); first[2:8, 2:8] = True
        second = np.zeros_like(first); second[7:13, 7:13] = True
        value = mask_separation(first, second)
        self.assertTrue(value["separated_at_0_15"])
        second[2:8, 2:8] = True
        self.assertFalse(mask_separation(first, second)["separated_at_0_15"])

    def test_projected_conflicts_require_material_overlap(self):
        target_mask = np.zeros((10, 10), dtype=bool); target_mask[:5, :5] = True
        near = np.zeros_like(target_mask); near[:5, :4] = True
        far = np.zeros_like(target_mask); far[8:, 8:] = True
        target = {"mask": target_mask, "obj_id": 1}
        rows = projected_conflicts(target, [target, {"mask": near, "obj_id": 2},
                                               {"mask": far, "obj_id": 3}])
        self.assertEqual([row[0]["obj_id"] for row in rows], [2])

    def test_depth_split_assigns_overlap_to_nearest_box(self):
        hull = np.zeros((10, 10), dtype=bool); hull[2:8, 2:8] = True
        projected = [
            {"obj_id": 1, "mask": hull, "camera_depth": 2.0},
            {"obj_id": 2, "mask": hull, "camera_depth": 3.0},
        ]
        source = {1: hull.copy(), 2: hull.copy()}
        split, metrics = depth_split_masks(projected, source)
        self.assertEqual(int(split[1].sum()), int(hull.sum()))
        self.assertEqual(int(split[2].sum()), 0)
        self.assertTrue(metrics[0]["candidate_pass"])
        self.assertFalse(metrics[1]["candidate_pass"])
        self.assertTrue(metrics[1]["occluded_unobservable_candidate"])
        self.assertTrue(metrics[1]["handled_candidate"])

    def test_select_source_rows_preserves_default_and_supports_full_audit(self):
        source = {"rows": [
            {"id": 1, "localization_pass": True, "instance_isolation_pass": False},
            {"id": 2, "localization_pass": True, "instance_isolation_pass": True},
            {"id": 3, "localization_pass": False, "instance_isolation_pass": False},
        ]}
        self.assertEqual([row["id"] for row in select_source_rows(source, "ambiguous")],
                         [1])
        self.assertEqual([row["id"] for row in select_source_rows(source, "all_localized")],
                         [1, 2])
        with self.assertRaises(ValueError):
            select_source_rows(source, "unsupported")


if __name__ == "__main__":
    unittest.main()
