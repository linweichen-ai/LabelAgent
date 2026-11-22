import unittest

import numpy as np

from tools.evaluate_pallet_box3d_overlap_pairs import (
    depth_split_masks, mask_separation, projected_conflicts,
)


class PalletBox3dOverlapPairsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
