import unittest

import numpy as np

from tools.audit_pallet_box3d_conflict_margin import (
    maximum_projected_overlap, threshold_counts,
)


class PalletBox3dConflictMarginTest(unittest.TestCase):
    def test_maximum_overlap_reports_stronger_fraction(self):
        target_mask = np.zeros((10, 10), dtype=bool); target_mask[:5, :5] = True
        narrow = np.zeros_like(target_mask); narrow[:5, :2] = True
        other = {"obj_id": 2, "mask": narrow}
        target = {"obj_id": 1, "mask": target_mask}
        value = maximum_projected_overlap(target, [target, other])
        self.assertEqual(value["other_obj_id"], 2)
        self.assertAlmostEqual(value["target_overlap"], .4)
        self.assertAlmostEqual(value["other_overlap"], 1.0)
        self.assertAlmostEqual(value["maximum_overlap"], 1.0)

    def test_threshold_counts_are_inclusive(self):
        self.assertEqual(threshold_counts([.05, .25, .65], [.10, .25, .50]),
                         {"0.10": 2, "0.25": 2, "0.50": 1})


if __name__ == "__main__":
    unittest.main()
