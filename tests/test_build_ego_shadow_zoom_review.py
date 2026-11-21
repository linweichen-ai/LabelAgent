import unittest

import numpy as np

from tools.build_ego_shadow_zoom_review import added_region_bbox


class EgoShadowZoomReviewTests(unittest.TestCase):
    def test_empty_added_mask_has_no_crop(self):
        self.assertIsNone(added_region_bbox(np.zeros((20, 30), dtype=bool)))

    def test_crop_contains_added_region_and_stays_in_bounds(self):
        added = np.zeros((100, 120), dtype=bool)
        added[2:5, 3:8] = True
        x0, y0, x1, y1 = added_region_bbox(
            added, padding=12, minimum_side=40)
        self.assertLessEqual(x0, 3)
        self.assertLessEqual(y0, 2)
        self.assertGreaterEqual(x1, 8)
        self.assertGreaterEqual(y1, 5)
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x1, 120)
        self.assertLessEqual(y1, 100)

    def test_invalid_crop_settings_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "invalid"):
            added_region_bbox(np.zeros((2, 2, 1)), padding=1)


if __name__ == "__main__":
    unittest.main()
