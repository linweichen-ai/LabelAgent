import unittest

import numpy as np

from tools.diagnose_ego_bottom_point_fallback import bounded_candidate_summary


class EgoBottomPointDiagnosticTests(unittest.TestCase):
    def test_bounded_summary_exposes_geometry_without_mask_pixels(self):
        shape = {
            "point": {"x": 50, "y": 90}, "score": 0.8,
            "guard_profile": "wide_bottom_chassis",
            "_record": {
                "bbox": [10, 70, 90, 100], "area": 1600,
                "mask": np.ones((30, 80), dtype=np.uint8),
            },
        }
        summary = bounded_candidate_summary(shape, 100, 100)
        self.assertEqual(summary["bbox"], [10, 70, 90, 100])
        self.assertEqual(summary["area_ratio"], 0.16)
        self.assertEqual(summary["width_ratio"], 0.8)
        self.assertEqual(summary["height_ratio"], 0.3)
        self.assertEqual(summary["bottom_gap"], 0)
        self.assertEqual(summary["center_ratio"], 0.5)
        self.assertNotIn("mask", summary)


if __name__ == "__main__":
    unittest.main()
