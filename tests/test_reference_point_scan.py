import io
import unittest

import numpy as np

from PIL import Image

from tools.evaluate_reference_point_scan import (
    fixed_point_grid, isolated_mask_image, known_coverage, multiscale_panel,
    supported_other_robot)


class ReferencePointScanTests(unittest.TestCase):
    def test_default_grid_is_resolution_scaled_and_bounded_to_top_two_thirds(self):
        points = fixed_point_grid(1920, 1536)
        self.assertEqual(len(points), 184)
        self.assertEqual(points[0], {"x": 80, "y": 128})
        self.assertEqual(points[-1], {"x": 1840, "y": 1024})
        self.assertEqual(len({(item["x"], item["y"]) for item in points}), 184)

    def test_known_coverage_measures_candidate_containment_not_scene_iou(self):
        candidate = np.zeros((8, 8), dtype=bool)
        candidate[2:6, 2:6] = True
        half = np.zeros_like(candidate)
        half[2:6, 2:4] = True
        result = known_coverage(candidate, 16, {"person": [half], "pallet": []})
        self.assertEqual(result, {"person": 0.5, "pallet": 0.0})

    def test_multiscale_panel_is_fixed_size_and_marks_the_candidate_at_every_scale(self):
        shape = {"id": 1, "label": "unknown_object", "bbox": [40, 30, 80, 60],
                 "points": [[40, 30], [80, 30], [80, 60], [40, 60]],
                 "holes": [], "parts": [{"points": [[40, 30], [80, 30],
                                                        [80, 60], [40, 60]],
                                            "holes": []}]}
        data = multiscale_panel(Image.new("RGB", (160, 120), "white"), shape, "C01")
        with Image.open(io.BytesIO(data)) as rendered:
            self.assertEqual(rendered.size, (1024, 768))

    def test_supported_other_robot_is_semantic_vote_only(self):
        valid = {"class": "other_robot"}
        self.assertTrue(supported_other_robot(valid))
        self.assertFalse(supported_other_robot({"class": "pallet"}))

    def test_isolated_mask_image_hides_surrounding_pixels(self):
        shape = {"id": 1, "label": "unknown_object", "bbox": [40, 30, 80, 60],
                 "points": [[40, 30], [80, 30], [80, 60], [40, 60]],
                 "holes": [], "parts": [{"points": [[40, 30], [80, 30],
                                                        [80, 60], [40, 60]],
                                            "holes": []}]}
        data = isolated_mask_image(Image.new("RGB", (160, 120), "white"), shape)
        with Image.open(io.BytesIO(data)) as rendered:
            self.assertEqual(rendered.size, (512, 384))
            self.assertLess(max(rendered.getpixel((0, 0))), 150)


if __name__ == "__main__":
    unittest.main()
