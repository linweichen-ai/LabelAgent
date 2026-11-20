import unittest

import numpy as np

from tools.audit_pallet_lidar_instance_prompts import (
    _xy_components,
    select_instance_lidar_prompts,
)


class PalletLidarInstancePromptTest(unittest.TestCase):
    def test_xy_components_separate_depth_adjacent_image_returns(self):
        xyz = np.array([
            [0.00, 0.00, 0.2], [0.05, 0.00, 0.2],
            [1.00, 0.00, 0.2], [1.05, 0.00, 0.2],
        ])
        labels = _xy_components(xyz, 0.1)
        self.assertEqual(labels.tolist(), [0, 0, 1, 1])

    def _scene(self):
        mask = np.zeros((120, 180), dtype=np.uint8)
        mask[48:63, 70:112] = 1
        pixels = np.array([
            [72, 54], [82, 54], [92, 54], [102, 54],
            [76, 57], [106, 57],
            [72, 55], [102, 55],
            [25, 30], [155, 30], [25, 95], [155, 95], [90, 105],
            [80, 35], [100, 32],
        ])
        heights = np.array([
            0.20, 0.21, 0.19, 0.20, 0.18, 0.22,
            0.20, 0.21,
            0.00, 0.01, -0.01, 0.00, 0.01,
            0.90, 1.10,
        ])
        xyz = np.array([
            [0.00, 0.00, 0.2], [0.05, 0.00, 0.2], [0.10, 0.00, 0.2],
            [0.15, 0.00, 0.2], [0.03, 0.05, 0.2], [0.13, 0.05, 0.2],
            [1.00, 0.00, 0.2], [1.05, 0.00, 0.2],
            [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0], [0.0, 1.2, 0.0],
            [0.2, -0.2, 0.9], [0.3, -0.2, 1.1],
        ])
        return mask, pixels, heights, xyz

    def test_selects_dominant_3d_component_and_tall_negatives(self):
        mask, pixels, heights, xyz = self._scene()
        result = select_instance_lidar_prompts(mask, mask, pixels, heights, xyz)
        self.assertTrue(result["ready"], result)
        self.assertEqual(result["winning_seed_points"], 6)
        self.assertEqual(result["runner_up_seed_points"], 2)
        self.assertGreaterEqual(result["floor_negative_points"], 2)
        self.assertGreaterEqual(result["tall_negative_points"], 1)
        self.assertTrue(all(point["source"] for point in result["points"]))

    def test_equal_seed_components_fail_closed(self):
        mask, pixels, heights, xyz = self._scene()
        # Retain two points in each low component and all negatives.
        keep = np.array([0, 1, 6, 7, *range(8, len(pixels))])
        result = select_instance_lidar_prompts(
            mask, mask, pixels[keep], heights[keep], xyz[keep]
        )
        self.assertFalse(result["ready"])
        self.assertEqual(result["reason"], "ambiguous_low_height_components")

    def test_missing_ground_ring_fails_closed(self):
        mask, pixels, heights, xyz = self._scene()
        keep = np.arange(8)
        result = select_instance_lidar_prompts(
            mask, mask, pixels[keep], heights[keep], xyz[keep]
        )
        self.assertFalse(result["ready"])
        self.assertEqual(result["reason"], "insufficient_ground_negative_points")

    def test_alignment_validation(self):
        mask = np.ones((4, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            select_instance_lidar_prompts(
                mask, mask, np.zeros((2, 2)), np.zeros(2), np.zeros((1, 3))
            )


if __name__ == "__main__":
    unittest.main()
