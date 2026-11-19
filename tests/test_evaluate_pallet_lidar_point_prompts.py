import unittest

import numpy as np

from tools.evaluate_pallet_lidar_point_prompts import (
    _sample_spread,
    point_policies,
    select_lidar_prompt_points,
    summarize_sam_attempts,
)


class PalletLidarPointPromptsTest(unittest.TestCase):
    def test_spread_sampling_is_unique_and_deterministic(self):
        points = np.array([[10, 10], [10, 10], [20, 10], [30, 10], [40, 10]])
        first = _sample_spread(points, 3, anchor=np.array([25, 10]))
        second = _sample_spread(points, 3, anchor=np.array([25, 10]))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(set(first)), 3)

    def test_points_use_low_object_component_and_floor_negatives(self):
        mask = np.zeros((120, 180), dtype=np.uint8)
        mask[50:60, 70:110] = 1
        existing = mask.copy()
        low = np.array([
            [70, 55], [80, 55], [90, 55], [100, 55], [110, 55],
            [120, 55], [130, 55],
        ])
        floor = np.array([
            [30, 30], [150, 30], [30, 90], [150, 90], [90, 100],
        ])
        pixels = np.vstack([low, floor])
        heights = np.array([.2] * len(low) + [0.0] * len(floor))

        result = select_lidar_prompt_points(mask, existing, pixels, heights,
                                            component_radius=12)

        self.assertTrue(result["ready"], result)
        self.assertGreaterEqual(result["positive_points"], 1)
        self.assertGreaterEqual(result["negative_points"], 2)
        self.assertTrue(all(point["label"] in {0, 1} for point in result["points"]))

    def test_no_low_seed_fails_closed(self):
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[40:50, 40:80] = 1
        pixels = np.array([[10, 10], [100, 90]])
        heights = np.array([.2, 0.0])

        result = select_lidar_prompt_points(mask, mask, pixels, heights)

        self.assertFalse(result["ready"])
        self.assertEqual(result["reason"], "no_low_height_seed_near_candidate")

    def test_insufficient_floor_negatives_fails_closed(self):
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[40:50, 40:80] = 1
        pixels = np.array([[45, 45], [55, 45], [65, 45]])
        heights = np.array([.2, .2, .2])

        result = select_lidar_prompt_points(mask, mask, pixels, heights)

        self.assertFalse(result["ready"])
        self.assertEqual(result["reason"], "insufficient_ground_negative_points")

    def test_invalid_shapes_are_rejected(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        with self.assertRaises(ValueError):
            select_lidar_prompt_points(mask, np.zeros((9, 10)),
                                       np.zeros((0, 2)), np.zeros(0))
        with self.assertRaises(ValueError):
            select_lidar_prompt_points(mask, mask, np.zeros((0, 2)), np.zeros(0))

    def test_point_policies_start_conservative_and_expand(self):
        points = [
            {"x": 10, "y": 10, "label": 1},
            {"x": 30, "y": 10, "label": 1},
            {"x": 20, "y": 20, "label": 1},
            {"x": 5, "y": 40, "label": 0},
            {"x": 40, "y": 40, "label": 0},
        ]
        policies = point_policies(points)
        self.assertEqual([name for name, _ in policies],
                         ["anchor_ring", "extent_ring", "all_low_ring"])
        self.assertEqual(sum(p["label"] == 1 for p in policies[0][1]), 1)
        self.assertEqual(sum(p["label"] == 1 for p in policies[1][1]), 2)

    def test_point_policies_require_safe_negative_ring(self):
        with self.assertRaises(ValueError):
            point_policies([{"x": 1, "y": 1, "label": 1}])

    def test_sam_attempt_summary_separates_frames_requests_and_failures(self):
        rows = [
            {
                "sam_called": True,
                "sam_attempts": [
                    {"sam_mask_pixels": 123},
                    {"http_status": 400, "error": "unsafe negative point"},
                    {"error_type": "ValueError", "error": "invalid polygon"},
                ],
            },
            {"sam_called": False},
            {
                "sam_called": True,
                "sam_attempts": [{"sam_mask_pixels": 456}],
            },
        ]

        self.assertEqual(summarize_sam_attempts(rows), {
            "sam_frame_count": 2,
            "sam_attempt_count": 4,
            "sam_successful_attempt_count": 2,
            "sam_http_error_count": 1,
            "sam_other_error_count": 1,
        })


if __name__ == "__main__":
    unittest.main()
