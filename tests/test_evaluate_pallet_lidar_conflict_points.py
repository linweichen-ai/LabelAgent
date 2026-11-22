import unittest

import numpy as np

from tools.evaluate_pallet_lidar_conflict_points import (
    accepted_candidate, prompt_groups, spread_points, unique_instance_pixels,
)


class PalletLidarConflictPointTest(unittest.TestCase):
    def test_spread_points_is_deterministic_and_keeps_extent(self):
        points = np.array([[0, 0], [0, 0], [5, 0], [10, 0], [5, 5]])
        selected = spread_points(points, 3)
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected, spread_points(points[::-1], 3))
        self.assertIn([0, 0], selected)
        self.assertIn([10, 0], selected)

    def test_prompt_groups_convert_global_points_to_roi_coordinates(self):
        groups = prompt_groups([[110, 210], [120, 220]],
                               [[130, 230], [140, 240]], (100, 200, 300, 400))
        self.assertEqual(groups[0][1][0], {"x": 10, "y": 10, "label": 1})
        self.assertEqual(groups[0][1][1], {"x": 30, "y": 30, "label": 0})
        self.assertEqual(len(groups), 3)

    def test_unique_pixels_exclude_points_inside_both_boxes(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        pixels = np.array([[10, 10], [20, 20], [30, 30]])
        indices = np.arange(3)
        target = {"obj_type": "Pallet", "obj_psr": {
            "position": {"x": .5, "y": 0, "z": 0},
            "rotation": {"roll": 0, "pitch": 0, "yaw": 0},
            "scale": {"x": 2, "y": 2, "z": 2}}}
        other = {"obj_type": "Pallet", "obj_psr": {
            "position": {"x": 1.5, "y": 0, "z": 0},
            "rotation": {"roll": 0, "pitch": 0, "yaw": 0},
            "scale": {"x": 2, "y": 2, "z": 2}}}
        positive, negative = unique_instance_pixels(
            points, indices, pixels, target, [other])
        self.assertEqual(positive.tolist(), [[10, 10]])
        self.assertEqual(negative.tolist(), [[30, 30]])

    def test_acceptance_requires_all_group_gates(self):
        metrics = {"proposal_pixels": 100, "projected_box_pixels": 200,
                   "projected_box_coverage": .5,
                   "proposal_inside_projected_box": .9,
                   "proposal_to_projected_box_area_ratio": .5,
                   "center_distance_pixels": 2,
                   "maximum_other_projected_box_fraction": .1}
        split = [{"candidate_pass": True}, {"candidate_pass": True}]
        separation = [{"separated_at_0_15": True}]
        self.assertTrue(accepted_candidate(metrics, 1, 1, split, separation))
        self.assertFalse(accepted_candidate(metrics, .5, 1, split, separation))
        self.assertFalse(accepted_candidate(
            metrics, 1, 1, split, [{"separated_at_0_15": False}]))


if __name__ == "__main__":
    unittest.main()
