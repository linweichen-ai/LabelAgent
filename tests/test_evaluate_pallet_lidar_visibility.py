import unittest

from tools.evaluate_pallet_lidar_visibility import classify_visibility


class PalletLidarVisibilityTest(unittest.TestCase):
    def test_no_return_is_not_called_occlusion(self):
        self.assertEqual(classify_visibility(0, 0, [0] * 9, .95),
                         "no_lidar_return_insufficient_evidence")

    def test_visibility_must_be_stable_across_grid(self):
        self.assertEqual(classify_visibility(20, 20, [3, 4, 8], 0),
                         "visible_lidar_supported")
        self.assertEqual(classify_visibility(20, 20, [2, 4, 8], 0),
                         "sparse_or_parameter_sensitive")

    def test_occlusion_requires_returns_in_view_and_nearer_overlap(self):
        self.assertEqual(classify_visibility(20, 12, [0, 1, 0], .8),
                         "occluded_lidar_supported")
        self.assertEqual(classify_visibility(20, 12, [0, 1, 0], .4),
                         "sparse_or_parameter_sensitive")


if __name__ == "__main__":
    unittest.main()
