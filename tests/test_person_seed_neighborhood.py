import unittest

import numpy as np

from tools.person_seed_neighborhood import expand_person_seeds
from tools.multiview_person_guard import rejects_novel_points


class PersonSeedNeighbourhoodTests(unittest.TestCase):
    def test_metric_ball_includes_boundary_and_does_not_chain(self):
        xyz = np.array([[0., 0., 0.], [.15, 0., 0.], [.29, 0., 0.], [0., .151, 0.]])
        mask = np.array([True, False, False, False])
        np.testing.assert_array_equal(expand_person_seeds(xyz, mask), [True, True, False, False])

    def test_euclidean_distance_not_axis_aligned_cube(self):
        xyz = np.array([[0., 0., 0.], [.12, .12, 0.], [.09, .09, 0.]])
        np.testing.assert_array_equal(expand_person_seeds(xyz, np.array([True, False, False])),
                                      [True, False, True])

    def test_rigid_transform_preserves_protection(self):
        xyz = np.array([[0., 0., 0.], [.1, .01, 0.], [.3, 0., 0.]])
        mask = np.array([True, False, False])
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        np.testing.assert_array_equal(expand_person_seeds(xyz, mask),
                                      expand_person_seeds(xyz @ rotation.T + [4., -3., 2.], mask))

    def test_empty_seeds_and_empty_scan_do_not_protect_unobserved_points(self):
        np.testing.assert_array_equal(expand_person_seeds(np.zeros((3, 3)), np.zeros(3, bool)), np.zeros(3, bool))
        self.assertEqual(expand_person_seeds(np.empty((0, 3)), np.empty(0, bool)).shape, (0,))

    def test_aligned_raw_order_is_preserved_and_inputs_are_unchanged(self):
        xyz = np.array([[8., 0., 0.], [.1, 0., 0.], [0., 0., 0.]])
        mask = np.array([False, False, True])
        before = xyz.copy(), mask.copy()
        np.testing.assert_array_equal(expand_person_seeds(xyz, mask), [False, True, True])
        np.testing.assert_array_equal(xyz, before[0])
        np.testing.assert_array_equal(mask, before[1])

    def test_invalid_alignment_nonfinite_geometry_and_radius_are_rejected(self):
        xyz = np.zeros((2, 3))
        for points, mask in ((xyz, [1, 0]), (xyz, np.zeros(3, bool)),
                             (np.array([[0., np.nan, 0.]]), np.ones(1, bool)),
                             (np.zeros((2, 2)), np.zeros(2, bool))):
            with self.assertRaises(ValueError):
                expand_person_seeds(points, mask)
        for radius in (0, -1, float('inf'), float('nan'), True):
            with self.assertRaises(ValueError):
                expand_person_seeds(xyz, np.zeros(2, bool), radius)

    def test_new_neighbor_is_vetoed_in_other_view_order_without_rejecting_old_hits(self):
        xyz = np.array([[0., 0., 0.], [.1, 0., 0.], [3., 0., 0.]])
        seeds = np.array([True, False, False])
        protected = expand_person_seeds(xyz, seeds)
        indices = np.array([2, 1, 0])
        old = np.array([False, False, True])
        candidate = np.array([False, True, True])
        self.assertFalse(rejects_novel_points(candidate, old, indices, seeds))
        self.assertTrue(rejects_novel_points(candidate, old, indices, protected))
        self.assertFalse(rejects_novel_points(old, old, indices, protected))


if __name__ == '__main__':
    unittest.main()
