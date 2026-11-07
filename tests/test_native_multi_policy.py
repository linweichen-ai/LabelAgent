import unittest
import numpy as np
from native_multi_policy import anchors, place


class NativePolicyTests(unittest.TestCase):
    def test_disconnected_and_single_pixel_retained(self):
        raw = np.zeros((20, 30), np.float32)
        raw[2:8, 4:10] = 1
        raw[17, 27] = 1
        full = place(raw, (100, 150, 130, 170))
        self.assertEqual(int(full.sum()), 37)
        np.testing.assert_array_equal(full[150:170, 100:130], raw)

    def test_points_are_in_seed_and_deterministic(self):
        seed = np.zeros((30, 40), bool)
        seed[2:10, 3:30] = True
        seed[20:27, 5:28] = True
        points = anchors(seed)
        self.assertEqual(points, anchors(seed))
        self.assertEqual(len(points), 2)
        self.assertTrue(all(seed[p['y'], p['x']] for p in points))
        self.assertGreaterEqual(points[1]['y'], 20)

    def test_invalid_values_or_shape_not_coerced(self):
        with self.assertRaises(ValueError):
            place(np.full((3, 4), .5), (0, 0, 4, 3))
        with self.assertRaises(ValueError):
            place(np.ones((3, 4)), (0, 0, 5, 3))


if __name__ == '__main__':
    unittest.main()
