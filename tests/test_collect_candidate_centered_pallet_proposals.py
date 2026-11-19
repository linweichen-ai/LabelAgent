import unittest

import numpy as np

from source.agent.edit_agent import _shape_from_mask
from tools.collect_candidate_centered_pallet_proposals import centered_crop_bounds


def target(x0, y0, x1, y1, width=640, height=480):
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return _shape_from_mask(mask, "pallet", 1)


class CandidateCenteredCropTest(unittest.TestCase):
    def test_crop_is_centered_and_bounded(self):
        bounds = centered_crop_bounds(
            target(280, 220, 360, 260), 640, 480, 3,
            minimum_side=128, maximum_side=400)

        self.assertEqual(bounds, (200, 120, 440, 360))

    def test_crop_shifts_inside_image_at_border(self):
        bounds = centered_crop_bounds(
            target(0, 0, 80, 40), 640, 480, 4,
            minimum_side=128, maximum_side=400)

        self.assertEqual(bounds, (0, 0, 320, 320))

    def test_crop_obeys_minimum_and_maximum_side(self):
        small = centered_crop_bounds(
            target(300, 220, 310, 230), 640, 480, 3,
            minimum_side=128, maximum_side=400)
        large = centered_crop_bounds(
            target(100, 100, 300, 300), 640, 480, 6,
            minimum_side=128, maximum_side=400)

        self.assertEqual(small[2] - small[0], 128)
        self.assertEqual(large[2] - large[0], 400)


if __name__ == "__main__":
    unittest.main()
