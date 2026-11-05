import unittest

import numpy as np
from PIL import Image
from tools.focused_target_views import focused_target_views


class FocusedTargetChecks(unittest.TestCase):
    def fixture(self):
        pixels = np.random.default_rng(3).integers(0, 256, (90, 90, 3), dtype=np.uint8)
        mask = np.zeros((90, 90), bool); mask[42:48, 82:88] = True
        return pixels, mask, (80, 40, 90, 50)

    def test_all_outside_pixels_can_change_without_affecting_classification_images(self):
        pixels, mask, box = self.fixture(); changed = np.zeros_like(pixels); changed[:] = (0, 0, 255)
        changed[40:50, 80:90] = pixels[40:50, 80:90]
        a, _ = focused_target_views(Image.fromarray(pixels), mask, box)
        b, _ = focused_target_views(Image.fromarray(changed), mask, box)
        self.assertEqual({k: v.tobytes() for k, v in a.items()}, {k: v.tobytes() for k, v in b.items()})

    def test_target_retained_and_pointer_inside_given_foreground(self):
        pixels, mask, box = self.fixture(); views, meta = focused_target_views(Image.fromarray(pixels), mask, box)
        self.assertEqual(meta['retained_target_pixels'], int(mask.sum()))
        self.assertTrue(meta['pointer_tip_in_original_target'])
        self.assertTrue(all(v.size == (512, 512) for v in views.values()))

    def test_crop_that_discards_target_is_rejected(self):
        pixels, mask, _ = self.fixture()
        with self.assertRaises(ValueError): focused_target_views(Image.fromarray(pixels), mask, (0, 0, 10, 10))

    def test_source_pixels_and_geometry_are_not_mutated(self):
        pixels, mask, box = self.fixture(); source = Image.fromarray(pixels); before = source.tobytes(), mask.tobytes()
        focused_target_views(source, mask, box)
        self.assertEqual(before, (source.tobytes(), mask.tobytes()))


if __name__ == '__main__':
    unittest.main()
