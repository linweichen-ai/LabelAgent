import unittest

import numpy as np

from tools.compare_sam3_segmentation_predictions import mask_metrics


class SegmentationComparisonTests(unittest.TestCase):
    def test_identical_masks_are_perfect(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[2:9, 3:10] = True
        metrics = mask_metrics(mask, mask)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["boundary_f1"], 1.0)

    def test_disjoint_masks_are_zero(self):
        first = np.zeros((12, 12), dtype=bool)
        second = np.zeros((12, 12), dtype=bool)
        first[1:3, 1:3] = True
        second[8:10, 8:10] = True
        metrics = mask_metrics(first, second)
        self.assertEqual(metrics["iou"], 0.0)
        self.assertEqual(metrics["dice"], 0.0)
        self.assertEqual(metrics["boundary_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
