import json
import unittest

import numpy as np

from tools.compare_sam3_segmentation_predictions import compare, mask_metrics


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

    def test_complete_comparison_is_json_serializable(self):
        coco = {
            "images": [{"id": 1, "file_name": "one.png", "height": 2, "width": 2}],
            "categories": [{"id": 1, "name": "pallet"}],
            "annotations": [{"image_id": 1, "category_id": 1,
                             "segmentation": {"size": [2, 2], "counts": [0, 1, 3]}}],
        }
        prediction = [{"image_id": 1, "category_id": 1, "score": 1.0,
                       "segmentation": {"size": [2, 2], "counts": [0, 1, 3]}}]
        value = compare(coco, prediction, prediction)
        json.dumps(value)
        self.assertEqual(value["catastrophic_frames"], [])


if __name__ == "__main__":
    unittest.main()
