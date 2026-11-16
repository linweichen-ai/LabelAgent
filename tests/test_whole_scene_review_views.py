import unittest

import numpy as np
from PIL import Image
from tools.whole_scene_review_views import context_boxes, whole_scene_views


class WholeSceneChecks(unittest.TestCase):
    def test_edges_and_odd_dimensions_retain_entire_target(self):
        for x, y in ((0, 0), (100, 0), (0, 58), (100, 58)):
            mask = np.zeros((59, 101), bool); mask[y, x] = True
            (x1, y1, x2, y2), _ = context_boxes(mask)
            self.assertEqual(int(mask[y1:y2, x1:x2].sum()), 1)
            self.assertEqual((x2 - x1, y2 - y1), (51, 30))

    def test_target_larger_than_half_frame_is_not_clipped(self):
        mask = np.zeros((60, 100), bool); mask[3:58, 2:98] = True
        (x1, y1, x2, y2), _ = context_boxes(mask)
        self.assertEqual(int(mask[y1:y2, x1:x2].sum()), int(mask.sum()))

    def test_identical_scene_rule_at_opposite_sides_no_hand_chosen_crop(self):
        a = np.zeros((60, 100), bool); a[20:24, 2:8] = True
        b = np.fliplr(a).copy()
        box_a, _ = context_boxes(a); box_b, _ = context_boxes(b)
        self.assertEqual(box_a[2] - box_a[0], box_b[2] - box_b[0])
        self.assertEqual(box_a[0], 100 - box_b[2])

    def test_scene_dimensions_and_complete_context_are_retained(self):
        mask = np.zeros((60, 100), bool); mask[20:24, 88:99] = True
        views, info = whole_scene_views(Image.new("RGB", (100, 60), "#245678"), mask)
        self.assertEqual(views["overview"].size, (1024, 1024))
        self.assertEqual(views["linked"].size, (1536, 512))
        self.assertEqual(info["whole_scene_box"], [0, 0, 100, 60])
        self.assertEqual(info["overview_panels"][0]["source_size"], [100, 60])
        self.assertEqual(info["linked_panels"][0]["source_size"], [100, 60])
        self.assertFalse(info["categories_IDs_scores_SAM_labels_rendered"])

    def test_source_pixels_and_target_mask_not_mutated(self):
        source = Image.new("RGB", (100, 60), "red")
        mask = np.zeros((60, 100), bool); mask[20:24, 88:99] = True
        before = source.tobytes(), mask.tobytes()
        whole_scene_views(source, mask)
        self.assertEqual(before, (source.tobytes(), mask.tobytes()))

    def test_empty_or_misaligned_source_rejected(self):
        with self.assertRaises(ValueError):
            context_boxes(np.zeros((60, 100), bool))
        with self.assertRaises(ValueError):
            whole_scene_views(Image.new("RGB", (100, 60)), np.ones((60, 99), bool))


if __name__ == "__main__":
    unittest.main()
