import unittest

import numpy as np

from tools.evaluate_ego_cvat_truth_shadow import (
    failed_evaluation,
    mask_metrics,
    render_label_mask,
)


class EgoCvatTruthShadowTest(unittest.TestCase):
    def test_render_label_mask_preserves_parts_and_holes(self):
        shapes = [{
            "label": "ego",
            "parts": [
                {"points": [[1, 1], [6, 1], [6, 6], [1, 6]],
                 "holes": [[[3, 3], [4, 3], [4, 4], [3, 4]]]},
                {"points": [[8, 1], [9, 1], [9, 2], [8, 2]], "holes": []},
            ],
        }]
        mask = render_label_mask(shapes, "ego", (12, 8))
        self.assertTrue(mask[1, 1])
        self.assertFalse(mask[3, 3])
        self.assertTrue(mask[1, 8])
        self.assertEqual(mask.dtype, np.bool_)

    def test_render_label_mask_ignores_other_labels(self):
        mask = render_label_mask([
            {"label": "floor", "points": [[0, 0], [2, 0], [2, 2]], "holes": []},
        ], "ego", (4, 4))
        self.assertFalse(mask.any())

    def test_render_label_mask_accepts_explicit_empty_candidate(self):
        mask = render_label_mask([
            {"label": "ego", "parts": [], "score": 0.0},
        ], "ego", (4, 4))
        self.assertFalse(mask.any())

    def test_mask_metrics_reports_iou_and_cross_class_overlap(self):
        candidate = np.zeros((4, 4), dtype=bool)
        truth = np.zeros((4, 4), dtype=bool)
        person = np.zeros((4, 4), dtype=bool)
        floor = np.zeros((4, 4), dtype=bool)
        candidate[0:2, 0:2] = True
        truth[0:2, 1:3] = True
        person[0, 0] = True
        floor[1, 0] = True
        metrics = mask_metrics(candidate, truth, person, floor)
        self.assertAlmostEqual(metrics["iou"], 1 / 3, places=6)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["person_overlap"], 0.25)
        self.assertEqual(metrics["floor_overlap"], 0.25)

    def test_empty_ego_truth_fails_closed(self):
        empty = np.zeros((2, 2), dtype=bool)
        with self.assertRaisesRegex(ValueError, "truth is empty"):
            mask_metrics(empty, empty, empty, empty)

    def test_request_failure_is_an_unavailable_empty_candidate(self):
        baseline, fixed, baseline_shape, fixed_shape, attempts = (
            failed_evaluation(ValueError("negative point conflict")))
        self.assertTrue(baseline["request_failed"])
        self.assertTrue(fixed["request_failed"])
        self.assertFalse(fixed["candidate_available"])
        self.assertFalse(fixed["baseline_retention_pass"])
        self.assertEqual(baseline_shape["parts"], [])
        self.assertEqual(fixed_shape["parts"], [])
        self.assertEqual(attempts, [])


if __name__ == "__main__":
    unittest.main()
