import unittest

import numpy as np

from tools.evaluate_pallet_lidar_centered_roi_sam import (
    _crop_box, _mask_metrics, select_associated_result,
)


def rectangle(x0, y0, x1, y1, score):
    return {"score": score, "parts": [{"points": [
        [x0, y0], [x1, y0], [x1, y1], [x0, y1]], "holes": []}]}


class PalletLidarCenteredRoiSamTest(unittest.TestCase):
    def test_center_association_beats_higher_score(self):
        wrong = rectangle(0, 0, 8, 8, .9)
        right = rectangle(10, 10, 20, 20, .6)
        current = np.zeros((32, 32), dtype=bool); current[10:21, 10:21] = True
        selected, mask, contains, distance, _ = select_associated_result(
            [wrong, right], current, 15, 15, 32, 32)
        self.assertIs(selected, right)
        self.assertTrue(contains)
        self.assertEqual(distance, 0.0)
        self.assertTrue(mask[15, 15])

    def test_score_breaks_tie_after_center_gate(self):
        low = rectangle(8, 8, 20, 20, .6)
        high = rectangle(10, 10, 22, 22, .8)
        current = np.zeros((32, 32), dtype=bool); current[10:21, 10:21] = True
        selected, _, _, _, _ = select_associated_result(
            [low, high], current, 15, 15, 32, 32)
        self.assertIs(selected, high)

    def test_current_instance_anchor_beats_unrelated_center_mask(self):
        incumbent = rectangle(4, 4, 9, 9, .5)
        unrelated = rectangle(14, 14, 24, 24, .95)
        current = np.zeros((32, 32), dtype=bool); current[4:10, 4:10] = True
        selected, _, _, _, summaries = select_associated_result(
            [unrelated, incumbent], current, 18, 18, 32, 32)
        self.assertIs(selected, incumbent)
        self.assertTrue(summaries[1]["anchored_to_current"])

    def test_crop_box_respects_bounds_and_minimum(self):
        self.assertEqual(_crop_box([0, 0, 20, 20], 300, 300, 100, 0),
                         [0, 0, 100, 100])

    def test_metrics_keep_prompt_and_collision_checks(self):
        current = np.zeros((20, 20), dtype=bool); current[4:10, 4:10] = True
        proposal = np.zeros_like(current); proposal[4:11, 4:11] = True
        other = np.zeros_like(current); other[15:18, 15:18] = True
        metrics = _mask_metrics(proposal, current, [other], [
            {"x": 5, "y": 5, "label": 1}, {"x": 1, "y": 1, "label": 0}])
        self.assertEqual(metrics["positive_coverage"], 1.0)
        self.assertEqual(metrics["negative_clear_fraction"], 1.0)
        self.assertEqual(metrics["maximum_other_candidate_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
