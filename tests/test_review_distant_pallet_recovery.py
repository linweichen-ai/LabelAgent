import unittest

import numpy as np

from source.agent.edit_agent import _shape_from_mask
from tools.review_distant_pallet_recovery import (
    expansion_metrics,
    match_expansion,
)


def shape_from_box(identifier, x0, y0, x1, y1, score=.8):
    mask = np.zeros((100, 160), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    shape = _shape_from_mask(mask, "pallet", identifier)
    shape["score"] = score
    return shape


class DistantPalletRecoveryReviewTest(unittest.TestCase):
    def test_metrics_measure_fragment_coverage_and_expansion(self):
        fragment = shape_from_box("fragment", 30, 30, 60, 45)
        complete = shape_from_box("complete", 20, 20, 100, 70)

        metrics = expansion_metrics(fragment, complete, 160, 100)

        self.assertEqual(metrics["baseline_coverage"], 1.0)
        self.assertGreater(metrics["proposal_to_baseline_area_ratio"], 8)
        self.assertLess(metrics["proposal_covered_by_baseline"], .2)

    def test_match_requires_overlap_and_larger_area(self):
        fragment = shape_from_box("fragment", 30, 30, 60, 45)
        unrelated = shape_from_box("unrelated", 100, 60, 140, 90, .99)
        same_size = shape_from_box("same", 30, 30, 60, 45, .95)
        complete = shape_from_box("complete", 20, 20, 100, 70, .7)

        proposal, metrics = match_expansion(
            fragment, [unrelated, same_size, complete], 160, 100)

        self.assertEqual(proposal["id"], "complete")
        self.assertEqual(metrics["baseline_coverage"], 1.0)

    def test_match_is_conservative_when_partial_overlap_is_too_low(self):
        fragment = shape_from_box("fragment", 30, 30, 60, 45)
        partial = shape_from_box("partial", 50, 35, 120, 80)

        proposal, metrics = match_expansion(
            fragment, [partial], 160, 100, minimum_baseline_coverage=.8)

        self.assertIsNone(proposal)
        self.assertIsNone(metrics)

    def test_minor_axis_guard_keeps_longitudinal_expansion(self):
        fragment = shape_from_box("fragment", 50, 40, 100, 50)
        longitudinal = shape_from_box("longitudinal", 20, 40, 130, 50)

        proposal, metrics = match_expansion(
            fragment, [longitudinal], 160, 100,
            maximum_minor_axis_ratio=1.5)

        self.assertEqual(proposal["id"], "longitudinal")
        self.assertGreater(metrics["proposal_major_axis_ratio"], 2)
        self.assertLess(metrics["proposal_minor_axis_ratio"], 1.1)

    def test_minor_axis_guard_rejects_perpendicular_background_expansion(self):
        fragment = shape_from_box("fragment", 50, 40, 100, 50)
        perpendicular = shape_from_box("perpendicular", 50, 20, 100, 70)

        proposal, metrics = match_expansion(
            fragment, [perpendicular], 160, 100,
            maximum_minor_axis_ratio=1.5)

        self.assertIsNone(proposal)
        self.assertIsNone(metrics)

    def test_other_instance_guard_rejects_cross_instance_merge(self):
        fragment = shape_from_box("fragment", 50, 40, 100, 50)
        neighbor = shape_from_box("neighbor", 100, 40, 140, 50)
        merged = shape_from_box("merged", 30, 40, 140, 50)

        proposal, metrics = match_expansion(
            fragment, [merged], 160, 100,
            other_baselines=[neighbor],
            maximum_other_baseline_fraction=.1)

        self.assertIsNone(proposal)
        self.assertIsNone(metrics)

    def test_other_instance_guard_keeps_isolated_expansion(self):
        fragment = shape_from_box("fragment", 50, 40, 100, 50)
        neighbor = shape_from_box("neighbor", 130, 70, 150, 90)
        expanded = shape_from_box("expanded", 30, 40, 110, 50)

        proposal, metrics = match_expansion(
            fragment, [expanded], 160, 100,
            other_baselines=[neighbor],
            maximum_other_baseline_fraction=.1)

        self.assertEqual(proposal["id"], "expanded")
        self.assertEqual(metrics["maximum_other_baseline_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
