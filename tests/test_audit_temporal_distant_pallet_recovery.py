import unittest

import numpy as np

from source.agent.edit_agent import _shape_from_mask
from tools.audit_temporal_distant_pallet_recovery import (
    mask_iou,
    select_temporal_consensus,
)


def box(identifier, x0, y0, x1, y1):
    mask = np.zeros((100, 160), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    shape = _shape_from_mask(mask, "pallet", identifier)
    shape["score"] = .8
    return shape


def proposal(source_frame, shape, coverage=.9, ratio=2.0):
    return {
        "source_frame": source_frame,
        "shape": shape,
        "metrics": {
            "baseline_coverage": coverage,
            "proposal_to_baseline_area_ratio": ratio,
        },
    }


class TemporalDistantPalletRecoveryTest(unittest.TestCase):
    def test_mask_iou_measures_overlap(self):
        first = box("one", 20, 20, 80, 60)
        second = box("two", 20, 20, 80, 60)
        third = box("three", 100, 60, 140, 90)
        self.assertEqual(mask_iou(first, second, 160, 100), 1.0)
        self.assertEqual(mask_iou(first, third, 160, 100), 0.0)

    def test_consensus_requires_two_distinct_source_frames(self):
        shape = box("one", 20, 20, 80, 60)
        selected, audit = select_temporal_consensus(
            [proposal(8, shape), proposal(8, shape)], 160, 100)
        self.assertIsNone(selected)
        self.assertFalse(audit["accepted"])

    def test_consensus_keeps_supported_expansion(self):
        first = box("one", 20, 20, 90, 60)
        second = box("two", 22, 20, 92, 60)
        unrelated = box("three", 100, 60, 140, 90)
        selected, audit = select_temporal_consensus([
            proposal(8, first, coverage=.95),
            proposal(12, second, coverage=.9),
            proposal(14, unrelated, coverage=.99),
        ], 160, 100, minimum_iou=.8)
        self.assertIsNotNone(selected)
        self.assertTrue(audit["accepted"])
        self.assertEqual(audit["supporting_source_frame_count"], 2)
        self.assertEqual(audit["supporting_source_frames"], [8, 12])

    def test_consensus_rejects_disagreeing_expansions(self):
        first = box("one", 10, 20, 60, 60)
        second = box("two", 70, 20, 120, 60)
        selected, audit = select_temporal_consensus([
            proposal(8, first), proposal(12, second)
        ], 160, 100, minimum_iou=.45)
        self.assertIsNone(selected)
        self.assertEqual(audit["reason"], "no_two_frame_consensus")

    def test_consensus_validates_thresholds(self):
        with self.assertRaises(ValueError):
            select_temporal_consensus([], 160, 100, minimum_iou=0)
        with self.assertRaises(ValueError):
            select_temporal_consensus([], 160, 100, minimum_source_frames=1)


if __name__ == "__main__":
    unittest.main()
