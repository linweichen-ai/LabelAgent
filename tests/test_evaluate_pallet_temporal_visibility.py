import unittest

from tools.evaluate_pallet_temporal_visibility import (
    projected_overlap, temporal_occlusion_gate,
)


class PalletTemporalVisibilityTest(unittest.TestCase):
    def test_temporal_gate_requires_both_sides_and_visibility_recovery(self):
        self.assertTrue(temporal_occlusion_gate(0, 2, 1, 11, 11, 0, 16))
        self.assertFalse(temporal_occlusion_gate(0, 2, 0, 11, 11, 0, 16))
        self.assertFalse(temporal_occlusion_gate(0, 2, 1, 10, 11, 0, 16))
        self.assertFalse(temporal_occlusion_gate(0, 2, 1, 11, 11, 10, 12))

    def test_projected_overlap_handles_missing_projection(self):
        self.assertEqual(projected_overlap(None, None), 0.0)


if __name__ == "__main__":
    unittest.main()
