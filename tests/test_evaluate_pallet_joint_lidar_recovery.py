import unittest

import numpy as np

from tools.evaluate_pallet_joint_lidar_recovery import (
    point_evidence, recovery_evidence_points,
)


class PalletJointLidarRecoveryTest(unittest.TestCase):
    def test_point_evidence_requires_both_positive_and_negative_support(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[1, 1] = True
        mask[2, 2] = True
        passed = point_evidence(mask, [[1, 1], [2, 2]], [[4, 4]])
        self.assertTrue(passed["point_gate_pass"])

        failed = point_evidence(mask, [[1, 1], [2, 2]], [[1, 1]])
        self.assertFalse(failed["point_gate_pass"])

    def test_new_receipt_requires_and_prefers_full_validation_points(self):
        row = {
            "acceptance_point_scope": "all_unique_visible_returns",
            "validation_positive_points": [[1, 1], [2, 2]],
            "validation_negative_points": [[4, 4], [5, 5]],
            "selected_positive_points": [[1, 1]],
            "selected_negative_points": [[4, 4]],
        }
        self.assertEqual(recovery_evidence_points(row), (
            row["validation_positive_points"], row["validation_negative_points"]))
        del row["validation_negative_points"]
        with self.assertRaisesRegex(ValueError, "missing validation points"):
            recovery_evidence_points(row)

    def test_legacy_receipt_fallback_is_explicit(self):
        row = {"selected_positive_points": [[1, 1]],
               "selected_negative_points": [[4, 4]]}
        self.assertEqual(recovery_evidence_points(row), (
            row["selected_positive_points"], row["selected_negative_points"]))


if __name__ == "__main__":
    unittest.main()
