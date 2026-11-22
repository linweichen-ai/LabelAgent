import unittest

import numpy as np

from tools.evaluate_pallet_joint_lidar_recovery import point_evidence


class PalletJointLidarRecoveryTest(unittest.TestCase):
    def test_point_evidence_requires_both_positive_and_negative_support(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[1, 1] = True
        mask[2, 2] = True
        passed = point_evidence(mask, [[1, 1], [2, 2]], [[4, 4]])
        self.assertTrue(passed["point_gate_pass"])

        failed = point_evidence(mask, [[1, 1], [2, 2]], [[1, 1]])
        self.assertFalse(failed["point_gate_pass"])


if __name__ == "__main__":
    unittest.main()
