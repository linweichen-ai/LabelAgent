import unittest

from tools.check_sam3_training_policy import check_policy, find_forbidden
from tools.sam3_trainable_inventory import set_trainable_stage
from tests.test_sam3_trainable_inventory import FakeSam3


class Sam3TrainingPolicyGateTests(unittest.TestCase):
    def test_current_policy_has_no_obsolete_scope_claims(self):
        self.assertEqual(check_policy(), [])

    def test_gate_detects_old_predeclared_stage_wording(self):
        self.assertNotEqual(find_forbidden("stage-1：开放完整 segmentation_head"), [])

    def test_inventory_tool_rejects_stage_above_zero(self):
        with self.assertRaisesRegex(ValueError, "only Stage-0 is authorized"):
            set_trainable_stage(FakeSam3(), 1)


if __name__ == "__main__":
    unittest.main()
