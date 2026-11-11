import json
from pathlib import Path
import unittest

from tools.sam3_peft_launch_gate import ROOT, blocked_smoke


class Sam3PeftLaunchGateTests(unittest.TestCase):
    def test_blocked_manifest_and_plan_validate_without_starting_training(self):
        result = blocked_smoke(
            ROOT / "configs/sam3_sfl_cpd15y_peft_plan.json",
            ROOT / "configs/sam3_sfl_cpd15y_peft_data_manifest.blocked.json",
        )
        self.assertEqual(result["status"], "blocked_launch_gate_smoke_passed")
        self.assertFalse(result["process_started"])
        self.assertFalse(result["formal_training_authorized"])
        self.assertEqual(result["world_size"], 2)
        self.assertEqual(result["backend"], "nccl")
        self.assertIsNone(result["command_preview"])
        self.assertEqual(result["command_contract"]["num_gpus"], 2)
        self.assertFalse(result["command_contract"]["executable_hydra_config_ready"])

    def test_test24_and_fixed_holdout_remain_excluded(self):
        manifest = json.loads(
            (ROOT / "configs/sam3_sfl_cpd15y_peft_data_manifest.blocked.json")
            .read_text(encoding="utf-8")
        )
        self.assertFalse(manifest["isolation"]["test24_in_training"])
        self.assertFalse(manifest["isolation"]["fixed_640_holdout_in_training"])
        self.assertEqual(manifest["isolation"]["train_sequences"], [f"{i:03d}" for i in range(9)])
        self.assertEqual(manifest["isolation"]["test_sequences"], ["010", "011", "012"])


if __name__ == "__main__":
    unittest.main()
