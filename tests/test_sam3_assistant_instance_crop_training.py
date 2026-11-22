import ast
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class AssistantInstanceCropTrainingTests(unittest.TestCase):
    def test_trainer_has_hard_safety_boundaries(self):
        path = ROOT / "source/sam3/sam3/train/assistant_instance_crop_trainer.py"
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("I_ACKNOWLEDGE_ASSISTANT_REVIEW_IS_NOT_HUMAN_TRUTH", source)
        self.assertIn('"human_truth": False', source)
        self.assertIn('"test24_in_training": False', source)
        self.assertIn("EXPECTED_STEPS = 36", source)
        self.assertIn("assistant_pallet_crop_mask_embed.pt", source)

    def test_stage0_detaches_the_fully_frozen_backbone(self):
        source = (
            ROOT / "source/sam3/sam3/train/stage0_scope.py"
        ).read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("_freeze_backbone_execution", source)
        self.assertIn('(\"forward_image\", \"forward_text\")', source)
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("Stage-0 backbone unexpectedly has trainable parameters", source)

    def test_config_is_one_epoch_and_candidate_only(self):
        path = ROOT / "source/sam3/sam3/train/configs/sfl_cpd15y/stage0_assistant_instance_crop_epoch.yaml"
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        trainer = value["trainer"]
        self.assertEqual(value["roboflow_train"]["num_images"], 71)
        self.assertEqual(trainer["mode"], "train_only")
        self.assertEqual(trainer["max_epochs"], 1)
        self.assertEqual(trainer["max_optimizer_steps"], 36)
        self.assertFalse(trainer["candidate_contract"]["human_truth"])
        self.assertFalse(trainer["candidate_contract"]["test24_in_training"])
        self.assertTrue(trainer["candidate_contract"]["not_for_deployment"])

        base_path = ROOT / "source/sam3/sam3/train/configs/sfl_cpd15y/stage0_mask_embed.yaml"
        base_text = base_path.read_text(encoding="utf-8")
        self.assertIn(
            "/configs/roboflow_v100/roboflow_v100_full_ft_100_images",
            base_text,
        )
        self.assertTrue(
            (
                ROOT
                / "source/sam3/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml"
            ).is_file()
        )

    def test_launcher_keeps_services_and_test24_guard(self):
        source = (ROOT / "tools/run_sam3_assistant_instance_crop_train.sh").read_text()
        self.assertIn("127.0.0.1:${port}/health", source)
        self.assertIn('[[ "$SAM3_STAGE0_TRAIN_ROOT" != *test24* ]]', source)
        self.assertIn("memory.free", source)
        self.assertNotIn("kill ", source)
        self.assertNotIn("systemctl", source)


if __name__ == "__main__":
    unittest.main()
