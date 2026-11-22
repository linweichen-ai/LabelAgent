import ast
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class AssistantCropHoldoutTests(unittest.TestCase):
    def test_evaluator_is_evaluation_only(self):
        path = ROOT / "source/sam3/sam3/train/assistant_crop_holdout_evaluator.py"
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("assistant holdout can never train", source)
        self.assertIn('"human_truth": False', source)
        self.assertIn('"test24_included": False', source)
        self.assertIn("_assistant_delegated_pallet_crops_holdout15.coco.json", source)

    def test_config_has_no_train_dataset_and_dumps_masks(self):
        path = ROOT / "source/sam3/sam3/train/configs/sfl_cpd15y/stage0_assistant_crop_holdout_predictions.yaml"
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        trainer = value["trainer"]
        self.assertEqual(trainer["mode"], "val")
        self.assertIsNone(trainer["data"]["train"])
        self.assertEqual(trainer["data"]["val"]["dataset"]["limit_ids"], 15)
        self.assertFalse(trainer["evaluation_contract"]["human_truth"])
        self.assertFalse(trainer["evaluation_contract"]["test24_included"])
        self.assertIsNone(
            trainer["meters"]["val"]["roboflow100"]["detection"][
                "pred_file_evaluators"
            ]
        )


if __name__ == "__main__":
    unittest.main()
