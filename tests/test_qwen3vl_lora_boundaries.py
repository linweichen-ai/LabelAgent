import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.train_qwen3vl_lora import (
    OneImageCollator, load_jsonl, slope_balanced_rows, verify_manifest,
)


def write_fixture(root: Path):
    row = {
        "split": "train", "track": "teacher_distillation",
        "candidate_kind": "machine_teacher_output", "human_truth": False,
        "production_training_approved": False, "requires_acceptance_gate": True,
    }
    train = root / "train.jsonl"
    train.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = {
        "status": "ready_for_isolated_lora_smoke", "track": "teacher_distillation",
        "human_truth": False, "production_training_approved": False,
        "requires_acceptance_gate": True, "test24_included": False,
        "train_jsonl_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return train, manifest_path, row, manifest


class Qwen3VlLoraBoundaryTests(unittest.TestCase):
    def test_slope_balancing_repeats_only_positive_rows(self):
        def row(sample_id, evidence):
            return {
                "sample_id": sample_id,
                "target": json.dumps({"assessments": {
                    "slope": {"evidence": evidence}}}),
            }

        rows, report = slope_balanced_rows(
            [row("positive", "present"), row("negative", "absent")], 8)
        self.assertEqual([item["sample_id"] for item in rows].count("positive"), 8)
        self.assertEqual([item["sample_id"] for item in rows].count("negative"), 1)
        self.assertEqual(report["effective_samples"], 9)

    def test_slope_balancing_rejects_missing_evidence(self):
        with self.assertRaises(ValueError):
            slope_balanced_rows([{"target": "{}"}], 2)

    def test_valid_teacher_distillation_fixture_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            train, manifest_path, _, _ = write_fixture(Path(directory))
            self.assertEqual(len(load_jsonl(train)), 1)
            self.assertFalse(verify_manifest(manifest_path, train)["test24_included"])

    def test_test24_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            train, manifest_path, _, manifest = write_fixture(Path(directory))
            manifest["test24_included"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_manifest(manifest_path, train)

    def test_promoted_training_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            train, _, row, _ = write_fixture(Path(directory))
            row["human_truth"] = True
            train.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_jsonl(train)

    def test_collator_matches_production_text_then_image_order(self):
        import torch
        from PIL import Image

        class Processor:
            def __init__(self):
                self.messages = []

            def apply_chat_template(self, messages, **_):
                self.messages.append(messages)
                return "full" if len(messages) == 2 else "prompt"

            def __call__(self, *, text, images, **_):
                del images
                ids = [1, 2, 3, 4] if text == ["full"] else [1, 2]
                return {"input_ids": torch.tensor([ids])}

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (8, 8), (1, 2, 3)).save(image_path)
            row = {
                "image_path": str(image_path),
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "instruction": "inspect", "target": "{}",
            }
            processor = Processor()
            batch = OneImageCollator(processor)([row])
            content = processor.messages[0][0]["content"]
            self.assertEqual([item["type"] for item in content], ["text", "image"])
            self.assertEqual(batch["labels"].tolist(), [[-100, -100, 3, 4]])


if __name__ == "__main__":
    unittest.main()
