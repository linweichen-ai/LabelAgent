import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.train_qwen3vl_lora import load_jsonl, verify_manifest


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


if __name__ == "__main__":
    unittest.main()
