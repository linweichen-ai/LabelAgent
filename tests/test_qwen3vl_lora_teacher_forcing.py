import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.evaluate_qwen3vl_lora_teacher_forcing import load_dev_rows, summarize


def fixture(root: Path):
    row = {
        "sample_id": "008_frame_camera_0", "split": "dev", "source_sequence": "008",
        "track": "teacher_distillation", "candidate_kind": "machine_teacher_output",
        "human_truth": False, "production_training_approved": False,
        "requires_acceptance_gate": True,
    }
    dev = root / "dev.jsonl"
    dev.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = {
        "status": "ready_for_isolated_lora_smoke", "track": "teacher_distillation",
        "human_truth": False, "production_training_approved": False,
        "requires_acceptance_gate": True, "test24_included": False,
        "dev_samples": 1,
        "dev_jsonl_sha256": hashlib.sha256(dev.read_bytes()).hexdigest(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return row, manifest


class Qwen3VlLoraTeacherForcingTests(unittest.TestCase):
    def test_dev_boundary_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            rows, _ = load_dev_rows(root)
            self.assertEqual(rows[0]["source_sequence"], "008")

    def test_test24_boundary_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = fixture(root)
            manifest["test24_included"] = True
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dev_rows(root)

    def test_summary_is_token_weighted(self):
        result = summarize([
            {"supervised_tokens": 2, "nll_sum": 1.0, "correct_tokens": 1,
             "elapsed_seconds": 0.2},
            {"supervised_tokens": 8, "nll_sum": 7.0, "correct_tokens": 2,
             "elapsed_seconds": 0.3},
        ])
        self.assertAlmostEqual(result["mean_nll"], 0.8)
        self.assertAlmostEqual(result["token_accuracy"], 0.3)
        self.assertAlmostEqual(result["elapsed_seconds"], 0.5)


if __name__ == "__main__":
    unittest.main()
