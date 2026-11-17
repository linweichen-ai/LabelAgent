import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.build_qwen3vl_slope_prompt_dataset import (
    SLOPE_GUIDANCE, build_rows, load_dev_rows,
)
from tools.score_qwen3vl_slope_review import score


class Qwen3VlSlopePromptTests(unittest.TestCase):
    def rows(self):
        return [{
            "sample_id": f"s{index}", "split": "dev",
            "source_sequence": "008", "track": "teacher_distillation",
            "candidate_kind": "machine_teacher_output", "human_truth": False,
            "production_training_approved": False,
            "requires_acceptance_gate": True, "instruction": "original",
        } for index in range(8)]

    def write_dataset(self, root: Path):
        rows = self.rows()
        dev = root / "dev.jsonl"
        dev.write_text("".join(json.dumps(row) + "\n" for row in rows),
                       encoding="utf-8")
        manifest = {
            "status": "ready_for_isolated_lora_smoke",
            "track": "teacher_distillation", "human_truth": False,
            "production_training_approved": False,
            "requires_acceptance_gate": True, "test24_included": False,
            "dev_samples": 8,
            "dev_jsonl_sha256": hashlib.sha256(dev.read_bytes()).hexdigest(),
        }
        (root / "manifest.json").write_text(json.dumps(manifest),
                                             encoding="utf-8")

    def test_builder_adds_geometry_guidance_without_truth_promotion(self):
        variant = build_rows(self.rows())
        self.assertEqual(len(variant), 8)
        self.assertTrue(all(SLOPE_GUIDANCE in row["instruction"]
                            for row in variant))
        self.assertTrue(all(row["prompt_variant_human_truth"] is False
                            for row in variant))

    def test_loader_rejects_test_sequence_or_hash_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_dataset(root)
            rows, _ = load_dev_rows(root)
            self.assertEqual(len(rows), 8)
            rows[0]["source_sequence"] = "010"
            dev = root / "dev.jsonl"
            dev.write_text("".join(json.dumps(row) + "\n" for row in rows),
                           encoding="utf-8")
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["dev_jsonl_sha256"] = hashlib.sha256(
                dev.read_bytes()).hexdigest()
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_dev_rows(root)

    def test_scorer_keeps_assistant_review_boundary_and_counts(self):
        result_rows = []
        decisions = []
        for index in range(8):
            visible = index < 4
            present = index in {0, 1, 4}
            result_rows.append({
                "sample_id": f"s{index}", "status": "ok",
                "content": {"assessments": {"slope": {
                    "evidence": "present" if present else "absent"}}},
            })
            decisions.append({
                "sample_id": f"s{index}",
                "assistant_visible_slope_candidate": visible,
            })
        review = {
            "schema_version": "labelagent.slope-visual-review.v1",
            "human_truth": False, "human_confirmed": False,
            "split": "dev8", "source_sequence": "008",
            "decisions": decisions,
        }
        summary = score({"per_sample": result_rows}, review)
        self.assertEqual(summary["true_positive_candidates"], 2)
        self.assertEqual(summary["false_positive_candidates"], 1)
        self.assertEqual(summary["false_negative_candidates"], 2)
        review["human_truth"] = True
        with self.assertRaises(ValueError):
            score({"per_sample": result_rows}, review)


if __name__ == "__main__":
    unittest.main()
