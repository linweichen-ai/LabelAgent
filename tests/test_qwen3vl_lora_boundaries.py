import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.train_qwen3vl_lora import (
    OneImageCollator, find_evidence_token_span, load_jsonl, slope_balanced_rows,
    verify_manifest,
)
from tools.run_qwen3vl_lora_shadow import verify_shadow_manifest


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
    def test_evidence_token_span_is_exact(self):
        class Tokenizer:
            pieces = {1: '{"evidence":"', 2: "present", 3: '"}'}

            def decode(self, values, skip_special_tokens=False):
                return "".join(self.pieces[value] for value in values)

        self.assertEqual(find_evidence_token_span(Tokenizer(), [1, 2, 3], "present"),
                         (1, 2))

    def test_parallel_wrappers_preserve_services_and_gate_shared_gpu(self):
        root = Path(__file__).parents[1]
        train = (root / "tools" / "run_remote_qwen3vl_lora_parallel.sh").read_text(
            encoding="utf-8")
        evaluate = (root / "tools" / "run_remote_qwen3vl_adapter_static_eval_parallel.sh").read_text(
            encoding="utf-8")
        teacher_force = (root / "tools" / "run_remote_qwen3vl_lora_eval_parallel.sh").read_text(
            encoding="utf-8")
        generation = (root / "tools" / "run_remote_qwen3vl_lora_generation_eval_parallel.sh").read_text(
            encoding="utf-8")
        shadow = (root / "tools" / "run_remote_qwen3vl_lora_shadow_parallel.sh").read_text(
            encoding="utf-8")
        for script in (train, evaluate, teacher_force, generation, shadow):
            self.assertIn("MINIMUM_FREE_MIB=12000", script)
            self.assertIn("SAM_INFLIGHT", script)
            self.assertIn("flock -n 9", script)
            self.assertIn("vlm-health-after.json", script)
            self.assertNotIn("tmux kill-session", script)
        self.assertIn('--seed "$SEED"', train)
        self.assertIn('TRITON_CACHE_DIR="$TRITON_CACHE_DIR"', evaluate)

    def test_shadow_manifest_remains_candidate_only(self):
        contract = {"contract_sha256": "abc"}
        manifest = {
            "schema_version": "labelagent.qwen3vl-shadow-manifest.v1",
            "human_truth": False,
            "candidate_only": True,
            "production_deployed": False,
            "contract_sha256": "abc",
            "images": [{
                "sample_id": "new-sequence-1",
                "image_path": "/tmp/image.png",
                "image_sha256": "0" * 64,
                "assistant_review_candidate": {
                    "review_kind": "assistant_visual_candidate",
                    "human_truth": False,
                    "expected_evidence": "absent",
                    "reason": "Only the ego forks are visible.",
                },
            }],
        }
        self.assertEqual(verify_shadow_manifest(manifest, contract), manifest["images"])
        manifest["human_truth"] = True
        with self.assertRaises(ValueError):
            verify_shadow_manifest(manifest, contract)

    def test_shadow_manifest_rejects_duplicate_samples(self):
        image = {"sample_id": "duplicate", "image_path": "/tmp/image.png",
                 "image_sha256": "0" * 64}
        manifest = {
            "schema_version": "labelagent.qwen3vl-shadow-manifest.v1",
            "human_truth": False, "candidate_only": True,
            "production_deployed": False, "contract_sha256": "abc",
            "images": [image, dict(image)],
        }
        with self.assertRaises(ValueError):
            verify_shadow_manifest(manifest, {"contract_sha256": "abc"})

    def test_shadow_manifest_rejects_assistant_review_promoted_to_truth(self):
        manifest = {
            "schema_version": "labelagent.qwen3vl-shadow-manifest.v1",
            "human_truth": False, "candidate_only": True,
            "production_deployed": False, "contract_sha256": "abc",
            "images": [{
                "sample_id": "candidate", "image_path": "/tmp/image.png",
                "image_sha256": "0" * 64,
                "assistant_review_candidate": {
                    "review_kind": "assistant_visual_candidate",
                    "human_truth": True, "expected_evidence": "absent",
                    "reason": "Only the ego vehicle is visible.",
                },
            }],
        }
        with self.assertRaises(ValueError):
            verify_shadow_manifest(manifest, {"contract_sha256": "abc"})

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

    def test_balancing_supports_an_explicit_project_label(self):
        def row(sample_id, evidence):
            return {"sample_id": sample_id, "target": json.dumps({
                "assessments": {"other_robot": {"evidence": evidence}}})}

        rows, report = slope_balanced_rows(
            [row("positive", "present"), row("negative", "absent")],
            2, "other_robot")
        self.assertEqual(len(rows), 3)
        self.assertEqual(report["positive_label"], "other_robot")
        self.assertNotIn("slope_positive_repeat", report)

        rows, report = slope_balanced_rows(
            [row("positive", "present"), row("negative", "absent")],
            3, "other_robot", "absent")
        self.assertEqual([item["sample_id"] for item in rows].count("negative"), 3)
        self.assertEqual(report["repeat_evidence"], "absent")

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

    def test_assistant_review_candidate_training_row_stays_non_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            train, _, row, _ = write_fixture(Path(directory))
            row["candidate_kind"] = "assistant_review_candidate"
            train.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(load_jsonl(train)[0]["candidate_kind"],
                             "assistant_review_candidate")
            row["production_training_approved"] = True
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
