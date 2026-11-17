import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.evaluate_qwen3vl_merged_generation import (
    compare_reference, semantic_signature,
)
from tools.merge_qwen3vl_lora import checkpoint_manifest, validate_inputs


class Qwen3VlLoraMergeTests(unittest.TestCase):
    def fixture(self, root: Path):
        model = root / "base"
        adapter = root / "adapter"
        model.mkdir()
        adapter.mkdir()
        (model / "config.json").write_text(
            json.dumps({"model_type": "qwen3_vl"}), encoding="utf-8")
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        weights = adapter / "adapter_model.safetensors"
        weights.write_bytes(b"audited adapter")
        return model, adapter, hashlib.sha256(weights.read_bytes()).hexdigest()

    def test_merge_inputs_bind_audited_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, adapter, digest = self.fixture(root)
            self.assertEqual(
                validate_inputs(model, adapter, root / "merged", digest), digest)
            with self.assertRaises(ValueError):
                validate_inputs(model, adapter, root / "merged", "0" * 64)

    def test_existing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, adapter, digest = self.fixture(root)
            output = root / "merged"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                validate_inputs(model, adapter, output, digest)

    def test_checkpoint_manifest_hashes_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights.safetensors").write_bytes(b"weights")
            (root / "merge_receipt.json").write_text("ignored", encoding="utf-8")
            rows = checkpoint_manifest(root)
            self.assertEqual([row["path"] for row in rows], ["weights.safetensors"])
            self.assertEqual(rows[0]["bytes"], 7)

    def test_semantic_parity_ignores_explanation_text_only(self):
        left = {"assessments": {"floor": {
            "evidence": "present", "reason": "a"}}, "unknown_candidates": []}
        right = {"assessments": {"floor": {
            "evidence": "present", "reason": "b"}}, "unknown_candidates": []}
        self.assertEqual(semantic_signature(left), semantic_signature(right))

    def test_reference_comparison_requires_complete_semantic_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            reference_path = Path(directory) / "reference.json"
            content = {"assessments": {"floor": {"evidence": "present"}},
                       "unknown_candidates": []}
            reference_path.write_text(json.dumps({"per_sample": {"adapter": [{
                "sample_id": "a", "status": "ok", "content_sha256": "same",
                "content": content,
            }]}}), encoding="utf-8")
            result = compare_reference([{
                "sample_id": "a", "status": "ok", "content_sha256": "same",
                "content": content,
            }], reference_path)
            self.assertTrue(result["all_exact"])
            self.assertTrue(result["all_semantic"])

    def test_reference_comparison_detects_one_semantic_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            reference_path = Path(directory) / "reference.json"
            reference_path.write_text(json.dumps({"per_sample": {"adapter": [{
                "sample_id": "a", "status": "ok", "content_sha256": "left",
                "content": {"assessments": {"other_robot": {
                    "evidence": "present"}}, "unknown_candidates": []},
            }]}}), encoding="utf-8")
            result = compare_reference([{
                "sample_id": "a", "status": "ok", "content_sha256": "right",
                "content": {"assessments": {"other_robot": {
                    "evidence": "absent"}}, "unknown_candidates": []},
            }], reference_path)
            self.assertFalse(result["all_exact"])
            self.assertFalse(result["all_semantic"])


if __name__ == "__main__":
    unittest.main()
