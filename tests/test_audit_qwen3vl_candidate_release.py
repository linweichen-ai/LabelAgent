import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.audit_qwen3vl_candidate_release import audit, sensitive_paths


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class CandidateReleaseAuditTests(unittest.TestCase):
    def fixture(self, root: Path):
        for directory in ("dataset", "training/adapter"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        train, dev, adapter = b"train\n", b"dev\n", b"adapter"
        (root / "dataset/train.jsonl").write_bytes(train)
        (root / "dataset/dev.jsonl").write_bytes(dev)
        (root / "training/adapter/adapter_model.safetensors").write_bytes(adapter)
        dataset = {"test24_included": False, "human_truth": False,
                   "production_training_approved": False, "train_samples": 1,
                   "dev_samples": 1, "train_jsonl_sha256": digest(train),
                   "dev_jsonl_sha256": digest(dev)}
        receipt = {"status": "training_complete", "track": "teacher_distillation",
                   "human_truth": False, "production_deployed": False,
                   "test24_used_for_training": False, "steps": 2,
                   "train_jsonl_sha256": digest(train),
                   "adapter_sha256": digest(adapter)}
        tensor = {"status": "passed", "adapter_sha256": digest(adapter)}
        config = {"r": 8}
        summary = {"human_truth": False, "production_deployed": False,
                   "test24_used_for_training": False,
                   "training": {"adapter_sha256": digest(adapter)}}
        values = (("dataset/manifest.json", dataset),
                  ("training/run_receipt.json", receipt),
                  ("training/audit.json", tensor),
                  ("training/adapter/adapter_config.json", config),
                  ("summary.json", summary))
        for name, value in values:
            (root / name).write_text(json.dumps(value), encoding="utf-8")
        source = root / "source.py"
        source.write_text("pass\n", encoding="utf-8")
        return root / "summary.json", source

    def test_valid_offline_candidate_is_manifested_without_publish_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, source = self.fixture(root)
            value = audit(root, summary, [source])
        self.assertEqual(value["status"], "passed_local_release_audit")
        self.assertFalse(value["boundaries"]["weights_published"])
        self.assertFalse(value["boundaries"]["distribution_approved"])

    def test_test24_or_adapter_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, source = self.fixture(root)
            receipt = json.loads((root / "training/run_receipt.json").read_text())
            receipt["test24_used_for_training"] = True
            (root / "training/run_receipt.json").write_text(json.dumps(receipt))
            value = audit(root, summary, [source])
        self.assertEqual(value["status"], "failed")
        self.assertIn("test24 entered training", value["errors"])

    def test_sensitive_json_values_are_rejected_but_empty_placeholders_are_not(self):
        self.assertEqual(sensitive_paths({"api_key": "secret-value"}), ["$.api_key"])
        self.assertEqual(sensitive_paths({"api_key": "", "token": None}), [])


if __name__ == "__main__":
    unittest.main()
