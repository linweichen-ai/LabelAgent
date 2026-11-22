import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.audit_vlm_split_isolation import audit_split_isolation


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class VlmSplitIsolationAuditTests(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict]) -> str:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                        encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def fixture(self, root: Path):
        def rows(split, sequences):
            result = []
            for index, sequence in enumerate(sequences):
                result.append({
                    "sample_id": f"{sequence}-c{index}", "split": split,
                    "image_sha256": digest(f"{split}-image-{index}"),
                    "source_sequence": sequence, "source_frame_id": str(index),
                    "source_camera": f"camera_{index % 4}",
                    "target_sha256": digest(f"target-{index % 2}"),
                    "human_truth": False, "production_training_approved": False,
                    "requires_acceptance_gate": True,
                })
            return result
        train_rows = rows("train", [f"{index:03d}" for index in range(8)])
        dev_rows = rows("dev", ["008"])
        test_rows = rows("test24", ["010", "011", "012"])
        for row in test_rows:
            row.update(evaluation_only=True, training_return_allowed=False,
                       test24=True)
        train, dev, test = root / "train.jsonl", root / "dev.jsonl", root / "test.jsonl"
        train_hash = self.write_jsonl(train, train_rows)
        dev_hash = self.write_jsonl(dev, dev_rows)
        test_hash = self.write_jsonl(test, test_rows)
        compact = root / "compact.json"
        compact.write_text(json.dumps({
            "train_jsonl_sha256": train_hash, "dev_jsonl_sha256": dev_hash,
            "test24_included": False, "train_samples": len(train_rows),
            "dev_samples": len(dev_rows),
        }), encoding="utf-8")
        external = root / "external.json"
        external.write_text(json.dumps({
            "test24_jsonl_sha256": test_hash, "evaluation_only": True,
            "training_return_allowed": False, "samples": len(test_rows),
        }), encoding="utf-8")
        return train, dev, compact, test, external

    def test_exact_split_isolation_passes_but_near_duplicates_remain_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            report = audit_split_isolation(*paths)
        self.assertEqual(report["status"], "passed_exact_isolation")
        self.assertEqual(report["errors"], [])
        self.assertIsNone(report["near_duplicate_audit"]["passed"])
        self.assertEqual(report["splits"]["train"]["unique_image_sha256"], 8)

    def test_exact_image_leakage_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, dev, compact, test, external = self.fixture(root)
            dev_rows = [json.loads(line) for line in dev.read_text().splitlines()]
            train_row = json.loads(train.read_text().splitlines()[0])
            dev_rows[0]["image_sha256"] = train_row["image_sha256"]
            compact_value = json.loads(compact.read_text())
            compact_value["dev_jsonl_sha256"] = self.write_jsonl(dev, dev_rows)
            compact.write_text(json.dumps(compact_value), encoding="utf-8")
            report = audit_split_isolation(train, dev, compact, test, external)
        self.assertEqual(report["status"], "failed")
        self.assertIn("train_vs_dev: exact image leakage", report["errors"])

    def test_test24_boundary_and_sequence_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, dev, compact, test, external = self.fixture(root)
            test_rows = [json.loads(line) for line in test.read_text().splitlines()]
            test_rows[0]["training_return_allowed"] = True
            test_rows[0]["source_sequence"] = "009"
            external_value = json.loads(external.read_text())
            external_value["test24_jsonl_sha256"] = self.write_jsonl(test, test_rows)
            external.write_text(json.dumps(external_value), encoding="utf-8")
            report = audit_split_isolation(train, dev, compact, test, external)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("evaluation boundary" in error for error in report["errors"]))
        self.assertIn("test24: sequence membership differs", report["errors"])


if __name__ == "__main__":
    unittest.main()
