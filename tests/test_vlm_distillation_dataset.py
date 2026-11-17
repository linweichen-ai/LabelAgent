import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.build_vlm_distillation_dataset import build_records


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fixture(root: Path):
    instruction = "inspect"
    contract_sha = "c" * 64
    (root / "contract.json").write_text(json.dumps({
        "instruction": instruction, "contract_sha256": contract_sha}), encoding="utf-8")
    rows = root / "rows"
    rows.mkdir()
    for index in range(72):
        sequence = f"{index % 9:03d}"
        target = {"assessments": {"pallet": {"evidence": "present"}}, "unknown_candidates": []}
        text = canonical(target)
        candidate = index not in {4, 13}
        row = {
            "split": "train", "human_truth": False, "training_authorized": False,
            "requires_acceptance_gate": True, "distillation_candidate": candidate,
            "image": {"file_name": f"{sequence}_{index}.png", "source_sha256": f"{index + 1:064x}",
                      "source_path": f"/data/{sequence}_{index}.png", "sequence": sequence,
                      "frame_id": str(index), "camera": f"camera_{index % 4}",
                      "production_png_sha256": f"{index + 100:064x}"},
            "target": target, "target_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "teacher": {"model": "teacher", "model_config_sha256": "m" * 64,
                        "raw_parsed_sha256": "r" * 64},
            "review": {"consistency_flags": [] if candidate else [{"reason_code": "guard"}]},
        }
        (rows / f"{index:03d}.json").write_text(json.dumps(row), encoding="utf-8")
    audit = {"status": "passed", "train72_exact_coverage": True,
             "test24_included": False, "teacher_model_config_sha256": "m" * 64}
    return audit


class DistillationDatasetTests(unittest.TestCase):
    def test_builds_sequence_held_out_dev_and_excludes_review_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, dev, context = build_records(root, fixture(root))
            self.assertEqual(len(train), 62)
            self.assertEqual(len(dev), 8)
            self.assertEqual(len(context["excluded"]), 2)
            self.assertTrue(all(row["source_sequence"] != "008" for row in train))
            self.assertTrue(all(row["source_sequence"] == "008" for row in dev))
            self.assertTrue(all(row["human_truth"] is False for row in train + dev))

    def test_failed_audit_blocks_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = fixture(root)
            audit["status"] = "failed"
            with self.assertRaises(ValueError):
                build_records(root, audit)

    def test_accepts_collector_hash_and_preserves_training_target_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = fixture(root)
            path = root / "rows" / "000.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            row["target"] = {"z_last": 1, "a_first": 2}
            row["target_sha256"] = hashlib.sha256(
                canonical(row["target"]).encode("utf-8")).hexdigest()
            path.write_text(json.dumps(row), encoding="utf-8")

            train, _, _ = build_records(root, audit)

            changed = next(item for item in train if item["sample_id"] == "000_0")
            self.assertEqual(changed["target"], '{"z_last":1,"a_first":2}')


if __name__ == "__main__":
    unittest.main()
