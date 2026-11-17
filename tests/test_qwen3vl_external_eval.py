import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.build_qwen3vl_external_eval_dataset import LABELS, build_rows
from tools.evaluate_qwen3vl_external_generation import (
    load_external_rows, verify_external_images,
)


class Qwen3VlExternalEvalTests(unittest.TestCase):
    def sources(self):
        images, runs = [], []
        for index in range(24):
            sequence = ["010", "011", "012"][index // 8]
            digest = f"{index:064x}"
            images.append({
                "image_id": index + 1, "split": "test", "sequence": sequence,
                "frame_id": str(index), "camera": f"camera_{index % 4}",
                "file_name": f"{sequence}_{index}.png",
                "source_path": f"/data/{sequence}/{index}.png",
                "source_sha256": digest,
            })
            runs.append({
                "image_sha256": digest, "failure": None,
                "assessments": [{"label": label, "evidence": "absent",
                                  "confidence": 0.9} for label in LABELS],
                "unknown_candidates": [],
            })
        return {"images": images}, {
            "runs": runs, "success_count": 24, "failure_count": 0,
            "labels": LABELS, "model": "teacher",
        }

    def test_builder_accepts_only_complete_test24(self):
        pack, teacher = self.sources()
        rows = build_rows(pack, teacher, "instruction", "a" * 64)
        self.assertEqual(len(rows), 24)
        self.assertTrue(all(row["evaluation_only"] for row in rows))
        self.assertTrue(all(not row["training_return_allowed"] for row in rows))

    def test_builder_rejects_non_test_sequence(self):
        pack, teacher = self.sources()
        pack["images"][0]["sequence"] = "009"
        with self.assertRaises(ValueError):
            build_rows(pack, teacher, "instruction", "a" * 64)

    def test_loader_rejects_training_return(self):
        pack, teacher = self.sources()
        rows = build_rows(pack, teacher, "instruction", "a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows_path = root / "test24.jsonl"
            rows[0]["training_return_allowed"] = True
            rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                                 encoding="utf-8")
            manifest = {
                "schema_version": "labelagent.qwen3vl-external-eval-manifest.v1",
                "status": "frozen_external_evaluation_only", "split": "test24",
                "evaluation_only": True, "training_return_allowed": False,
                "human_truth": False, "production_training_approved": False,
                "test24_included": True, "samples": 24,
                "sequences": ["010", "011", "012"],
                "test24_jsonl_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_external_rows(root)

    def test_external_image_verification_binds_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(b"image")
            row = {"sample_id": "x", "image_path": str(path),
                   "image_sha256": hashlib.sha256(b"image").hexdigest()}
            self.assertEqual(verify_external_images([row])["sha256_matches"], 1)
            row["image_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                verify_external_images([row])


if __name__ == "__main__":
    unittest.main()
