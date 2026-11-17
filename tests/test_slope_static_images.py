import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.evaluate_slope_static_images import load_inputs, summarize_presence


class SlopeStaticImageEvalTests(unittest.TestCase):
    def test_command_line_entrypoint_can_load_project(self):
        script = Path(__file__).resolve().parents[1] / "tools" / (
            "evaluate_slope_static_images.py")
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=tempfile.gettempdir(), capture_output=True, text=True,
            check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dataset-dir", result.stdout)

    def fixture(self, root: Path):
        images = root / "images"
        dataset = root / "dataset"
        images.mkdir()
        dataset.mkdir()
        rows = []
        decisions = []
        for index in range(8):
            sample_id = f"008_frame_camera_{index}"
            payload = f"image-{index}".encode()
            (images / f"{sample_id}.png").write_bytes(payload)
            rows.append({
                "sample_id": sample_id, "split": "dev",
                "source_sequence": "008", "human_truth": False,
                "production_training_approved": False,
                "image_sha256": hashlib.sha256(payload).hexdigest(),
            })
            decisions.append({
                "sample_id": sample_id,
                "assistant_visible_slope_candidate": index < 4,
                "view": "fixture",
            })
        dev = dataset / "dev.jsonl"
        dev.write_text("".join(json.dumps(row) + "\n" for row in rows),
                       encoding="utf-8")
        (dataset / "manifest.json").write_text(json.dumps({
            "status": "ready_for_isolated_lora_smoke",
            "human_truth": False, "production_training_approved": False,
            "test24_included": False, "dev_samples": 8,
            "dev_jsonl_sha256": hashlib.sha256(dev.read_bytes()).hexdigest(),
        }), encoding="utf-8")
        review = root / "review.json"
        review.write_text(json.dumps({
            "schema_version": "labelagent.slope-visual-review.v1",
            "split": "dev8", "source_sequence": "008",
            "human_truth": False, "human_confirmed": False,
            "training_authorized": False, "decisions": decisions,
        }), encoding="utf-8")
        return dataset, review, images

    def test_hash_bound_dev8_loads_without_truth_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, review, images = self.fixture(Path(directory))
            rows, metadata = load_inputs(dataset, review, images)
            self.assertEqual(len(rows), 8)
            self.assertFalse(metadata["human_truth"])

    def test_changed_image_or_test_boundary_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, review, images = self.fixture(Path(directory))
            first = next(images.iterdir())
            first.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                load_inputs(dataset, review, images)
            first.write_bytes(b"image-0")
            payload = json.loads(review.read_text())
            payload["human_truth"] = True
            review.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_inputs(dataset, review, images)

    def test_presence_summary_counts_candidate_only_comparison(self):
        rows = []
        for index in range(8):
            rows.append({
                "candidate_positive": index in {0, 1, 4},
                "assistant_visible_slope_candidate": index < 4,
                "slope_candidate_count": 1 if index in {0, 1, 4} else 0,
                "crop_fallback_used": index % 2 == 0,
                "elapsed_seconds": 0.5,
            })
        summary = summarize_presence(rows)
        self.assertEqual(summary["true_positive_candidates"], 2)
        self.assertEqual(summary["false_positive_candidates"], 1)
        self.assertEqual(summary["false_negative_candidates"], 2)
        self.assertEqual(summary["crop_fallback_images"], 4)


if __name__ == "__main__":
    unittest.main()
