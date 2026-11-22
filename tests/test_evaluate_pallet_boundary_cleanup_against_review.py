import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.evaluate_pallet_boundary_cleanup_against_review import evaluate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PalletBoundaryReviewComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pack = self.root / "pack"
        self.pack.mkdir()
        truth = [{"id": 21, "label": "pallet", "points": [[2, 2], [11, 2],
                  [11, 11], [2, 11]], "holes": [], "parts": [{"points":
                  [[2, 2], [11, 2], [11, 11], [2, 11]], "holes": []}]}]
        self.file = "reviewed.mcap"
        self.topic = "/camera"
        sessions = {self.file: {"topic": self.topic,
                    "truth": {"0": truth}}}
        self.sessions = self.root / "sessions.json"
        self.sessions.write_text(json.dumps(sessions), encoding="utf-8")
        truth_hash = hashlib.sha256(json.dumps(
            truth, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        review = {"file": self.file, "topic": self.topic, "frame": 0,
                  "frame_sha256": "f" * 64, "expected_frame_sha256": "f" * 64,
                  "truth_sha256": truth_hash, "decision": "correct_candidate",
                  "training_return_allowed": False, "review_id": "review",
                  "identity_assurance": "self_declared", "truth_role":
                  "workbench_human_truth_candidate", "dataset_split": "unassigned",
                  "split_policy": "blocked_unassigned"}
        self.review = self.root / "review.json"
        self.review.write_text(json.dumps(review), encoding="utf-8")
        current = np.zeros((16, 16), dtype=np.uint8)
        current[2:12, 2:12] = 255
        refined = current.copy(); refined[2, 2:5] = 0
        for name, mask in (("current.png", current), ("refined.png", refined)):
            Image.fromarray(mask).save(self.pack / name)
        manifest = {
            "schema_version": "labelagent.candidate-comparison-pack.v1",
            "human_truth": False, "training_authorized": False,
            "rows": [{"frame": 0, "source_frame_sha256": "f" * 64,
                      "current_candidate_id": 7,
                      "current": {"mask": "current.png",
                                  "mask_sha256": digest(self.pack / "current.png")},
                      "variants": [{"mask": "refined.png",
                                    "mask_sha256": digest(self.pack / "refined.png")}]}]}
        self.manifest = self.pack / "manifest.json"
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def run_evaluation(self, name="out"):
        return evaluate(
            self.pack, self.sessions, self.review, self.root / name,
            expected_pack_sha256=digest(self.manifest),
            expected_sessions_sha256=digest(self.sessions),
            expected_review_sha256=digest(self.review))

    def test_fixed_match_reports_regression_without_authorizing_training(self):
        report = self.run_evaluation()
        self.assertEqual(report["status"], "development_gate_failed")
        self.assertFalse(report["development_gate_passed"])
        self.assertLess(report["aggregate"]["iou"]["delta"], 0)
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["generalization_claim_allowed"])
        self.assertFalse(report["production_deployment_authorized"])
        self.assertTrue((self.root / "out/contact_sheet.png").exists())
        public = json.loads((self.root / "out/public_receipt.json").read_text())
        self.assertNotIn("rows", public)
        self.assertFalse(public["private_instance_ids_published"])

    def test_pack_hash_drift_fails_before_output(self):
        with self.assertRaisesRegex(ValueError, "pack manifest SHA-256 differs"):
            evaluate(self.pack, self.sessions, self.review, self.root / "bad",
                     expected_pack_sha256="0" * 64,
                     expected_sessions_sha256=digest(self.sessions),
                     expected_review_sha256=digest(self.review))
        self.assertFalse((self.root / "bad").exists())

    def test_source_frame_drift_fails_closed(self):
        value = json.loads(self.manifest.read_text())
        value["rows"][0]["source_frame_sha256"] = "e" * 64
        self.manifest.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source frame differs"):
            self.run_evaluation("frame-drift")
        self.assertFalse((self.root / "frame-drift").exists())


if __name__ == "__main__":
    unittest.main()
