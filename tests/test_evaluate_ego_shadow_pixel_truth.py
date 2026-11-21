import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.evaluate_ego_shadow_pixel_truth import evaluate_manifest


def write_mask(path: Path, box=None, *, bad_value=None):
    mask = np.zeros((12, 16), dtype=np.uint8)
    if box:
        left, top, right, bottom = box
        mask[top:bottom, left:right] = 255
    if bad_value is not None:
        mask[0, 0] = bad_value
    Image.fromarray(mask, "L").save(path)


class EgoShadowPixelTruthTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def manifest(self, samples, **overrides):
        candidate_rows = []
        reviewed_masks = []
        for sample in samples:
            item = {"sample_id": sample.get("sample_id")}
            candidate_path = self.root / str(sample.get("candidate_mask", ""))
            candidate_sha256 = (hashlib.sha256(candidate_path.read_bytes()).hexdigest()
                                if candidate_path.is_file() else "0" * 64)
            candidate_rows.append({
                "sample_id": sample.get("sample_id"),
                "camera": sample.get("camera"),
                "candidate_only": True,
                "human_truth": False,
                "approved_for_training": False,
                "eligible_for_candidate_merge": False,
                "state_written": False,
                "candidate_observed": True,
                "candidate_available": True,
                "status": "human_review_required",
                "candidate_mask": sample.get("candidate_mask"),
                "candidate_mask_sha256": candidate_sha256,
            })
            valid = True
            for manifest_key, receipt_key in (
                    ("ego_truth_mask", "ego_truth_sha256"),
                    ("person_truth_mask", "person_truth_sha256"),
                    ("floor_truth_mask", "floor_truth_sha256")):
                path = self.root / str(sample.get(manifest_key, ""))
                if not path.is_file():
                    valid = False
                    break
                item[receipt_key] = hashlib.sha256(path.read_bytes()).hexdigest()
            if valid:
                item["candidate_mask_sha256"] = candidate_sha256
                reviewed_masks.append(item)
        queue_path = self.root / "candidate-review-queue.json"
        queue_path.write_text(json.dumps({
            "schema_version": "labelagent.ego_pixel_truth_review_packet.v1",
            "candidate_only": True,
            "human_truth": False,
            "human_confirmed": False,
            "approved_for_training": False,
            "production_deployed": False,
            "formal_state_modified": False,
            "samples": candidate_rows,
        }), encoding="utf-8")
        queue_sha256 = hashlib.sha256(queue_path.read_bytes()).hexdigest()
        receipt_path = self.root / "human-review-receipt.json"
        receipt_path.write_text(json.dumps({
            "schema_version": "labelagent.ego_human_pixel_truth_receipt.v2",
            "human_confirmed": True,
            "assistant_generated": False,
            "machine_generated": False,
            "candidate_review_queue_sha256": queue_sha256,
            "reviewed_masks": reviewed_masks,
        }), encoding="utf-8")
        payload = {
            "schema_version": "labelagent.ego_pixel_truth_manifest.v2",
            "dataset_split": "validation",
            "candidate_provenance": {
                "kind": "machine_candidate_review_packet",
                "review_queue": queue_path.name,
                "review_queue_sha256": queue_sha256,
            },
            "truth_provenance": {
                "kind": "human_pixel_truth",
                "human_confirmed": True,
                "assistant_generated": False,
                "machine_generated": False,
                "confirmation_receipt": receipt_path.name,
                "confirmation_receipt_sha256": hashlib.sha256(
                    receipt_path.read_bytes()).hexdigest(),
            },
            "gates": {
                "minimum_mean_iou": .8,
                "minimum_sample_iou": .7,
                "maximum_person_overlap": .05,
                "maximum_floor_overlap": .05,
                "minimum_samples_per_camera": 1,
                "required_cameras": ["camera0"],
            },
            "samples": samples,
        }
        payload.update(overrides)
        path = self.root / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def sample(self, *, candidate=(2, 2, 10, 10), truth=(2, 2, 10, 10),
               person=None, floor=None):
        for name, box in (("candidate", candidate), ("truth", truth),
                          ("person", person), ("floor", floor)):
            write_mask(self.root / f"{name}.png", box)
        return {
            "sample_id": "camera0-frame0",
            "camera": "camera0",
            "candidate_mask": "candidate.png",
            "ego_truth_mask": "truth.png",
            "person_truth_mask": "person.png",
            "floor_truth_mask": "floor.png",
        }

    def test_exact_human_truth_match_passes_without_promoting_candidate(self):
        report = evaluate_manifest(self.manifest([self.sample()]))
        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(report["summary"]["mean_iou"], 1.0)
        self.assertTrue(report["human_truth"])
        self.assertFalse(report["approved_for_training"])
        self.assertFalse(report["eligible_for_candidate_merge"])
        self.assertNotIn("candidate_mask", report["samples"][0])

    def test_overlap_and_low_iou_fail_the_gate(self):
        report = evaluate_manifest(self.manifest([self.sample(
            candidate=(2, 2, 12, 10), truth=(2, 2, 8, 10),
            person=(8, 2, 12, 10), floor=(10, 2, 12, 10))]))
        self.assertFalse(report["gate"]["passed"])
        self.assertIn("sample_iou_below_gate", report["gate"]["reasons"])
        self.assertIn("person_overlap_above_gate", report["gate"]["reasons"])
        self.assertIn("floor_overlap_above_gate", report["gate"]["reasons"])

    def test_rejects_assistant_or_machine_truth_without_human_receipt(self):
        for provenance in (
            {"kind": "assistant_review", "human_confirmed": True,
             "assistant_generated": True, "machine_generated": False,
             "confirmation_receipt": "human-review-receipt.json",
             "confirmation_receipt_sha256": "a" * 64},
            {"kind": "human_pixel_truth", "human_confirmed": False,
             "assistant_generated": False, "machine_generated": True,
             "confirmation_receipt": "human-review-receipt.json",
             "confirmation_receipt_sha256": "a" * 64},
        ):
            with self.subTest(provenance=provenance), self.assertRaisesRegex(
                    ValueError, "human confirmation"):
                evaluate_manifest(self.manifest(
                    [self.sample()], truth_provenance=provenance))

    def test_rejects_path_escape_nonbinary_mask_and_training_split(self):
        sample = self.sample()
        sample["candidate_mask"] = "../outside.png"
        with self.assertRaisesRegex(ValueError, "escapes"):
            evaluate_manifest(self.manifest([sample]))
        sample = self.sample()
        write_mask(self.root / "candidate.png", (2, 2, 10, 10), bad_value=17)
        with self.assertRaisesRegex(ValueError, "not binary"):
            evaluate_manifest(self.manifest([sample]))
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            evaluate_manifest(self.manifest([self.sample()], dataset_split="train"))

    def test_receipt_hash_and_mask_binding_cannot_be_bypassed(self):
        manifest = self.manifest([self.sample()])
        receipt = self.root / "human-review-receipt.json"
        receipt.write_text(receipt.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "receipt hash mismatch"):
            evaluate_manifest(manifest)

        manifest = self.manifest([self.sample()])
        write_mask(self.root / "truth.png", (3, 2, 10, 10))
        with self.assertRaisesRegex(ValueError, "mask hash mismatch"):
            evaluate_manifest(manifest)

    def test_candidate_packet_and_receipt_binding_cannot_be_bypassed(self):
        sample = self.sample()
        sample["candidate_mask"] = sample["ego_truth_mask"]
        with self.assertRaisesRegex(ValueError, "paths must be distinct"):
            evaluate_manifest(self.manifest([sample]))

        manifest = self.manifest([self.sample()])
        write_mask(self.root / "candidate.png", (1, 1, 9, 9))
        with self.assertRaisesRegex(ValueError, "mask hash mismatch"):
            evaluate_manifest(manifest)

        manifest = self.manifest([self.sample()])
        queue = self.root / "candidate-review-queue.json"
        queue.write_text(queue.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "queue hash mismatch"):
            evaluate_manifest(manifest)

    def test_human_receipt_must_bind_candidate_queue_and_mask(self):
        manifest = self.manifest([self.sample()])
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        receipt = self.root / "human-review-receipt.json"
        receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
        receipt_payload["candidate_review_queue_sha256"] = "f" * 64
        receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
        manifest_payload["truth_provenance"]["confirmation_receipt_sha256"] = (
            hashlib.sha256(receipt.read_bytes()).hexdigest())
        manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "candidate queue hash mismatch"):
            evaluate_manifest(manifest)

        manifest = self.manifest([self.sample()])
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
        receipt_payload["candidate_review_queue_sha256"] = manifest_payload[
            "candidate_provenance"]["review_queue_sha256"]
        receipt_payload["reviewed_masks"][0]["candidate_mask_sha256"] = "f" * 64
        receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
        manifest_payload["truth_provenance"]["confirmation_receipt_sha256"] = (
            hashlib.sha256(receipt.read_bytes()).hexdigest())
        manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "mask hash mismatch"):
            evaluate_manifest(manifest)

    def test_legacy_unbound_manifest_and_receipt_are_rejected(self):
        manifest = self.manifest([self.sample()])
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["schema_version"] = "labelagent.ego_pixel_truth_manifest.v1"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            evaluate_manifest(manifest)

        manifest = self.manifest([self.sample()])
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        receipt = self.root / "human-review-receipt.json"
        receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
        receipt_payload["schema_version"] = (
            "labelagent.ego_human_pixel_truth_receipt.v1")
        receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
        payload["truth_provenance"]["confirmation_receipt_sha256"] = (
            hashlib.sha256(receipt.read_bytes()).hexdigest())
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "receipt content is invalid"):
            evaluate_manifest(manifest)

    def test_requires_every_declared_camera_and_unique_sample_ids(self):
        gates = {
            "minimum_mean_iou": .8, "minimum_sample_iou": .7,
            "maximum_person_overlap": .05, "maximum_floor_overlap": .05,
            "minimum_samples_per_camera": 1,
            "required_cameras": ["camera0", "camera1"],
        }
        report = evaluate_manifest(self.manifest([self.sample()], gates=gates))
        self.assertFalse(report["gate"]["passed"])
        self.assertIn("camera1:insufficient_samples", report["gate"]["reasons"])
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate_manifest(self.manifest([self.sample(), self.sample()]))


if __name__ == "__main__":
    unittest.main()
