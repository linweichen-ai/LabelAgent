import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.audit_ego_union_pixels import (
    DEFAULT_THRESHOLDS, build_audit, mask_metrics,
    package_balanced_reference, threshold_alerts)
from tools.build_ego_component_union_review import file_sha256


class EgoUnionPixelAuditTests(unittest.TestCase):
    def _fixture(self, root: Path, packages=("000", "001", "002")):
        identities = {}
        queue_specs = []
        rows_by_package = {}
        for package_index, package in enumerate(packages):
            queue_dir = root / f"queue-{package}"
            queue_dir.mkdir(parents=True)
            rows = []
            for camera in ("camera1", "camera3"):
                for frame_offset in range(2):
                    frame_id = str(10000 + package_index * 100 + int(camera[-1]) * 10
                                   + frame_offset)
                    sample_id = f"{package}-{camera}-{frame_id}"
                    image_path = root / "dataset" / package / f"camera_{camera[-1]}" / f"{frame_id}.png"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (10, 8),
                              (60 + package_index * 10, 80, 100)).save(image_path)
                    mask = np.zeros((8, 10), dtype=np.uint8)
                    if camera == "camera1":
                        mask[4:8, 1:5] = 255
                    else:
                        mask[5:7, 2:8] = 255
                    if frame_offset:
                        mask[5, 5] = 255
                    mask_path = queue_dir / f"{sample_id}-union-mask.png"
                    Image.fromarray(mask, mode="L").save(mask_path)
                    image_sha = file_sha256(image_path)
                    identities[sample_id] = {
                        "package": package,
                        "camera": camera,
                        "frame_id": frame_id,
                        "image_sha256": image_sha,
                    }
                    rows.append({
                        "sample_id": sample_id,
                        "union_available": True,
                        "source_image": str(image_path),
                        "source_image_sha256": image_sha,
                        "union_mask": mask_path.name,
                        "union_mask_sha256": file_sha256(mask_path),
                    })
            rows_by_package[package] = rows
            queue_specs.append((queue_dir, rows))
        manifest_path = root / "identity.json"
        manifest_path.write_text(json.dumps({"samples": identities}), encoding="utf-8")
        manifest_sha = file_sha256(manifest_path)
        queue_paths = []
        for queue_dir, rows in queue_specs:
            queue_path = queue_dir / "review_queue.json"
            queue_path.write_text(json.dumps({
                "candidate_only": True,
                "human_truth": False,
                "approved_for_training": False,
                "production_deployed": False,
                "safety_gates": {
                    "allowed_source_root": str((root / "dataset").resolve()),
                    "component_area_guard_enabled": True,
                    "identity_manifest_enabled": True,
                    "identity_manifest_sha256": manifest_sha,
                },
                "samples": rows,
            }), encoding="utf-8")
            queue_paths.append(queue_path)
        return queue_paths, manifest_path, rows_by_package

    def test_metrics_compute_iou_recall_and_excess(self):
        candidate = np.array([[1, 1], [0, 1]], dtype=bool)
        reference = np.array([[1, 0], [1, 1]], dtype=bool)
        metrics = mask_metrics(candidate, reference, small_island_pixels=2)
        self.assertEqual(metrics["intersection_pixels"], 2)
        self.assertEqual(metrics["reference_iou"], 0.5)
        self.assertEqual(metrics["reference_recall"], 0.666667)
        self.assertEqual(metrics["candidate_excess"], 0.333333)

    def test_metrics_reject_empty_or_mismatched_masks(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            mask_metrics(np.zeros((2, 2)), np.ones((2, 2)))
        with self.assertRaisesRegex(ValueError, "share a 2D shape"):
            mask_metrics(np.ones((2, 2)), np.ones((3, 2)))

    def test_threshold_breach_only_creates_alerts(self):
        metrics = {
            "reference_iou": 0.1, "reference_recall": 0.2,
            "candidate_excess": 0.9, "small_island_area_ratio": 0.2,
            "hole_area_ratio": 0.2,
        }
        alerts = threshold_alerts(metrics, "camera1", DEFAULT_THRESHOLDS)
        self.assertEqual(len(alerts), 5)

    def test_package_balanced_reference_does_not_overweight_more_frames(self):
        base = np.zeros((4, 4), dtype=bool)
        common = base.copy()
        common[1:3, 1:3] = True
        noisy = common.copy()
        noisy[0, 0] = True
        reference, required, counts = package_balanced_reference({
            "001": [common],
            "002": [common],
            "003": [noisy, noisy, noisy, noisy, noisy],
        })
        self.assertEqual(required, 2)
        self.assertEqual(counts, {"001": 1, "002": 1, "003": 5})
        self.assertFalse(reference[0, 0])
        self.assertTrue(np.array_equal(reference[1:3, 1:3],
                                       np.ones((2, 2), dtype=bool)))

    def test_package_internal_tie_fails_closed(self):
        clear = np.zeros((2, 2), dtype=bool)
        marked = clear.copy()
        marked[0, 0] = True
        reference, _, _ = package_balanced_reference({
            "001": [clear, marked],
            "002": [clear, marked],
        })
        self.assertFalse(reference[0, 0])

    def test_builds_candidate_only_audit_and_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queues, manifest, _ = self._fixture(root)
            audit = build_audit(queues, manifest, root / "audit")
            self.assertEqual(audit["sample_count"], 12)
            self.assertTrue(audit["candidate_only"])
            self.assertFalse(audit["human_truth"])
            self.assertFalse(audit["approved_for_training"])
            self.assertFalse(audit["production_deployed"])
            self.assertEqual(
                audit["reference_strategy"],
                "package_balanced_two_stage_majority")
            self.assertTrue((root / "audit" / "audit.json").is_file())
            self.assertTrue((root / "audit" / audit["contact_sheet"]).is_file())

    def test_requires_three_packages_for_leave_one_out(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queues, manifest, _ = self._fixture(root, packages=("000", "001"))
            with self.assertRaisesRegex(ValueError, "at least three packages"):
                build_audit(queues, manifest, root / "audit")

    def test_rejects_union_mask_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queues, manifest, _ = self._fixture(root)
            packet = json.loads(queues[0].read_text(encoding="utf-8"))
            packet["samples"][0]["union_mask_sha256"] = "0" * 64
            queues[0].write_text(json.dumps(packet), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "union mask is missing or hash-mismatched"):
                build_audit(queues, manifest, root / "audit")

    def test_rejects_duplicate_sample_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queues, manifest, _ = self._fixture(root)
            first = json.loads(queues[0].read_text(encoding="utf-8"))["samples"][0]
            packet = json.loads(queues[1].read_text(encoding="utf-8"))
            packet["samples"].append(first)
            queues[1].write_text(json.dumps(packet), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate sample identity"):
                build_audit(queues, manifest, root / "audit")

    def test_rejects_identity_binding_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queues, manifest, _ = self._fixture(root)
            packet = json.loads(queues[0].read_text(encoding="utf-8"))
            packet["samples"][0]["sample_id"] = "000-camera1-99999"
            queues[0].write_text(json.dumps(packet), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity binding mismatch"):
                build_audit(queues, manifest, root / "audit")

    def test_rejects_manifest_sample_not_present_in_queues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queues, manifest, _ = self._fixture(root)
            packet = json.loads(queues[0].read_text(encoding="utf-8"))
            packet["samples"].pop()
            queues[0].write_text(json.dumps(packet), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity manifest differ"):
                build_audit(queues, manifest, root / "audit")

    def test_preserves_all_input_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queues, manifest, _ = self._fixture(root)
            inputs = [manifest, *queues]
            before = {str(path): file_sha256(path) for path in inputs}
            build_audit(queues, manifest, root / "audit")
            after = {str(path): file_sha256(path) for path in inputs}
            self.assertEqual(before, after)

    def test_contact_sheet_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queues, manifest, _ = self._fixture(root)
            first = build_audit(queues, manifest, root / "audit-a")
            second = build_audit(queues, manifest, root / "audit-b")
            self.assertEqual(first["contact_sheet_sha256"],
                             second["contact_sheet_sha256"])


if __name__ == "__main__":
    unittest.main()
