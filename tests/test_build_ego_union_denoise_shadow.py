import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.build_ego_component_union_review import file_sha256
from tools.build_ego_union_denoise_shadow import (
    build_shadow, denoise_small_islands)


class EgoUnionDenoiseShadowTests(unittest.TestCase):
    def _fixture(self, root: Path):
        dataset = root / "dataset" / "003" / "camera_1"
        dataset.mkdir(parents=True)
        image_path = dataset / "12345.png"
        Image.new("RGB", (10, 8), (70, 90, 110)).save(image_path)
        image_sha = file_sha256(image_path)
        sample_id = "003-camera1-12345"
        identity_path = root / "identity.json"
        identity_path.write_text(json.dumps({"samples": {sample_id: {
            "package": "003", "camera": "camera1", "frame_id": "12345",
            "image_sha256": image_sha,
        }}}), encoding="utf-8")
        guard_path = root / "guard.json"
        guard_path.write_text(json.dumps({"components": {
            "camera1-left-seat-console-v2": {
                "min_area_ratio": 0.0, "max_area_ratio": 1.0},
            "camera1-right-hood-v2": {
                "min_area_ratio": 0.0, "max_area_ratio": 1.0},
        }}), encoding="utf-8")
        diagnostic_root = root / "diagnostic"
        sample_dir = diagnostic_root / sample_id
        sample_dir.mkdir(parents=True)
        components = []
        masks = []
        definitions = (
            ("camera1-left-seat-console-v2", (slice(4, 8), slice(1, 5)),
             [(2, 5)], [(8, 1)], (9, 0)),
            ("camera1-right-hood-v2", (slice(4, 7), slice(6, 9)),
             [(7, 5)], [(0, 0)], None),
        )
        for index, (name, slices, positives, negatives, speck) in enumerate(definitions):
            mask = np.zeros((8, 10), dtype=np.uint8)
            mask[slices] = 255
            if speck is not None:
                mask[speck[1], speck[0]] = 255
            mask_path = sample_dir / f"{index:02d}-{name}-mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)
            masks.append(mask > 0)
            components.append({
                "name": name,
                "candidate_available": True,
                "prompt_constraints_passed": True,
                "candidate_mask": mask_path.name,
                "candidate_mask_sha256": file_sha256(mask_path),
                "points": ([{"x": x, "y": y, "label": 1} for x, y in positives]
                           + [{"x": x, "y": y, "label": 0} for x, y in negatives]),
            })
        report_path = sample_dir / "results.json"
        report_path.write_text(json.dumps({
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "production_deployed": False,
            "source_state_modified": False,
            "image_sha256": image_sha,
            "rows": components,
        }), encoding="utf-8")
        queue_dir = root / "queue"
        queue_dir.mkdir()
        union_path = queue_dir / f"{sample_id}-union-mask.png"
        Image.fromarray(np.logical_or.reduce(masks).astype(np.uint8) * 255,
                        mode="L").save(union_path)
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
                "identity_manifest_sha256": file_sha256(identity_path),
            },
            "samples": [{
                "sample_id": sample_id,
                "source_image": str(image_path),
                "source_image_sha256": image_sha,
                "union_mask": union_path.name,
                "union_mask_sha256": file_sha256(union_path),
            }],
        }), encoding="utf-8")
        return queue_path, diagnostic_root, identity_path, guard_path, union_path

    def test_removes_small_unprotected_component(self):
        mask = np.zeros((8, 10), dtype=bool)
        mask[2:5, 2:5] = True
        mask[0, 9] = True
        cleaned, metrics = denoise_small_islands(mask, [(3, 3)], 5)
        self.assertFalse(cleaned[0, 9])
        self.assertEqual(metrics["removed_pixels"], 1)

    def test_preserves_small_component_with_positive_point(self):
        mask = np.zeros((8, 10), dtype=bool)
        mask[2:5, 2:5] = True
        mask[0, 9] = True
        cleaned, metrics = denoise_small_islands(mask, [(9, 0)], 5)
        self.assertTrue(cleaned[0, 9])
        self.assertEqual(metrics["removed_pixels"], 0)

    def test_threshold_is_strictly_less_than_minimum(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[1:11, 1:11] = True
        cleaned, _ = denoise_small_islands(mask, [(5, 5)], 100)
        self.assertEqual(int(cleaned.sum()), 100)

    def test_rejects_positive_point_outside_mask(self):
        mask = np.zeros((8, 10), dtype=bool)
        mask[2:5, 2:5] = True
        with self.assertRaisesRegex(ValueError, "outside the source mask"):
            denoise_small_islands(mask, [(0, 0)], 5)

    def test_denoise_is_idempotent(self):
        mask = np.zeros((8, 10), dtype=bool)
        mask[2:5, 2:5] = True
        mask[0, 9] = True
        first, _ = denoise_small_islands(mask, [(3, 3)], 5)
        second, _ = denoise_small_islands(first, [(3, 3)], 5)
        self.assertTrue(np.array_equal(first, second))

    def test_persistent_region_protects_small_component(self):
        mask = np.zeros((8, 10), dtype=bool)
        mask[2:5, 2:5] = True
        mask[0, 9] = True
        persistent = np.zeros_like(mask)
        persistent[0, 9] = True
        cleaned, metrics = denoise_small_islands(
            mask, [(3, 3)], 5, persistent)
        self.assertTrue(cleaned[0, 9])
        self.assertEqual(metrics["persistent_protected_component_count"], 1)

    def test_builds_reversible_candidate_only_shadow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue, diagnostic, identity, guard, union = self._fixture(root)
            before = file_sha256(union)
            packet = build_shadow(
                [queue], [diagnostic], identity, guard, root / "output",
                minimum_pixels=5, maximum_removed_ratio=0.1,
                expected_size=(10, 8), require_temporal_context=False)
            self.assertEqual(packet["sample_count"], 1)
            self.assertEqual(packet["removed_pixel_total"], 1)
            self.assertEqual(packet["added_pixel_total"], 0)
            self.assertTrue(packet["positive_points_preserved_all"])
            self.assertTrue(packet["negative_points_preserved_all"])
            self.assertFalse(packet["source_masks_modified"])
            self.assertEqual(before, file_sha256(union))

    def test_rejects_resolution_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue, diagnostic, identity, guard, _ = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "unexpected component mask dimensions"):
                build_shadow([queue], [diagnostic], identity, guard,
                             root / "output")

    def test_requires_temporal_package_context_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue, diagnostic, identity, guard, _ = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "insufficient packages"):
                build_shadow(
                    [queue], [diagnostic], identity, guard, root / "output",
                    minimum_pixels=5, maximum_removed_ratio=0.1,
                    expected_size=(10, 8))

    def test_fails_closed_when_component_budget_is_exceeded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue, diagnostic, identity, guard, _ = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "removal budget exceeded"):
                build_shadow(
                    [queue], [diagnostic], identity, guard, root / "output",
                    minimum_pixels=5, maximum_removed_ratio=0.0,
                    expected_size=(10, 8), require_temporal_context=False)

    def test_rejects_union_not_reconstructed_from_components(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue, diagnostic, identity, guard, union = self._fixture(root)
            bad = np.zeros((8, 10), dtype=np.uint8)
            bad[1, 1] = 255
            Image.fromarray(bad, mode="L").save(union)
            packet = json.loads(queue.read_text(encoding="utf-8"))
            packet["samples"][0]["union_mask_sha256"] = file_sha256(union)
            queue.write_text(json.dumps(packet), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not reconstruct union"):
                build_shadow(
                    [queue], [diagnostic], identity, guard, root / "output",
                    minimum_pixels=5, maximum_removed_ratio=0.1,
                    expected_size=(10, 8), require_temporal_context=False)

    def test_contact_sheet_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue, diagnostic, identity, guard, _ = self._fixture(root)
            first = build_shadow(
                [queue], [diagnostic], identity, guard, root / "first",
                minimum_pixels=5, maximum_removed_ratio=0.1,
                expected_size=(10, 8), require_temporal_context=False)
            second = build_shadow(
                [queue], [diagnostic], identity, guard, root / "second",
                minimum_pixels=5, maximum_removed_ratio=0.1,
                expected_size=(10, 8), require_temporal_context=False)
            self.assertEqual(first["contact_sheet_sha256"],
                             second["contact_sheet_sha256"])


if __name__ == "__main__":
    unittest.main()
