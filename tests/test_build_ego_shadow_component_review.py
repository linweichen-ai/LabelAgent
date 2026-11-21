import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from tests import test_build_ego_union_denoise_shadow as denoise_tests
from tools.build_ego_component_union_review import file_sha256
from tools.build_ego_shadow_component_review import (
    _fit, build_component_review, expanded_local_box,
    extract_added_components)
from tools.build_ego_union_hole_fill_shadow import build_shadow


class EgoShadowComponentReviewTests(unittest.TestCase):
    def _fixture(self, root: Path):
        base = denoise_tests.EgoUnionDenoiseShadowTests()
        queue, diagnostic, identity, guard, union = base._fixture(root)
        sample_id = "003-camera1-12345"
        sample_dir = diagnostic / sample_id
        left = sample_dir / "00-camera1-left-seat-console-v2-mask.png"
        left_mask = np.asarray(Image.open(left).convert("L")).copy()
        left_mask[6, 3] = 0
        Image.fromarray(left_mask, mode="L").save(left)
        report_path = sample_dir / "results.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["rows"][0]["candidate_mask_sha256"] = file_sha256(left)
        report_path.write_text(json.dumps(report), encoding="utf-8")
        right = sample_dir / "01-camera1-right-hood-v2-mask.png"
        right_mask = np.asarray(Image.open(right).convert("L")) > 0
        rebuilt = np.logical_or(left_mask > 0, right_mask)
        Image.fromarray(rebuilt.astype(np.uint8) * 255, mode="L").save(union)
        queue_packet = json.loads(queue.read_text(encoding="utf-8"))
        queue_packet["samples"][0]["union_mask_sha256"] = file_sha256(union)
        queue.write_text(json.dumps(queue_packet), encoding="utf-8")
        shadow_dir = root / "shadow"
        shadow = build_shadow(
            [queue], [diagnostic], identity, guard, shadow_dir,
            maximum_hole_pixels=5, maximum_added_ratio=0.5,
            expected_size=(10, 8), require_persistence_context=False)
        self.assertEqual(shadow["filled_hole_total"], 1)
        return queue, identity, shadow_dir / "hole_fill_shadow.json"

    def test_far_components_stay_separate_and_reconstruct(self):
        added = np.zeros((20, 20), dtype=bool)
        added[1:3, 2:4] = True
        added[15:17, 16:19] = True
        rows = extract_added_components(added, maximum_pixels_exclusive=16)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["area"] for row in rows], [4, 6])
        reconstructed = np.logical_or.reduce([row["mask"] for row in rows])
        self.assertTrue(np.array_equal(reconstructed, added))

    def test_diagonal_pixels_are_one_component_under_eight_connectivity(self):
        added = np.zeros((6, 6), dtype=bool)
        added[2, 2] = True
        added[3, 3] = True
        rows = extract_added_components(added, maximum_pixels_exclusive=16)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["area"], 2)

    def test_border_component_bbox_is_clamped_by_definition(self):
        added = np.zeros((7, 9), dtype=bool)
        added[0, 0] = True
        rows = extract_added_components(added, maximum_pixels_exclusive=16)
        self.assertEqual(rows[0]["bbox"], (0, 0, 1, 1))

    def test_rejects_component_at_frozen_exclusive_limit(self):
        added = np.zeros((5, 5), dtype=bool)
        added[1:5, 1:5] = True
        with self.assertRaisesRegex(ValueError, "frozen size guard"):
            extract_added_components(added, maximum_pixels_exclusive=16)

    def test_component_order_is_raster_stable(self):
        added = np.zeros((10, 10), dtype=bool)
        added[7, 1] = True
        added[1, 8] = True
        added[1, 2] = True
        first = extract_added_components(added, 16)
        second = extract_added_components(added.copy(), 16)
        self.assertEqual([row["bbox"] for row in first],
                         [(2, 1, 3, 2), (8, 1, 9, 2), (1, 7, 2, 8)])
        self.assertEqual([row["digest"] for row in first],
                         [row["digest"] for row in second])

    def test_component_area_limit_is_strictly_exclusive(self):
        added = np.zeros((5, 5), dtype=bool)
        added[1:4, 1:5] = True
        rows = extract_added_components(added, maximum_pixels_exclusive=13)
        self.assertEqual(rows[0]["area"], 12)

    def test_micro_view_is_integer_upscaled(self):
        source = Image.new("RGB", (32, 20), (255, 0, 0))
        fitted = _fit(source, (260, 260), nearest=True)
        pixels = np.asarray(fitted)
        red = np.logical_and(pixels[:, :, 0] == 255,
                             pixels[:, :, 1] == 0)
        ys, xs = np.where(red)
        self.assertEqual((xs.max() - xs.min() + 1,
                          ys.max() - ys.min() + 1), (256, 160))

    def test_one_pixel_target_has_locator_gap(self):
        box = expanded_local_box((10, 10, 11, 11), (0, 0, 32, 32), gap=3)
        self.assertEqual(box, (7, 7, 13, 13))
        self.assertTrue(box[0] < 10 < box[2])
        self.assertTrue(box[1] < 10 < box[3])

    def test_builds_one_hash_bound_review_item(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue, identity, shadow = self._fixture(root)
            packet = build_component_review(
                [queue], identity, shadow, root / "review")
            self.assertEqual(packet["component_count"], 1)
            self.assertEqual(packet["added_pixel_total"], 1)
            self.assertEqual(packet["review_status"], "unreviewed")
            self.assertTrue((root / "review" /
                             packet["rows"][0]["review_image"]).is_file())
            self.assertFalse(packet["approved_for_training"])

    def test_render_failure_leaves_no_final_or_staging_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue, identity, shadow = self._fixture(root)
            with mock.patch(
                    "tools.build_ego_shadow_component_review._render_triptych",
                    side_effect=OSError("simulated render failure")):
                with self.assertRaisesRegex(OSError, "simulated render failure"):
                    build_component_review(
                        [queue], identity, shadow, root / "review")
            self.assertFalse((root / "review").exists())
            self.assertEqual(list(root.glob(".review.staging-*")), [])

    def test_duplicate_queue_sample_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue, identity, shadow = self._fixture(root)
            packet = json.loads(queue.read_text(encoding="utf-8"))
            packet["samples"].append(dict(packet["samples"][0]))
            queue.write_text(json.dumps(packet), encoding="utf-8")
            with self.assertRaisesRegex(ValueError,
                                        "duplicate or missing sample identity"):
                build_component_review(
                    [queue], identity, shadow, root / "review")

    def test_tampered_shadow_mask_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue, identity, shadow = self._fixture(root)
            packet = json.loads(shadow.read_text(encoding="utf-8"))
            mask = shadow.parent / packet["samples"][0]["mask"]
            Image.new("L", (10, 8), 255).save(mask)
            with self.assertRaisesRegex(ValueError, "hash-mismatched"):
                build_component_review(
                    [queue], identity, shadow, root / "review")

    def test_component_ids_and_outputs_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue, identity, shadow = self._fixture(root)
            first = build_component_review(
                [queue], identity, shadow, root / "first")
            second = build_component_review(
                [queue], identity, shadow, root / "second")
            self.assertEqual(first, second)
            self.assertEqual(
                file_sha256(root / "first" / "component_review.json"),
                file_sha256(root / "second" / "component_review.json"))


if __name__ == "__main__":
    unittest.main()
