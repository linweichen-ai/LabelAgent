import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from tests import test_build_ego_union_denoise_shadow as denoise_tests
from tools.build_ego_component_union_review import file_sha256
from tools.build_ego_union_hole_fill_shadow import (
    added_pixels_within_budget, build_shadow, fill_small_holes)


class EgoUnionHoleFillShadowTests(unittest.TestCase):
    def test_fills_small_enclosed_hole(self):
        mask = np.ones((7, 7), dtype=bool)
        mask[3, 3] = False
        filled, metrics = fill_small_holes(mask, [], maximum_pixels=5)
        self.assertTrue(filled[3, 3])
        self.assertEqual(metrics["added_pixels"], 1)
        self.assertEqual(metrics["filled_hole_count"], 1)

    def test_does_not_fill_border_background(self):
        mask = np.ones((7, 7), dtype=bool)
        mask[0, 3] = False
        filled, metrics = fill_small_holes(mask, [], maximum_pixels=5)
        self.assertFalse(filled[0, 3])
        self.assertEqual(metrics["added_pixels"], 0)

    def test_negative_point_protects_hole(self):
        mask = np.ones((7, 7), dtype=bool)
        mask[3, 3] = False
        filled, metrics = fill_small_holes(mask, [(3, 3)], maximum_pixels=5)
        self.assertFalse(filled[3, 3])
        self.assertEqual(metrics["negative_point_protected_hole_count"], 1)

    def test_persistent_region_protects_hole(self):
        mask = np.ones((7, 7), dtype=bool)
        mask[3, 3] = False
        persistent = np.zeros_like(mask)
        persistent[3, 3] = True
        filled, metrics = fill_small_holes(
            mask, [], maximum_pixels=5, persistent_protection=persistent)
        self.assertFalse(filled[3, 3])
        self.assertEqual(metrics["persistent_protected_hole_count"], 1)

    def test_threshold_is_strictly_less_than_maximum(self):
        mask = np.ones((8, 8), dtype=bool)
        mask[3:5, 3:5] = False
        filled, _ = fill_small_holes(mask, [], maximum_pixels=4)
        self.assertFalse(filled[3, 3])

    def test_hole_fill_is_idempotent(self):
        mask = np.ones((7, 7), dtype=bool)
        mask[3, 3] = False
        first, _ = fill_small_holes(mask, [], maximum_pixels=5)
        second, _ = fill_small_holes(first, [], maximum_pixels=5)
        self.assertTrue(np.array_equal(first, second))

    def test_integer_budget_does_not_round_up_boundary(self):
        self.assertTrue(added_pixels_within_budget(299, 29999, 0.01))
        self.assertFalse(added_pixels_within_budget(300, 29999, 0.01))
        with self.assertRaisesRegex(ValueError, "settings are invalid"):
            added_pixels_within_budget(1, 10, float("nan"))

    def test_builds_reversible_candidate_only_shadow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = denoise_tests.EgoUnionDenoiseShadowTests()
            queue, diagnostic, identity, guard, union = fixture._fixture(root)
            before = file_sha256(union)
            packet = build_shadow(
                [queue], [diagnostic], identity, guard, root / "output",
                maximum_hole_pixels=5, maximum_added_ratio=0.1,
                expected_size=(10, 8), require_persistence_context=False)
            self.assertEqual(packet["sample_count"], 1)
            self.assertEqual(packet["removed_pixel_total"], 0)
            self.assertFalse(packet["source_masks_modified"])
            self.assertEqual(before, file_sha256(union))

    def test_rejects_component_addition_over_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = denoise_tests.EgoUnionDenoiseShadowTests()
            queue, diagnostic, identity, guard, _ = fixture._fixture(root)
            with self.assertRaisesRegex(ValueError, "settings are invalid"):
                build_shadow(
                    [queue], [diagnostic], identity, guard, root / "output",
                    maximum_hole_pixels=100, maximum_added_ratio=-1.0,
                    expected_size=(10, 8), require_persistence_context=False)
            self.assertFalse((root / "output").exists())

    def test_write_failure_leaves_no_final_or_staging_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = denoise_tests.EgoUnionDenoiseShadowTests()
            queue, diagnostic, identity, guard, _ = fixture._fixture(root)
            with mock.patch(
                    "tools.build_ego_union_hole_fill_shadow._write_overlay",
                    side_effect=OSError("simulated disk failure")):
                with self.assertRaisesRegex(OSError, "simulated disk failure"):
                    build_shadow(
                        [queue], [diagnostic], identity, guard, root / "output",
                        maximum_hole_pixels=5, maximum_added_ratio=0.1,
                        expected_size=(10, 8), require_persistence_context=False)
            self.assertFalse((root / "output").exists())
            self.assertEqual(list(root.glob(".output.staging-*")), [])

    def test_rejects_source_outside_declared_allowed_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = denoise_tests.EgoUnionDenoiseShadowTests()
            queue, diagnostic, identity, guard, _ = fixture._fixture(root)
            packet = json.loads(queue.read_text(encoding="utf-8"))
            other = root / "other"
            other.mkdir()
            packet["safety_gates"]["allowed_source_root"] = str(other.resolve())
            queue.write_text(json.dumps(packet), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside allowed root"):
                build_shadow(
                    [queue], [diagnostic], identity, guard, root / "output",
                    maximum_hole_pixels=5, maximum_added_ratio=0.1,
                    expected_size=(10, 8), require_persistence_context=False)


if __name__ == "__main__":
    unittest.main()
