import tempfile
from pathlib import Path
import unittest

import numpy as np

from tests import test_build_ego_union_denoise_shadow as denoise_tests
from tools.build_ego_component_union_review import file_sha256
from tools.build_ego_union_hole_fill_shadow import build_shadow, fill_small_holes


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
            with self.assertRaisesRegex(ValueError, "addition budget exceeded"):
                build_shadow(
                    [queue], [diagnostic], identity, guard, root / "output",
                    maximum_hole_pixels=100, maximum_added_ratio=-1.0,
                    expected_size=(10, 8), require_persistence_context=False)
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
