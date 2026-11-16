from copy import deepcopy
import unittest

import numpy as np
from tools.sam_class_evidence import cached_class_masks, guard_proposal, mask_support


class ClassEvidenceChecks(unittest.TestCase):
    def test_inclusive_half_and_class_symmetry(self):
        target = np.ones((2, 2), bool)
        half = np.array([[True, True], [False, False]])
        for category, other in (("pallet", "person"), ("person", "pallet")):
            support = mask_support(target, {category: [half], other: []})
            self.assertEqual(guard_proposal(category, support)["category"], category)
            self.assertEqual(guard_proposal(other, support)["category"], "uncertain")
            self.assertEqual(guard_proposal("uncertain", support)["category"], "uncertain")

    def test_both_supported_abstains_for_either_proposal(self):
        target = np.ones((2, 2), bool)
        support = mask_support(target, {"pallet": [target], "person": [target]})
        for category in ("pallet", "person"):
            self.assertEqual(guard_proposal(category, support)["category"], "uncertain")

    def test_separate_fragments_cannot_accumulate_support(self):
        target = np.ones((2, 2), bool)
        a = np.array([[True, False], [False, False]])
        b = np.array([[False, True], [False, False]])
        support = mask_support(target, {"pallet": [a, b], "person": []})
        self.assertEqual(support.pallet_max_overlap_pixels, 1)
        self.assertEqual(guard_proposal("pallet", support)["category"], "uncertain")

    def test_nearby_object_does_not_support_selected_target(self):
        target = np.zeros((10, 10), bool); target[:2, :2] = True
        nearby = np.zeros_like(target); nearby[2:, :] = True
        support = mask_support(target, {"pallet": [nearby], "person": []})
        self.assertEqual(support.supported_classes, ())

    def test_empty_nonboolean_or_misaligned_geometry_rejected(self):
        for target in (np.zeros((2, 2), bool), np.ones((2, 2), np.uint8)):
            with self.assertRaises(ValueError):
                mask_support(target, {"pallet": [], "person": []})
        with self.assertRaises(ValueError):
            mask_support(np.ones((2, 2), bool), {"pallet": [np.ones((3, 3), bool)], "person": []})
        with self.assertRaises(ValueError):
            mask_support(np.ones((2, 2), bool), {"pallet": []})

    def receipt(self):
        return {"input_sha256": "bound-source", "frame": 3, "camera": "camera_1", "status": "ok",
                "prompt": "person", "label": "person", "masks": 1,
                "response": {"results": [{"parts": [{"points": [[0, 0], [8, 0], [8, 8], [0, 8]],
                     "holes": [[[2, 2], [4, 2], [4, 4], [2, 4]]]}]}]}}

    def masks(self, receipt):
        return cached_class_masks(receipt, image_sha256="bound-source", frame=3, camera="camera_1", size=(10, 10), category="person")

    def test_cache_identity_class_and_count_must_match(self):
        for key, value in (("input_sha256", "other-source"), ("frame", 4), ("camera", "camera_2"),
                           ("prompt", "pallet"), ("label", "pallet"), ("status", "error"), ("masks", 2)):
            receipt = self.receipt(); receipt[key] = value
            with self.assertRaises(ValueError):
                self.masks(receipt)

    def test_holes_and_invalid_polygons_are_not_solid_boxes(self):
        receipt = self.receipt()
        mask = self.masks(receipt)[0]
        self.assertFalse(mask[3, 3]); self.assertTrue(mask[1, 1])
        receipt["response"]["results"][0]["parts"][0]["points"][0][0] = -1
        with self.assertRaises(ValueError):
            self.masks(receipt)

    def test_masks_and_cache_not_mutated_and_no_replacement_class(self):
        receipt = self.receipt(); before = deepcopy(receipt)
        masks = self.masks(receipt)
        target = masks[0].copy(); pixels = target.tobytes()
        support = mask_support(target, {"person": masks, "pallet": []})
        result = guard_proposal("pallet", support)
        self.assertEqual(result["category"], "uncertain")
        self.assertEqual(receipt, before); self.assertEqual(target.tobytes(), pixels)


if __name__ == "__main__":
    unittest.main()
