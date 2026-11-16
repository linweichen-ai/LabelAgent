from copy import deepcopy
import unittest

import numpy as np
from source.agent.edit_agent import _rasterize
from tools.sam_class_evidence import cached_class_masks, guard_proposal, mask_support


class CropGeometryChecks(unittest.TestCase):
    def test_offset_and_holes_match_full_source_crop(self):
        geometry = {"points": [[13, 14], [24, 14], [24, 25], [13, 25]],
                    "holes": [[[16, 17], [19, 17], [19, 20], [16, 20]]]}
        full = _rasterize(geometry, 40, 40).astype(bool)
        local = deepcopy(geometry)
        for ring in [local["points"], *local["holes"]]:
            for p in ring:
                p[0] -= 10; p[1] -= 11
        self.assertTrue(np.array_equal(full[11:31, 10:30], _rasterize(local, 20, 20).astype(bool)))

    def test_original_image_hash_cannot_stand_in_for_crop_hash(self):
        receipt = {"input_sha256": "original-full-image", "frame": 1, "camera": "camera_0",
                   "status": "ok", "prompt": "pallet", "label": "pallet", "masks": 0,
                   "response": {"results": []}}
        with self.assertRaises(ValueError):
            cached_class_masks(receipt, image_sha256="actual-crop-bytes", frame=1,
                               camera="camera_0", size=(20, 20), category="pallet")

    def test_local_support_can_fill_fullframe_miss(self):
        target = np.ones((2, 2), bool); miss = np.zeros_like(target)
        support = mask_support(target, {"pallet": [miss, target], "person": []})
        self.assertEqual(guard_proposal("pallet", support)["category"], "pallet")

    def test_new_opposite_local_support_causes_abstention(self):
        target = np.ones((2, 2), bool)
        support = mask_support(target, {"pallet": [target], "person": [target]})
        self.assertEqual(guard_proposal("pallet", support)["category"], "uncertain")


if __name__ == "__main__":
    unittest.main()
