import io
import unittest
from copy import deepcopy

import numpy as np
from PIL import Image
from tools.evaluate_confirmed_region_context import aggregate, expanded_context_views


class ContextChecks(unittest.TestCase):
    def image(self):
        pixels = np.arange(240 * 200 * 3, dtype=np.uint32).reshape(200, 240, 3).astype(np.uint8)
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format="PNG")
        return stream.getvalue(), pixels

    def shape(self):
        return {"id": 1, "label": "pallet", "points": [[0, 0], [23, 0], [23, 19], [0, 19]],
                "holes": [[[4, 4], [8, 4], [8, 8], [4, 8]]], "parts": []}

    def test_boundary_crop_keeps_native_pixels_and_whole_selection(self):
        raw, pixels = self.image()
        images, meta = expanded_context_views(raw, self.shape())
        self.assertEqual(meta["crop_box"], [0, 0, 160, 160])
        np.testing.assert_array_equal(np.asarray(images[0]), pixels[:160, :160])
        self.assertGreater(meta["selected_pixels"], 0)
        self.assertFalse(meta["image_resized"])

    def test_category_id_score_cannot_change_model_pixels(self):
        raw, _ = self.image()
        changed = deepcopy(self.shape())
        changed.update(id=999, label="person", score=0.01, source="opposite role")
        a, ma = expanded_context_views(raw, self.shape())
        b, mb = expanded_context_views(raw, changed)
        self.assertEqual(ma, mb)
        self.assertEqual([i.tobytes() for i in a], [i.tobytes() for i in b])

    def test_context_clips_to_small_source_without_invented_pixels(self):
        image = Image.new("RGB", (40, 30), "green")
        stream = io.BytesIO(); image.save(stream, format="PNG")
        views, meta = expanded_context_views(stream.getvalue(), self.shape())
        self.assertEqual(views[0].size, image.size)
        self.assertEqual(views[0].tobytes(), image.tobytes())
        self.assertEqual(meta["crop_box"], [0, 0, 40, 30])

    def test_incomplete_or_abstaining_pair_cannot_be_correct(self):
        cases = [{"id": "x", "expected_category": "pallet"}]
        rows = [{"case_id": "x", "view": "single", "parsed": {"category": "pallet"}},
                {"case_id": "x", "view": "context", "parsed": {"category": "pallet"}}]
        result = aggregate(cases, rows)
        self.assertEqual(result["tight_context"]["correct_consensus"], 1)
        self.assertEqual(result["expanded_context"]["correct_consensus"], 0)
        self.assertEqual(result["expanded_context"]["valid_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
