import io
import unittest

from PIL import Image

from tools.diagnose_slope_sam_dense_crops import dense_crop_images, summarize


class SlopeSamDenseCropTests(unittest.TestCase):
    def test_dense_crops_cover_all_edges_deterministically(self):
        buffer = io.BytesIO()
        Image.new("RGB", (640, 512), "black").save(buffer, "PNG")
        crops = dense_crop_images(buffer.getvalue(), 4, 0.35)
        self.assertEqual(len(crops), 16)
        self.assertEqual(crops[0]["bbox"][:2], [0, 0])
        self.assertEqual(crops[-1]["bbox"][2:], [640, 512])
        self.assertTrue(all(crop["width"] == 224 for crop in crops))
        self.assertTrue(all(crop["height"] == 179 for crop in crops))

    def test_summary_exposes_false_positives(self):
        rows = [{
            "assistant_visible_slope_candidate": index < 4,
            "prompts": {key: {"candidate_count": int(index in {0, 1, 4})}
                        for key in ("baseline", "ramp")},
        } for index in range(8)]
        result = summarize(rows)
        self.assertEqual(result["baseline"]["true_positive_images"], 2)
        self.assertEqual(result["baseline"]["false_positive_images"], 1)


if __name__ == "__main__":
    unittest.main()
