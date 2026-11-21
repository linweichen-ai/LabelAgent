import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.evaluate_ego_subset_cover import (
    evaluate_scope,
    evaluate_subset_cover_image,
)


class EgoSubsetCoverTest(unittest.TestCase):
    @staticmethod
    def response(_image, points):
        positives = [point for point in points if point["label"] == 1]
        parts = []
        for positive in positives[:2]:
            x, y = positive["x"], positive["y"]
            parts.append({
                "points": [[max(0, x - 2), max(0, y - 2)],
                           [min(39, x + 2), max(0, y - 2)],
                           [min(39, x + 2), min(39, y + 2)],
                           [max(0, x - 2), min(39, y + 2)]],
                "holes": [],
            })
        return {"parts": parts, "score": .8, "warnings": []}

    def test_subset_cover_uses_two_pair_masks_for_four_points(self):
        stream = io.BytesIO()
        Image.new("RGB", (40, 40), "white").save(stream, "PNG")
        baseline, subset, _baseline_shape, union_shape, attempts = (
            evaluate_subset_cover_image(stream.getvalue(), self.response))
        self.assertFalse(baseline["candidate_available"])
        self.assertTrue(subset["candidate_available"])
        self.assertEqual(subset["selected_attempt_count"], 2)
        self.assertEqual(subset["positive_points_covered"], 4)
        self.assertEqual(subset["negative_points_violated"], 0)
        self.assertEqual(len(attempts), 15)
        self.assertTrue(union_shape["parts"])

    def test_scope_keeps_candidate_boundary_and_held_out_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            for frame in (0, 1):
                Image.new("RGB", (40, 40), "white").save(
                    images / f"camera1-frame{frame}.png")
            report = evaluate_scope(
                images, root / "output", self.response,
                development_frames=(0,), held_out_frames=(1,))
            self.assertEqual(report["development"]["sample_count"], 1)
            self.assertEqual(report["held_out"]["sample_count"], 1)
            self.assertFalse(report["human_truth"])
            self.assertFalse(report["approved_for_training"])
            self.assertFalse(report["production_deployed"])
            self.assertTrue((root / "output" / "contact_sheet.jpg").is_file())

    def test_scope_rejects_overlapping_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            with self.assertRaisesRegex(ValueError, "disjoint"):
                evaluate_scope(images, root / "output", self.response,
                               development_frames=(0,), held_out_frames=(0,))


if __name__ == "__main__":
    unittest.main()
