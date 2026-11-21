from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.evaluate_ego_component_union import evaluate_image, evaluate_scope


class EgoComponentUnionTest(unittest.TestCase):
    @staticmethod
    def response(_image, points):
        positive = next(point for point in points if point["label"] == 1)
        x, y = positive["x"], positive["y"]
        return {
            "parts": [{"points": [[max(0, x - 3), max(0, y - 3)],
                                   [min(39, x + 3), max(0, y - 3)],
                                   [min(39, x + 3), min(39, y + 3)],
                                   [max(0, x - 3), min(39, y + 3)]],
                       "holes": []}],
            "score": .8,
            "warnings": ([] if len([p for p in points if p["label"] == 1]) == 1
                         else ["disconnected positives"]),
        }

    def test_component_union_can_cover_disconnected_positive_points(self):
        image = Image.new("RGB", (40, 40), "white")
        import io
        stream = io.BytesIO()
        image.save(stream, "PNG")
        baseline, union, _baseline_shape, shape, components = evaluate_image(
            stream.getvalue(), self.response)
        self.assertFalse(baseline["candidate_available"])
        self.assertTrue(union["candidate_available"])
        self.assertEqual(union["positive_points_covered"], 4)
        self.assertEqual(union["negative_points_violated"], 0)
        self.assertEqual(len(components), 4)
        self.assertTrue(shape["parts"])

    def test_scope_keeps_held_out_split_and_never_claims_truth(self):
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

    def test_one_invalid_component_is_recorded_without_aborting(self):
        import io
        calls = {"count": 0}

        def partly_invalid(image, points):
            calls["count"] += 1
            if calls["count"] == 2:
                return {"parts": [{"points": [[-10, -10], [-5, -10],
                                                [-5, -5]], "holes": []}],
                        "score": .8, "warnings": []}
            return self.response(image, points)

        stream = io.BytesIO()
        Image.new("RGB", (40, 40), "white").save(stream, "PNG")
        _baseline, union, _baseline_shape, _shape, components = evaluate_image(
            stream.getvalue(), partly_invalid)
        self.assertFalse(union["candidate_available"])
        self.assertEqual(len(components), 4)
        self.assertIn("error", components[0])

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
