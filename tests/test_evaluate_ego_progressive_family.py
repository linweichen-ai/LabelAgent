import io
import unittest

from PIL import Image

from tools.evaluate_ego_progressive_family import (
    evaluate_progressive_image,
    fallback_order,
)


class EgoProgressiveFamilyTest(unittest.TestCase):
    def test_fallback_order_is_fixed_and_excludes_family_and_full_set(self):
        order = fallback_order(4, ((3,), (0, 2), (1, 3)))
        self.assertNotIn((3,), order)
        self.assertNotIn((0, 2), order)
        self.assertNotIn((0, 1, 2, 3), order)
        self.assertEqual(order[0], (0, 1, 2))
        self.assertEqual(order[-1], (2,))

    @staticmethod
    def response(_image, points):
        positives = [point for point in points if point["label"] == 1]
        # The fixed two singletons remain tiny.  The first fallback triple
        # returns broad boxes around its points and restores baseline coverage.
        radius = 25 if len(positives) >= 3 else 5
        parts = []
        for positive in positives:
            x, y = positive["x"], positive["y"]
            parts.append({
                "points": [[max(0, x - radius), max(0, y - radius)],
                           [min(199, x + radius), max(0, y - radius)],
                           [min(199, x + radius), min(199, y + radius)],
                           [max(0, x - radius), min(199, y + radius)]],
                "holes": [],
            })
        return {"parts": parts, "score": .8, "warnings": []}

    def test_progressive_fallback_stops_after_guard_passes(self):
        stream = io.BytesIO()
        Image.new("RGB", (200, 200), "white").save(stream, "PNG")
        baseline, result, baseline_shape, shape, attempts = evaluate_progressive_image(
            stream.getvalue(), ((0,), (1,), (2,), (3,)), self.response)
        self.assertTrue(baseline_shape["parts"])
        self.assertTrue(shape["parts"])
        self.assertTrue(result["candidate_available"])
        self.assertGreaterEqual(result["baseline_retention"], .9)
        self.assertTrue(result["fallback_used"])
        self.assertGreater(result["additional_request_count"], 4)
        self.assertLess(result["additional_request_count"], 15)
        self.assertEqual(len(attempts), result["additional_request_count"])

    def test_invalid_positive_count_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "greater than one"):
            fallback_order(1, ())


if __name__ == "__main__":
    unittest.main()
