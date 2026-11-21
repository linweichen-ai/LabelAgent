from io import BytesIO
import unittest
from unittest import mock
import urllib.error

from tools.diagnose_ego_guided_points import (
    materialize_normalized_point_sets, request_points, validate_point_sets)


class EgoGuidedPointDiagnosticTests(unittest.TestCase):
    def test_validates_bounded_positive_and_negative_points(self):
        payload = [{"name": "one", "points": [
            {"x": 5, "y": 6, "label": 1},
            {"x": 2, "y": 3, "label": 0},
        ]}]
        self.assertEqual(validate_point_sets(payload, 10, 10), payload)

    def test_rejects_missing_positive_duplicate_name_and_out_of_bounds(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_point_sets([{"name": "one", "points": [
                {"x": 5, "y": 6, "label": 0}]}], 10, 10)
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_point_sets([
                {"name": "same", "points": [{"x": 1, "y": 1, "label": 1}]},
                {"name": "same", "points": [{"x": 2, "y": 2, "label": 1}]},
            ], 10, 10)
        with self.assertRaisesRegex(ValueError, "invalid"):
            validate_point_sets([{"name": "one", "points": [
                {"x": 10, "y": 2, "label": 1}]}], 10, 10)

    def test_surfaces_bounded_point_sam_http_error_detail(self):
        error = urllib.error.HTTPError(
            "http://127.0.0.1:28766/segment_points", 400, "Bad Request",
            {}, BytesIO(b'{"error":"negative point outside promptable region"}'))
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(
                    RuntimeError, "negative point outside promptable region"):
                request_points("http://127.0.0.1:28766", b"image", [
                    {"x": 1, "y": 2, "label": 1}])

    def test_materializes_normalized_points_at_image_bounds(self):
        payload = [{"name": "camera-template", "points": [
            {"x_ratio": 0, "y_ratio": 0, "label": 0},
            {"x_ratio": 0.5, "y_ratio": 0.5, "label": 1},
            {"x_ratio": 1, "y_ratio": 1, "label": 1},
        ]}]
        self.assertEqual(materialize_normalized_point_sets(payload, 11, 21), [{
            "name": "camera-template", "points": [
                {"x": 0, "y": 0, "label": 0},
                {"x": 5, "y": 10, "label": 1},
                {"x": 10, "y": 20, "label": 1},
            ]}])

    def test_rejects_invalid_normalized_ratios(self):
        for invalid in (-0.01, 1.01, True, "0.5"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                    ValueError, "ratios"):
                materialize_normalized_point_sets([{
                    "name": "invalid", "points": [{
                        "x_ratio": invalid, "y_ratio": 0.5, "label": 1,
                    }],
                }], 10, 10)


if __name__ == "__main__":
    unittest.main()
