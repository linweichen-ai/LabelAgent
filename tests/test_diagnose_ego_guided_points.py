from io import BytesIO
import unittest
from unittest import mock
import urllib.error

from tools.diagnose_ego_guided_points import request_points, validate_point_sets


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


if __name__ == "__main__":
    unittest.main()
