from io import BytesIO
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error

import numpy as np
from PIL import Image

from tools.diagnose_ego_guided_points import (
    materialize_normalized_point_sets, prompt_constraint_summary,
    request_point_groups, request_points, validate_point_sets,
    write_candidate_mask)


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

    def test_rejects_invalid_dimensions_boolean_labels_and_duplicate_coordinates(self):
        payload = [{"name": "one", "points": [
            {"x": 1, "y": 1, "label": 1}]}]
        for width, height in ((0, 10), (10, 0), (-1, 10), (10.0, 10),
                              (True, 10)):
            with self.subTest(width=width, height=height), self.assertRaisesRegex(
                    ValueError, "dimensions"):
                validate_point_sets(payload, width, height)
        with self.assertRaisesRegex(ValueError, "invalid"):
            validate_point_sets([{"name": "one", "points": [
                {"x": 1, "y": 1, "label": True}]}], 10, 10)
        with self.assertRaisesRegex(ValueError, "unique within"):
            validate_point_sets([{"name": "one", "points": [
                {"x": 1, "y": 1, "label": 1},
                {"x": 1, "y": 1, "label": 0},
            ]}], 10, 10)

    def test_rejects_names_that_only_differ_by_surrounding_whitespace(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_point_sets([
                {"name": "same", "points": [{"x": 1, "y": 1, "label": 1}]},
                {"name": " same ", "points": [{"x": 2, "y": 2, "label": 1}]},
            ], 10, 10)

    def test_surfaces_bounded_point_sam_http_error_detail(self):
        error = urllib.error.HTTPError(
            "http://127.0.0.1:28766/segment_points", 400, "Bad Request",
            {}, BytesIO(b'{"error":"negative point outside promptable region"}'))
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(
                    RuntimeError, "negative point outside promptable region"):
                request_points("http://127.0.0.1:28766", b"image", [
                    {"x": 1, "y": 2, "label": 1}])

    def test_rejects_malformed_point_sam_parts_and_scores(self):
        responses = (
            ({"parts": {}}, "bounded list"),
            ({"parts": ["mask"]}, "mask geometry"),
            ({"parts": [{"points": {}, "holes": []}]}, "mask geometry"),
            ({"parts": [], "score": math.nan}, "finite"),
            ({"parts": [], "score": True}, "finite"),
        )
        for payload, message in responses:
            response = mock.MagicMock()
            response.__enter__.return_value = BytesIO(
                __import__("json").dumps(payload).encode("utf-8"))
            response.__exit__.return_value = False
            with self.subTest(payload=payload), mock.patch(
                    "urllib.request.urlopen", return_value=response):
                with self.assertRaisesRegex(ValueError, message):
                    request_points("http://127.0.0.1:28766", b"image", [
                        {"x": 1, "y": 2, "label": 1}])

    def test_group_request_forwards_all_sets_to_shared_encoding_helper(self):
        point_sets = [
            {"name": "first", "points": [{"x": 1, "y": 2, "label": 1}]},
            {"name": "second", "points": [{"x": 3, "y": 4, "label": 1}]},
        ]
        expected = {"items": [], "diagnostics": {"image_encoded_once": True}}
        with mock.patch(
                "tools.diagnose_ego_guided_points.sam_point_groups",
                return_value=expected) as grouped:
            self.assertIs(
                request_point_groups(
                    "http://127.0.0.1:28766", b"image", point_sets),
                expected)
        grouped.assert_called_once_with(
            b"image",
            [point_sets[0]["points"], point_sets[1]["points"]],
            {"sam3_point_url": "http://127.0.0.1:28766",
             "sam3_url": "http://127.0.0.1:28766"},
        )

    def test_reports_positive_coverage_and_negative_violations_from_mask_crop(self):
        record = {
            "bbox": [10, 20, 14, 23],
            "mask": np.array([
                [True, False, False, False],
                [False, True, True, False],
                [False, False, False, True],
            ], dtype=bool),
        }
        points = [
            {"x": 10, "y": 20, "label": 1},
            {"x": 12, "y": 21, "label": 1},
            {"x": 9, "y": 20, "label": 1},
            {"x": 11, "y": 21, "label": 0},
            {"x": 13, "y": 20, "label": 0},
        ]
        self.assertEqual(prompt_constraint_summary(record, points), {
            "positive_point_count": 3,
            "positive_points_covered": 2,
            "uncovered_positive_points": [
                {"x": 9, "y": 20, "label": 1},
            ],
            "negative_point_count": 2,
            "negative_points_violated": 1,
            "violated_negative_points": [
                {"x": 11, "y": 21, "label": 0},
            ],
            "passed": False,
        })

    def test_prompt_constraint_pass_is_stricter_than_mask_availability(self):
        record = {
            "bbox": [0, 0, 2, 1],
            "mask": np.array([[True, False]], dtype=bool),
        }
        summary = prompt_constraint_summary(record, [
            {"x": 0, "y": 0, "label": 1},
            {"x": 1, "y": 0, "label": 1},
        ])
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["positive_points_covered"], 1)

    def test_writes_bounded_mask_crop_as_full_frame_png(self):
        record = {
            "bbox": [1, 2, 4, 4],
            "mask": np.array([
                [True, False, True],
                [False, True, False],
            ], dtype=bool),
        }
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mask.png"
            write_candidate_mask(record, 5, 6, target)
            with Image.open(target) as image:
                pixels = np.asarray(image)
            self.assertEqual(pixels.shape, (6, 5))
            self.assertEqual(int((pixels > 0).sum()), 3)
            self.assertEqual(int(pixels[2, 1]), 255)

    def test_rejects_mask_crop_with_inconsistent_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "bounded bbox"):
                write_candidate_mask({
                    "bbox": [1, 2, 4, 4],
                    "mask": np.ones((1, 3), dtype=bool),
                }, 5, 6, Path(directory) / "mask.png")

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
        for invalid in (-0.01, 1.01, True, "0.5", math.nan, math.inf, -math.inf):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                    ValueError, "ratios"):
                materialize_normalized_point_sets([{
                    "name": "invalid", "points": [{
                        "x_ratio": invalid, "y_ratio": 0.5, "label": 1,
                    }],
                }], 10, 10)

    def test_rejects_invalid_normalized_dimensions_and_boolean_label(self):
        payload = [{"name": "one", "points": [
            {"x_ratio": 0.5, "y_ratio": 0.5, "label": 1}]}]
        with self.assertRaisesRegex(ValueError, "dimensions"):
            materialize_normalized_point_sets(payload, 0, 10)
        with self.assertRaisesRegex(ValueError, "labels"):
            materialize_normalized_point_sets([{"name": "one", "points": [{
                "x_ratio": 0.5, "y_ratio": 0.5, "label": True,
            }]}], 10, 10)


if __name__ == "__main__":
    unittest.main()
