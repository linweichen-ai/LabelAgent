import io
import unittest

import numpy as np

from PIL import Image

from tools.evaluate_reference_point_scan import (
    artifact_frame_token, bounded_mask_record, fixed_point_grid,
    isolated_mask_image, known_coverage, known_coverage_record,
    isolated_semantic_route, merge_exemplar_results, multiscale_panel,
    supported_other_robot)
from tools.tiled_candidate_utils import _mask_record


class ReferencePointScanTests(unittest.TestCase):
    def test_default_grid_is_resolution_scaled_and_bounded_to_top_two_thirds(self):
        points = fixed_point_grid(1920, 1536)
        self.assertEqual(len(points), 184)
        self.assertEqual(points[0], {"x": 80, "y": 128})
        self.assertEqual(points[-1], {"x": 1840, "y": 1024})
        self.assertEqual(len({(item["x"], item["y"]) for item in points}), 184)

    def test_known_coverage_measures_candidate_containment_not_scene_iou(self):
        candidate = np.zeros((8, 8), dtype=bool)
        candidate[2:6, 2:6] = True
        half = np.zeros_like(candidate)
        half[2:6, 2:4] = True
        result = known_coverage(candidate, 16, {"person": [half], "pallet": []})
        self.assertEqual(result, {"person": 0.5, "pallet": 0.0})

    def test_bounded_raster_is_pixel_identical_to_full_image_record(self):
        shape = {
            "points": [[-3, 2], [15, 2], [15, 14], [-3, 14]],
            "holes": [[[4, 5], [8, 5], [8, 9], [4, 9]]],
            "parts": [
                {"points": [[-3, 2], [15, 2], [15, 14], [-3, 14]],
                 "holes": [[[4, 5], [8, 5], [8, 9], [4, 9]]]},
                {"points": [[20, 4], [27, 4], [27, 10], [20, 10]],
                 "holes": []},
            ],
        }
        expected = _mask_record(shape, 24, 18)
        actual = bounded_mask_record(shape, 24, 18)
        self.assertEqual(actual["bbox"], expected["bbox"])
        self.assertEqual(actual["area"], expected["area"])
        np.testing.assert_array_equal(actual["mask"], expected["mask"])
        known = np.zeros((18, 24), dtype=bool)
        known[2:9, 0:6] = True
        full = np.zeros_like(known)
        x0, y0, x1, y1 = expected["bbox"]
        full[y0:y1, x0:x1] = expected["mask"].astype(bool)
        self.assertEqual(
            known_coverage_record(actual, {"person": [known]}),
            known_coverage(full, actual["area"], {"person": [known]}))

    def test_multiscale_panel_is_fixed_size_and_marks_the_candidate_at_every_scale(self):
        shape = {"id": 1, "label": "unknown_object", "bbox": [40, 30, 80, 60],
                 "points": [[40, 30], [80, 30], [80, 60], [40, 60]],
                 "holes": [], "parts": [{"points": [[40, 30], [80, 30],
                                                        [80, 60], [40, 60]],
                                            "holes": []}]}
        data = multiscale_panel(Image.new("RGB", (160, 120), "white"), shape, "C01")
        with Image.open(io.BytesIO(data)) as rendered:
            self.assertEqual(rendered.size, (1024, 768))

    def test_supported_other_robot_is_semantic_vote_only(self):
        valid = {"class": "other_robot"}
        self.assertTrue(supported_other_robot(valid))
        self.assertFalse(supported_other_robot({"class": "pallet"}))

    def test_long_static_frame_id_uses_short_traceable_artifact_token(self):
        token = artifact_frame_token(
            "capture1103-camera0-t100-positive-with-extra-context")
        self.assertRegex(token, r"^id-[0-9a-f]{12}$")
        self.assertEqual(artifact_frame_token(74), "74")

    def test_isolated_mask_image_hides_surrounding_pixels(self):
        shape = {"id": 1, "label": "unknown_object", "bbox": [40, 30, 80, 60],
                 "points": [[40, 30], [80, 30], [80, 60], [40, 60]],
                 "holes": [], "parts": [{"points": [[40, 30], [80, 30],
                                                        [80, 60], [40, 60]],
                                            "holes": []}]}
        data = isolated_mask_image(Image.new("RGB", (160, 120), "white"), shape)
        with Image.open(io.BytesIO(data)) as rendered:
            self.assertEqual(rendered.size, (512, 384))
            self.assertLess(max(rendered.getpixel((0, 0))), 150)

    def test_adaptive_exemplar_skip_cannot_promote_baseline_candidate(self):
        shape = {"id": 1}
        baseline = [
            {"id": "C01", "result": {"class": "pallet"},
             "artifact": "baseline-1", "_shape": shape},
            {"id": "C02", "result": {"class": "other_robot"},
             "artifact": "baseline-2", "_shape": shape},
        ]
        exemplar = [
            {"id": "C02", "result": {"class": "other_robot"},
             "artifact": "exemplar-2", "_shape": shape},
        ]
        merged = merge_exemplar_results(baseline, exemplar, allow_subset=True)
        self.assertFalse(merged[0]["exemplar_evaluated"])
        self.assertFalse(merged[0]["dual_pass_consensus"])
        self.assertTrue(merged[1]["exemplar_evaluated"])
        self.assertTrue(merged[1]["dual_pass_consensus"])

    def test_isolated_semantic_route_requires_literal_robot_evidence(self):
        self.assertEqual(
            isolated_semantic_route({"class": "other_robot"}),
            "auto_candidate")
        self.assertEqual(
            isolated_semantic_route({"class": "unknown_object"}),
            "review_required")
        for semantic_class in ("pallet", "vehicle_or_forklift", "ego_robot_part",
                               "tool_or_equipment", "floor_or_structure"):
            self.assertEqual(
                isolated_semantic_route({"class": semantic_class}),
                "explicit_negative")
        self.assertEqual(isolated_semantic_route(None), "not_evaluated")


if __name__ == "__main__":
    unittest.main()
