"""Offline coordinate/protocol checks; no model or annotation writes."""
import copy
import unittest

import numpy as np
from PIL import Image

from native_resume_policy import PLAN_PROMPT, bind_points, planning_views, validate_decision
from visible_base_native_policy import planning_views as original_native_views


def decision(positive=None, negative=None):
    return {
        "object": "visible object",
        "instance_scope": "single_visible_object",
        "positive_points": positive if positive is not None else [{"x": 500, "y": 500}, {"x": 700, "y": 700}],
        "negative_points": negative if negative is not None else [],
        "evidence": "Separate visible structural parts in the first image.",
    }


def binding(box):
    return {"source_box": box, "coordinate_system": "normalized_edge_0_1000"}


class NativeResumeTests(unittest.TestCase):
    def test_center_and_asymmetric_coordinates_map_to_source_edges_once(self):
        seed = np.zeros((1200, 1500), np.uint8)
        seed[562, 612] = 1
        points, why = bind_points(decision(), binding([100, 50, 1124, 1074]), seed)
        self.assertIsNone(why)
        self.assertEqual(points, [{"x": 612, "y": 562, "label": 1}, {"x": 817, "y": 767, "label": 1}])

    def test_bottom_and_right_edges_rejected_without_clamping(self):
        seed = np.ones((1024, 1024), np.uint8)
        for point in ({"x": 1000, "y": 500}, {"x": 500, "y": 1000}):
            with self.subTest(point=point):
                result, why = bind_points(decision([{"x": 500, "y": 500}, point]), binding([0, 0, 1024, 1024]), seed)
                self.assertIsNone(result)
                self.assertIn("outside", why)

    def test_top_and_left_edges_are_valid_if_in_source(self):
        seed = np.zeros((100, 100), np.uint8)
        seed[0, 0] = 1
        result, why = bind_points(decision([{"x": 0, "y": 0}, {"x": 500, "y": 500}]), binding([0, 0, 100, 100]), seed)
        self.assertIsNone(why)
        self.assertEqual(result[0], {"x": 0, "y": 0, "label": 1})

    def test_rounding_out_of_source_rejected_instead_of_snapped(self):
        seed = np.ones((100, 100), np.uint8)
        result, why = bind_points(decision([{"x": 500, "y": 500}, {"x": 999, "y": 500}]), binding([0, 0, 100, 100]), seed)
        self.assertIsNone(result)
        self.assertIn("rounding", why)

    def test_schema_rejects_bool_float_string_missing_extra_and_bad_counts(self):
        variants = []
        for value in (True, 500.0, "500", None, -1, 1001):
            item = decision()
            item["positive_points"][0]["x"] = value
            variants.append(item)
        item = decision(); del item["positive_points"][0]["x"]; variants.append(item)
        item = decision(); item["positive_points"][0]["label"] = 1; variants.append(item)
        item = decision(); del item["evidence"]; variants.append(item)
        item = decision(); item["suggested_class"] = "pallet"; variants.append(item)
        item = decision(); item["positive_points"] = item["positive_points"][:1]; variants.append(item)
        item = decision(); item["negative_points"] = [{"x": i, "y": i} for i in range(5)]; variants.append(item)
        item = decision(); item["object"] = " "; variants.append(item)
        item = decision(); item["instance_scope"] = "single"; variants.append(item)
        item = decision(); item["positive_points"] = {}; variants.append(item)
        for item in variants:
            with self.subTest(item=item):
                valid, why = validate_decision(item)
                self.assertFalse(valid)
                self.assertTrue(why.startswith("schema:"))

    def test_abstention_and_coordinate_convention_cannot_silently_convert(self):
        item = decision([], [])
        item["instance_scope"] = "uncertain"
        self.assertEqual(validate_decision(item), (True, None))
        result, why = bind_points(item, binding([0, 0, 100, 100]), np.ones((100, 100), np.uint8))
        self.assertIsNone(result)
        self.assertIn("single", why)
        result, why = bind_points(decision(), {"source_box": [0, 0, 100, 100]}, np.ones((100, 100), np.uint8))
        self.assertIsNone(result)
        self.assertIn("coordinate", why)

    def test_original_source_mask_decision_and_native_views_are_unchanged(self):
        pixels = np.random.default_rng(11).integers(0, 256, (180, 250, 3), dtype=np.uint8)
        source = Image.fromarray(pixels)
        mask = np.zeros((180, 250), np.uint8); mask[77:88, 83:165] = 1
        before = mask.copy()
        original, original_details, old_binding = original_native_views(source, mask)
        views, details, bound = planning_views(source, mask)
        for key in original:
            np.testing.assert_array_equal(np.asarray(views[key]), np.asarray(original[key]))
        for old, new in zip(original_details, details):
            np.testing.assert_array_equal(np.asarray(old), np.asarray(new))
        self.assertEqual(bound["source_box"], old_binding["source_box"])
        self.assertEqual(bound["seed_pixels"], int(mask.sum()))
        np.testing.assert_array_equal(mask, before)
        np.testing.assert_array_equal(np.asarray(source), pixels)
        item = decision(); item_before = copy.deepcopy(item)
        bind_points(item, binding([0, 0, 100, 100]), mask)
        self.assertEqual(item, item_before)
        np.testing.assert_array_equal(mask, before)

    def test_seed_tolerance_is_radius_three_without_forcing_first_positive(self):
        mask = np.zeros((100, 100), np.uint8); mask[50, 50] = 1
        item = decision([{"x": 800, "y": 800}, {"x": 530, "y": 500}])
        result, why = bind_points(item, binding([0, 0, 100, 100]), mask)
        self.assertIsNone(why)
        self.assertEqual(result[1], {"x": 53, "y": 50, "label": 1})
        item["positive_points"][1]["x"] = 540
        result, why = bind_points(item, binding([0, 0, 100, 100]), mask)
        self.assertIsNone(result)
        self.assertIn("no positive", why)

    def test_negative_seed_conflict_and_duplicate_are_rejected(self):
        mask = np.zeros((100, 100), np.uint8); mask[50, 50] = 1
        result, why = bind_points(decision(negative=[{"x": 500, "y": 500}]), binding([0, 0, 100, 100]), mask)
        self.assertIsNone(result)
        self.assertIn("conflicting", why)
        item = decision([{"x": 470, "y": 500}, {"x": 700, "y": 700}], [{"x": 500, "y": 500}])
        result, why = bind_points(item, binding([0, 0, 100, 100]), mask)
        self.assertIsNone(result)
        self.assertIn("negative", why)

    def test_prompt_has_one_explicit_coordinate_contract(self):
        self.assertIn("normalized integer edge coordinates", PLAN_PROMPT)
        self.assertNotIn("gray padding", PLAN_PROMPT)
        self.assertNotIn("first positive", PLAN_PROMPT)
        self.assertNotIn("pallet", PLAN_PROMPT)


if __name__ == "__main__":
    unittest.main()
