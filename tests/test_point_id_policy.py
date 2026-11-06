"""Source binding and rejection tests for the offline point-ID interface."""
import copy
import math
import unittest

import numpy as np
from PIL import Image

from native_resume_policy import planning_views
from point_id_policy import PLAN_PROMPT, bind_ids, make_views, schema_for


class PointIDTests(unittest.TestCase):
    def setUp(self):
        self.pixels = np.random.default_rng(12).integers(0, 256, (200, 320, 3), dtype=np.uint8)
        self.source = Image.fromarray(self.pixels)
        self.mask = np.zeros((200, 320), np.uint8)
        self.mask[84:110, 80:240] = 1
        self.views, self.clean, self.binding = make_views(self.source, self.mask)
        self.catalog = self.binding["point_catalog"]
        self.inside = [p for p in self.catalog if self.mask[p["y"], p["x"]]]
        self.outside = [p for p in self.catalog if not self.mask[p["y"], p["x"]]]

    def decision(self, positive=None, negative=None):
        return {"object": "one visible object", "instance_scope": "single_visible_object",
                "positive_ids": positive if positive is not None else [self.inside[0]["id"], self.outside[0]["id"]],
                "negative_ids": negative if negative is not None else [],
                "evidence": "Visible structural parts of the same marked object."}

    def test_selected_ids_map_to_exact_source_pixels_without_normalized_round_trip(self):
        positive = [self.outside[0], self.inside[0]]
        negative = [self.outside[-1]]
        result, why = bind_ids(self.decision([p["id"] for p in positive], [p["id"] for p in negative]), self.binding, self.mask)
        self.assertIsNone(why)
        self.assertEqual(result, [{"x": p["x"], "y": p["y"], "label": label}
                                  for points, label in ((positive, 1), (negative, 0)) for p in points])

    def test_unknown_duplicate_and_positive_negative_conflicts_are_not_repaired(self):
        variants = [self.decision([self.inside[0]["id"], "P99"]),
                    self.decision([self.inside[0]["id"], self.inside[0]["id"]]),
                    self.decision(negative=[self.inside[0]["id"]])]
        for decision, reason in zip(variants, ("unknown", "duplicate", "conflict")):
            before = copy.deepcopy(decision)
            result, why = bind_ids(decision, self.binding, self.mask)
            self.assertIsNone(result)
            self.assertIn(reason, why)
            self.assertEqual(decision, before)

    def test_abstention_keeps_empty_lists_and_never_injects_a_seed_anchor(self):
        decision = self.decision([], [])
        decision["instance_scope"] = "edge_only"
        result, why = bind_ids(decision, self.binding, self.mask)
        self.assertIsNone(result)
        self.assertIn("single", why)
        self.assertEqual(decision["positive_ids"], [])
        decision["positive_ids"] = [self.inside[0]["id"]]
        result, why = bind_ids(decision, self.binding, self.mask)
        self.assertIsNone(result)
        self.assertIn("abstention", why)

    def test_local_schema_rejects_malformed_ids_fields_and_counts(self):
        variants = []
        for value in (True, 1, None, {"id": "P01"}):
            item = self.decision(); item["positive_ids"][0] = value; variants.append(item)
        item = self.decision(); del item["object"]; variants.append(item)
        item = self.decision(); item["x"] = 500; variants.append(item)
        item = self.decision(); item["positive_ids"] = item["positive_ids"][:1]; variants.append(item)
        item = self.decision(); item["negative_ids"] = [p["id"] for p in self.outside[:5]]; variants.append(item)
        for item in variants:
            result, why = bind_ids(item, self.binding, self.mask)
            self.assertIsNone(result)
            self.assertTrue(why.startswith("schema:"))

    def test_input_photo_and_entire_seed_preserved_with_same_native_square(self):
        before = self.mask.copy()
        _, _, native = planning_views(self.source, self.mask)
        self.assertEqual(self.binding["source_box"], native["source_box"])
        box = self.binding["source_box"]
        expected = self.source.crop(tuple(box)).resize((512, 512), Image.Resampling.LANCZOS)
        np.testing.assert_array_equal(np.asarray(self.clean), np.asarray(expected))
        np.testing.assert_array_equal(np.asarray(self.source), self.pixels)
        np.testing.assert_array_equal(self.mask, before)
        self.assertEqual(int(self.mask[box[1]:box[3], box[0]:box[2]].sum()), int(self.mask.sum()))
        self.assertEqual(set(self.views), {"pointer", "corners"})
        self.assertEqual(self.binding["request_image_count"], 2)
        self.assertFalse(self.binding["detail_views_used"])

    def test_menu_reproducible_bounded_and_not_visually_dense(self):
        views, clean, bound = make_views(self.source, self.mask)
        self.assertEqual(bound, self.binding)
        self.assertLessEqual(len(self.catalog), 23)
        self.assertGreater(len(self.catalog), 10)
        self.assertEqual(len({p["id"] for p in self.catalog}), len(self.catalog))
        for key in views:
            np.testing.assert_array_equal(np.asarray(views[key]), np.asarray(self.views[key]))
        for i, point in enumerate(self.catalog):
            for other in self.catalog[i + 1:]:
                self.assertGreaterEqual(math.dist(point["display_xy"], other["display_xy"]), 34.5)

    def test_no_positive_seed_binding_and_negative_seed_intrusion_rejected(self):
        far = [p for p in self.outside if abs(p["y"] - 97) > 30]
        result, why = bind_ids(self.decision([p["id"] for p in far[:2]]), self.binding, self.mask)
        self.assertIsNone(result)
        self.assertIn("no positive", why)
        result, why = bind_ids(self.decision([self.inside[0]["id"], far[0]["id"]], [self.inside[1]["id"]]), self.binding, self.mask)
        self.assertIsNone(result)
        self.assertIn("negative", why)

    def test_distinct_ids_cannot_hide_sub_three_pixel_coordinate_conflicts(self):
        bound = copy.deepcopy(self.binding)
        bound["point_catalog"][1].update(x=bound["point_catalog"][0]["x"] + 1, y=bound["point_catalog"][0]["y"])
        result, why = bind_ids(self.decision([p["id"] for p in bound["point_catalog"][:2]]), bound, self.mask)
        self.assertIsNone(result)
        self.assertIn("conflicting", why)

    def test_catalog_boundary_and_wrong_coordinate_convention_are_rejected(self):
        bound = copy.deepcopy(self.binding)
        bound["point_catalog"][0]["x"] = bound["source_box"][2]
        result, why = bind_ids(self.decision(), bound, self.mask)
        self.assertIsNone(result)
        self.assertIn("outside", why)
        bound["coordinate_system"] = "normalized_edge_0_1000"
        result, why = bind_ids(self.decision(), bound, self.mask)
        self.assertIsNone(result)
        self.assertIn("convention", why)

    def test_schema_limits_choices_without_encoding_a_category_answer(self):
        schema = schema_for(self.binding)
        self.assertEqual(schema["properties"]["positive_ids"]["items"]["enum"], [p["id"] for p in self.catalog])
        self.assertNotIn("enum", schema["properties"]["object"])
        self.assertNotIn("pallet", PLAN_PROMPT.lower())
        self.assertNotIn("person", PLAN_PROMPT.lower())
        self.assertFalse(self.binding["expected_class_rendered"])
        self.assertTrue(self.binding["not_single_factor_ablation"])
        self.assertIn("original_instance_purity_support", self.binding["required_downstream_gates"])


if __name__ == "__main__":
    unittest.main()
