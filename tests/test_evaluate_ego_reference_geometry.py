import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from tools import evaluate_ego_reference_geometry as evaluator
from tools.evaluate_ego_reference_geometry import (
    component_anchor,
    connected_components,
    evaluate_reference_image,
    load_exclusion_evidence,
    reference_prompt_contract,
)


class EgoReferenceGeometryTest(unittest.TestCase):
    def test_connected_components_are_sorted_and_filtered(self):
        mask = np.zeros((8, 12), dtype=bool)
        mask[1:5, 1:4] = True
        mask[2:4, 8:10] = True
        mask[7, 11] = True
        components = connected_components(mask, minimum_area=2)
        self.assertEqual([len(component) for component in components], [12, 4])

    def test_component_anchor_is_a_component_pixel(self):
        component = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 2]],
                             dtype=np.int32)
        x, y = component_anchor(component)
        self.assertIn((y, x), map(tuple, component.tolist()))

    def test_reference_contract_uses_ego_components_and_floor_negatives(self):
        ego = np.zeros((20, 30), dtype=bool)
        ego[8:19, 3:8] = True
        ego[7:19, 20:25] = True
        floor = np.logical_not(ego)
        contract = reference_prompt_contract(ego, floor)
        self.assertEqual(contract["reference_component_count"], 2)
        self.assertEqual(len(contract["positive_points"]), 2)
        self.assertEqual(len(contract["negative_points"]), 3)
        self.assertEqual(contract["major_component_indices"], [0, 1])
        self.assertEqual(len(contract["component_area_ratios"]), 2)
        for point in contract["positive_points"]:
            self.assertTrue(ego[point["y"], point["x"]])
        for point in contract["negative_points"]:
            self.assertTrue(floor[point["y"], point["x"]])
            self.assertFalse(ego[point["y"], point["x"]])

    def test_invalid_masks_fail_closed(self):
        empty = np.zeros((4, 4), dtype=bool)
        with self.assertRaisesRegex(ValueError, "reference masks"):
            reference_prompt_contract(empty, empty)

    def test_grouped_request_runs_once_and_preserves_per_group_results(self):
        output = io.BytesIO()
        Image.new("RGB", (20, 20), "gray").save(output, "PNG")
        contract = {
            "positive_points": [
                {"x": 3, "y": 15, "label": 1},
                {"x": 16, "y": 15, "label": 1},
            ],
            "negative_points": [{"x": 10, "y": 3, "label": 0}],
        }
        groups = [[positive, *contract["negative_points"]]
                  for positive in contract["positive_points"]]
        response = {"items": [
            {"index": 0, "points": groups[0], "result": {"id": 0}},
            {"index": 1, "points": groups[1], "result": {"id": 1}},
        ]}
        masks = []
        for positive in contract["positive_points"]:
            mask = np.zeros((20, 20), dtype=bool)
            mask[14:20, max(0, positive["x"] - 2):positive["x"] + 3] = True
            masks.append(mask)
        decoded = [
            ({"score": .9, "parts": [{"points": [[1, 1], [2, 1], [2, 2]],
                                       "holes": []}]}, mask, [])
            for mask in masks
        ]
        request_groups = mock.Mock(return_value=response)
        with mock.patch.object(evaluator, "_decode_response",
                               side_effect=decoded):
            result, _ = evaluate_reference_image(
                output.getvalue(), contract, request_groups=request_groups)
        request_groups.assert_called_once_with(output.getvalue(), groups)
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["result"]["request_count"], 2)
        self.assertNotIn("error", result["attempts"][0])
        self.assertNotIn("error", result["attempts"][1])

    def test_grouped_item_error_fails_only_its_own_attempt(self):
        output = io.BytesIO()
        Image.new("RGB", (20, 20), "gray").save(output, "PNG")
        contract = {
            "positive_points": [
                {"x": 3, "y": 15, "label": 1},
                {"x": 16, "y": 15, "label": 1},
            ],
            "negative_points": [{"x": 10, "y": 3, "label": 0}],
        }
        groups = [[positive, *contract["negative_points"]]
                  for positive in contract["positive_points"]]
        response = {"items": [
            {"index": 0, "points": groups[0], "error": "occluded"},
            {"index": 1, "points": groups[1], "result": {"id": 1}},
        ]}
        mask = np.zeros((20, 20), dtype=bool)
        mask[14:20, 14:19] = True
        decoded = ({"score": .9, "parts": [{
            "points": [[1, 1], [2, 1], [2, 2]], "holes": []}]}, mask, [])
        with mock.patch.object(evaluator, "_decode_response",
                               return_value=decoded):
            result, _ = evaluate_reference_image(
                output.getvalue(), contract,
                request_groups=mock.Mock(return_value=response))
        self.assertIn("occluded", result["attempts"][0]["error"])
        self.assertNotIn("error", result["attempts"][1])
        self.assertFalse(result["result"]["candidate_available"])

    def test_exclusion_evidence_is_hashed_merged_and_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first.json", root / "second.json"
            first.write_text(json.dumps({"selected_frames": [1, 7, 14]}),
                             encoding="utf-8")
            second.write_text(json.dumps({"selected_frames": [7, 20]}),
                              encoding="utf-8")
            frames, sources = load_exclusion_evidence([first, second])
            self.assertEqual(frames, {1, 7, 14, 20})
            self.assertEqual([source["frame_count"] for source in sources], [3, 2])
            self.assertTrue(all(len(source["sha256"]) == 64 for source in sources))
            second.write_text(json.dumps({"selected_frames": [150]}),
                              encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid selected_frames"):
                load_exclusion_evidence([second])

    def test_major_reference_component_rejects_a_tiny_prompt_fragment(self):
        output = io.BytesIO()
        Image.new("RGB", (20, 20), "gray").save(output, "PNG")
        contract = {
            "positive_points": [{"x": 3, "y": 15, "label": 1}],
            "negative_points": [{"x": 10, "y": 3, "label": 0}],
            "component_area_ratios": [0.10],
            "major_component_indices": [0],
        }
        tiny = np.zeros((20, 20), dtype=bool)
        tiny[15, 3] = True
        response = {"items": [{"index": 0,
                                "points": [contract["positive_points"][0],
                                           *contract["negative_points"]],
                                "result": {"id": 0}}]}
        decoded = ({"score": .95, "parts": [{
            "points": [[2, 14], [4, 14], [3, 16]], "holes": []}]},
                   tiny, [])
        with mock.patch.object(evaluator, "_decode_response",
                               return_value=decoded):
            result, _ = evaluate_reference_image(
                output.getvalue(), contract,
                request_groups=mock.Mock(return_value=response))
        attempt = result["attempts"][0]
        self.assertTrue(attempt["requested_point_covered"])
        self.assertFalse(attempt["component_area_guard_passed"])
        self.assertFalse(attempt["attempt_available"])
        self.assertFalse(result["result"]["candidate_available"])


if __name__ == "__main__":
    unittest.main()
