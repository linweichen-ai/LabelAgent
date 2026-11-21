import unittest

import numpy as np

from tools.evaluate_ego_reference_geometry import (
    component_anchor,
    connected_components,
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
        for point in contract["positive_points"]:
            self.assertTrue(ego[point["y"], point["x"]])
        for point in contract["negative_points"]:
            self.assertTrue(floor[point["y"], point["x"]])
            self.assertFalse(ego[point["y"], point["x"]])

    def test_invalid_masks_fail_closed(self):
        empty = np.zeros((4, 4), dtype=bool)
        with self.assertRaisesRegex(ValueError, "reference masks"):
            reference_prompt_contract(empty, empty)


if __name__ == "__main__":
    unittest.main()
