import unittest

import numpy as np

from tools.evaluate_pallet_box3d_roi_sam import (
    localization_gate, project_box, select_target,
)


def pallet(identifier=1, attr="empty", center=(0, 0, 5), scale=(2, 1, .2)):
    return {"obj_id": identifier, "obj_type": "Pallet", "obj_attr": attr,
            "ignore": False, "obj_psr": {
                "position": dict(zip("xyz", center)), "scale": dict(zip("xyz", scale)),
                "rotation": {"roll": 0, "pitch": 0, "yaw": 0}}}


def camera():
    return {"width": 200, "height": 120,
            "intrinsics": np.array([100., 100., 100., 60.]),
            "transform": np.eye(4)}


class PalletBox3dRoiSamTest(unittest.TestCase):
    def test_projects_visible_box(self):
        row = project_box(pallet(), camera())
        self.assertIsNotNone(row)
        self.assertAlmostEqual(row["center"][0], 100)
        self.assertAlmostEqual(row["center"][1], 60)
        self.assertGreater(row["projected_pixels"], 100)

    def test_rejects_box_behind_camera(self):
        self.assertIsNone(project_box(pallet(center=(0, 0, -5)), camera()))

    def test_default_prefers_empty_pallet(self):
        loaded = pallet(1, "loaded", scale=(3, 2, .3))
        empty = pallet(2, "empty", center=(.5, 0, 5), scale=(2, 1, .2))
        selected, projected = select_target([loaded, empty], camera())
        self.assertEqual(len(projected), 2)
        self.assertEqual(selected["obj_id"], 2)

    def test_prefers_isolated_over_larger_overlapping_pallet(self):
        overlapping_large = pallet(1, "loaded", center=(0, 0, 5), scale=(3, 2, .3))
        duplicate = pallet(2, "loaded", center=(0, 0, 5), scale=(3, 2, .3))
        isolated = pallet(3, "empty", center=(3, 0, 7), scale=(1.5, 1, .2))
        selected, _ = select_target(
            [overlapping_large, duplicate, isolated], camera(), policy="isolated")
        self.assertEqual(selected["obj_id"], 3)

    def test_rejects_unknown_target_policy(self):
        with self.assertRaises(ValueError):
            select_target([pallet()], camera(), policy="unknown")

    def test_localization_gate_keeps_geometry_thresholds_explicit(self):
        metrics = {"center_distance_pixels": 10, "projected_box_coverage": .2,
                   "proposal_inside_projected_box": .8}
        self.assertTrue(localization_gate(metrics))
        self.assertFalse(localization_gate({**metrics, "center_distance_pixels": 25}))


if __name__ == "__main__":
    unittest.main()
