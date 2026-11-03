import importlib.util
from pathlib import Path
import unittest
import numpy as np

SPEC = importlib.util.spec_from_file_location("boxes3d", Path(__file__).resolve().parents[1] / "tools/compare_boxes3d.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def box(center=(0, 0, 0), size=(1, 1, 1), angles=(0, 0, 0)):
    return MODULE.Box({"obj_type": "Pallet", "obj_psr": {
        "position": dict(zip(("x", "y", "z"), center)),
        "scale": dict(zip(("x", "y", "z"), size)),
        "rotation": dict(zip(("roll", "pitch", "yaw"), angles))}})


class Boxes3DTests(unittest.TestCase):
    def test_axis_aligned_half_overlap_has_analytic_iou(self):
        self.assertAlmostEqual(MODULE.iou3d(box(), box(center=(0.5, 0, 0))), 1/3, places=7)

    def test_rotated_identical_cuboids_and_symmetry(self):
        a = box(center=(8, -4, -2), size=(1.2, 0.7, 0.15), angles=(0.07, -0.02, 1.3))
        self.assertAlmostEqual(MODULE.iou3d(a, a), 1, places=6)
        b = box(center=(8.1, -4, -2), size=(1.2, 0.7, 0.15), angles=(0.06, -0.01, 1.4))
        self.assertAlmostEqual(MODULE.iou3d(a, b), MODULE.iou3d(b, a), places=7)

    def test_fully_contained_rotated_box(self):
        self.assertAlmostEqual(MODULE.iou3d(box(size=(4, 4, 4)), box(angles=(0.2, 0.4, 0.7))), 1/64, places=7)

    def test_disjoint_and_touching_boxes(self):
        self.assertEqual(MODULE.iou3d(box(), box(center=(2, 0, 0))), 0)
        self.assertEqual(MODULE.iou3d(box(), box(center=(1, 0, 0))), 0)

    def test_matching_maximizes_valid_matches_before_overlap_sum(self):
        matrix = np.array([[0.99, 0.51], [0.51, 0]])
        matches = MODULE.match(matrix, 0.5)
        self.assertEqual({item[:2] for item in matches}, {(0, 1), (1, 0)})
        self.assertEqual(MODULE.match(np.zeros((0, 2)), 0.5), [])


if __name__ == "__main__":
    unittest.main()
