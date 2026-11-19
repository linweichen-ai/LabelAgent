import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from source.agent.edit_agent import _shape_from_mask
from tools.audit_pallet_geometry import audit_candidate, run_audit, select_stratified


class PalletGeometryAuditTest(unittest.TestCase):
    def test_stratified_selection_is_deterministic_and_bounded(self):
        rows = [{"topic": "camera", "frame": index,
                 "shape": {"id": index, "points": [[0, 0], [index + 1, 0],
                                                       [index + 1, 2], [0, 2]],
                           "parts": []}}
                for index in range(10)]
        first = select_stratified(rows, 4)
        second = select_stratified(list(reversed(rows)), 4)
        self.assertEqual([row["shape"]["id"] for row in first],
                         [row["shape"]["id"] for row in second])
        self.assertEqual(len(first), 4)

    def test_audit_preserves_long_slot_and_cleans_micro_hole(self):
        mask = np.zeros((100, 140), dtype=np.uint8)
        mask[10:90, 10:130] = 1
        mask[25:30, 25:30] = 0
        mask[55:62, 40:95] = 0
        shape = _shape_from_mask(mask, "pallet", 1)
        result = audit_candidate(
            {"topic": "camera", "frame": "1", "shape": shape}, 140, 100)
        self.assertTrue(result["adjustment_applied"])
        self.assertEqual(result["components_before"], result["components_after"])
        self.assertGreaterEqual(result["minimum_structural_hole_retention"], 0.75)
        self.assertLess(result["holes_after"], result["holes_before"])
        self.assertLessEqual(result["change_ratio"], 0.025)

    def test_audit_reports_bounded_boundary_vertex_reduction(self):
        mask = np.zeros((300, 420), dtype=np.uint8)
        for y in range(30, 270):
            mask[y, 30 + y % 6:390 - (y + 2) % 6] = 1
        shape = _shape_from_mask(mask, "pallet", 2)

        result = audit_candidate(
            {"topic": "camera", "frame": "2", "shape": shape}, 420, 300)

        self.assertTrue(result["boundary_simplification_applied"])
        self.assertLess(result["boundary_vertices_after"],
                        result["boundary_vertices_before"])
        self.assertLessEqual(result["change_ratio"], .025)

    def test_workbench_state_uses_nested_candidates_by_topic(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[5:35, 5:35] = 1
        shape = _shape_from_mask(mask, "pallet", 1)
        state = {"candidates_by_topic": {"camera": {"0": [shape]}}}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "state.json")
            source.write_text(json.dumps(state), encoding="utf-8")
            result = run_audit(source, width=40, height=40, per_topic=4)
        self.assertEqual(result["input_scope"],
                         "workbench_state.candidates_by_topic")
        self.assertEqual(result["sample_count"], 1)
        self.assertIn("boundary_simplified_count", result)
        self.assertIn("median_boundary_vertex_reduction_ratio", result)


if __name__ == "__main__":
    unittest.main()
