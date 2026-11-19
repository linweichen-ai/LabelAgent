import json
import tempfile
import unittest
from pathlib import Path

import cv2
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

    def test_audit_uses_v5_compact_hole_contract_without_reclassifying_long_slot(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[40:440, 80:560] = 1
        # Compact surface drop-out: wider than the old 32-pixel structural rule.
        cv2.ellipse(mask, (180, 170), (24, 8), 0, 0, 360, 0, thickness=cv2.FILLED)
        # Long board slot: still structural under the v3 aspect-ratio gate.
        mask[300:310, 240:310] = 0
        shape = _shape_from_mask(mask, "pallet", 3)

        result = audit_candidate(
            {"topic": "camera", "frame": "3", "shape": shape}, 640, 480)

        self.assertEqual(result["geometry_cleanup_version"], 5)
        self.assertGreaterEqual(result["filled_compact_texture_hole_count"], 1)
        self.assertEqual(result["structural_holes_before"], 1)
        self.assertGreaterEqual(result["minimum_structural_hole_retention"], .75)
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

    def test_workbench_state_overlays_active_topic_candidates_on_stale_topic_snapshot(self):
        stale_mask = np.zeros((60, 60), dtype=np.uint8)
        stale_mask[5:55, 5:55] = 1
        stale_mask[20:25, 20:25] = 0
        current_mask = np.zeros((60, 60), dtype=np.uint8)
        current_mask[10:50, 10:50] = 1
        stale = _shape_from_mask(stale_mask, "pallet", 1)
        current = _shape_from_mask(current_mask, "pallet", 2)
        state = {
            "topic": "camera",
            "candidates": {"7": [current]},
            "candidates_by_topic": {"camera": {"3": [stale]}},
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "state.json")
            source.write_text(json.dumps(state), encoding="utf-8")
            result = run_audit(source, width=60, height=60, per_topic=4)

        self.assertEqual(
            result["input_scope"],
            "workbench_state.current_candidates_overlaid_on_candidates_by_topic")
        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(result["results"][0]["frame"], "7")
        self.assertEqual(result["results"][0]["shape_id"], 2)

    def test_image_guided_audit_loads_matching_topic_frame_without_writing_source(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[40:440, 80:560] = 1
        mask[180:210, 200:240] = 0
        shape = _shape_from_mask(mask, "pallet", 4)
        payload = {"/capture/camera_0/image": {"0": [shape]}}
        image = np.full((480, 640, 3), 180, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "candidates.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            camera = root / "images" / "camera_0"
            camera.mkdir(parents=True)
            image_path = camera / "000.png"
            cv2.imwrite(str(image_path), image)
            before = source.read_bytes()

            result = run_audit(
                source, width=640, height=480, per_topic=4,
                image_root=root / "images")

            self.assertEqual(source.read_bytes(), before)
            self.assertTrue(result["image_guidance"])
            self.assertEqual(result["sample_count"], 1)
            self.assertTrue(result["results"][0]["image_guidance_used"])
            self.assertEqual(
                Path(result["results"][0]["image_path"]), image_path.resolve())

    def test_image_guided_audit_rejects_missing_camera_mapping(self):
        payload = {"camera": {"0": []}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "candidates.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "map topic"):
                run_audit(
                    source, width=40, height=40, per_topic=4,
                    image_root=root / "images")


if __name__ == "__main__":
    unittest.main()
