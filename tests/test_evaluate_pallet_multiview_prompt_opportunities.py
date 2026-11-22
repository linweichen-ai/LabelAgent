import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from tools.evaluate_pallet_multiview_prompt_opportunities import (
    analyze_target, evaluate, hash_indices, load_center_frame, plan_targets,
)


def pallet(obj_id, center=(0, 0, 5), scale=(4, 4, 4)):
    return {
        "obj_id": obj_id, "obj_type": "Pallet", "ignore": False,
        "obj_psr": {
            "position": dict(zip(("x", "y", "z"), center)),
            "rotation": {"roll": 0, "pitch": 0, "yaw": 0},
            "scale": dict(zip(("x", "y", "z"), scale)),
        },
    }


def overlap(rows):
    return {
        "schema_version": "labelagent.pallet_box3d_overlap_pairs.v1",
        "status": "offline_instance_diagnostic_not_deployed",
        "human_pixel_truth": False,
        "approved_for_training": False,
        "production_deployed": False,
        "workbench_writes": 0,
        "input_sha256": {"frozen": "hash"},
        "rows": rows,
    }


def unresolved(sequence="010", frame="100", camera=0, obj_id="target"):
    return {
        "sequence": sequence, "frame_id": frame, "camera": camera,
        "depth_split_metrics": [
            {"obj_id": obj_id, "handled_candidate": False}],
    }


def fake_cameras():
    return {camera: {"camera": camera} for camera in range(4)}


class PalletMultiviewPromptOpportunitiesTest(unittest.TestCase):
    def test_shared_xyz_indices_survive_different_camera_local_orders(self):
        points = np.array([[index, 0, 5] for index in range(8)], dtype=float)
        objects = [pallet("target", center=(3.5, 0, 5), scale=(9, 2, 2))]

        def projected(_points, camera, cell, slack):
            values = {
                0: [0, 1], 1: [6, 2, 4, 0],
                2: [4, 6], 3: [1],
            }[camera["camera"]]
            if cell == 8 and camera["camera"] == 1:
                values = [4, 2, 6]
            return np.array(values), np.zeros((len(values), 2), dtype=int)

        with patch("tools.evaluate_pallet_multiview_prompt_opportunities.project_visible",
                   side_effect=projected):
            result = analyze_target(points, objects, fake_cameras(), 0, "target")
        self.assertEqual(result["best_camera"], 1)
        camera = result["cameras"][1]
        self.assertEqual(camera["shared_across_grids_raw_index_count"], 3)
        self.assertEqual(camera["shared_across_grids_raw_index_sha256"],
                         hash_indices(np.array([2, 4, 6])))

    def test_overlapping_3d_box_points_are_excluded(self):
        points = np.array([[0, 0, 5], [1.5, 0, 5], [-1.5, 0, 5],
                           [0, 1.5, 5], [0, -1.5, 5]], dtype=float)
        objects = [pallet("target"), pallet("peer", center=(0, 0, 5),
                                            scale=(1, 1, 1))]

        def projected(_points, camera, cell, slack):
            values = np.arange(len(_points))
            return values, np.zeros((len(values), 2), dtype=int)

        with patch("tools.evaluate_pallet_multiview_prompt_opportunities.project_visible",
                   side_effect=projected):
            result = analyze_target(points, objects, fake_cameras(), 0, "target")
        self.assertEqual(result["ambiguous_3d_box_raw_index_count"], 1)
        self.assertEqual(result["exclusive_target_raw_index_count"], 4)
        self.assertEqual(result["cameras"][1]["shared_across_grids_raw_index_count"], 4)

    def test_primary_only_or_no_other_view_is_insufficient(self):
        points = np.array([[index, 0, 5] for index in range(5)], dtype=float)
        objects = [pallet("target", center=(2, 0, 5), scale=(6, 2, 2))]

        def projected(_points, camera, cell, slack):
            values = np.array([0, 1, 2]) if camera["camera"] == 0 else np.array([0, 1])
            return values, np.zeros((len(values), 2), dtype=int)

        with patch("tools.evaluate_pallet_multiview_prompt_opportunities.project_visible",
                   side_effect=projected):
            result = analyze_target(points, objects, fake_cameras(), 0, "target")
        self.assertEqual(result["evidence_status"], "insufficient")
        self.assertIsNone(result["best_camera"])

    def test_best_camera_uses_maximum_minimum_then_stable_camera_tie_break(self):
        points = np.array([[index, 0, 5] for index in range(8)], dtype=float)
        objects = [pallet("target", center=(3.5, 0, 5), scale=(9, 2, 2))]

        def projected(_points, camera, cell, slack):
            count = {0: 1, 1: 5, 2: 5, 3: 4}[camera["camera"]]
            if cell == 8 and camera["camera"] in (1, 2):
                count = 4
            values = np.arange(count)
            return values, np.zeros((count, 2), dtype=int)

        with patch("tools.evaluate_pallet_multiview_prompt_opportunities.project_visible",
                   side_effect=projected):
            result = analyze_target(points, objects, fake_cameras(), 0, "target")
        self.assertEqual(result["best_camera"], 1)
        self.assertEqual(result["best_camera_minimum_grid_point_count"], 4)

    def test_input_identity_and_only_unhandled_selection_fail_closed(self):
        payload = overlap([unresolved(), {
            **unresolved(obj_id="done"),
            "depth_split_metrics": [{"obj_id": "done", "handled_candidate": True}],
        }])
        self.assertEqual([job["target_obj_id"] for job in plan_targets(payload)],
                         ["target"])
        unbound = dict(payload); unbound.pop("input_sha256")
        with self.assertRaisesRegex(ValueError, "frozen non-deployed"):
            plan_targets(unbound)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = root / "010" / "labels"; pcd = root / "010" / "pcd"
            labels.mkdir(parents=True); pcd.mkdir()
            (labels / "100.json").write_text(
                json.dumps({"frame_id": "101", "objects": []}), encoding="utf-8")
            (pcd / "100.pcd").write_bytes(b"not-read")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                load_center_frame(root, "010", "100")

    def test_center_frame_fails_closed_when_input_changes_during_parse(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = root / "010" / "labels"; pcd = root / "010" / "pcd"
            labels.mkdir(parents=True); pcd.mkdir()
            label_path = labels / "100.json"
            label_path.write_text(
                json.dumps({"frame_id": "100", "objects": []}),
                encoding="utf-8")
            (pcd / "100.pcd").write_bytes(b"frozen-pcd")

            def mutate_after_read(_path):
                label_path.write_text(
                    json.dumps({"frame_id": "100", "objects": [pallet("late")]}),
                    encoding="utf-8")
                return np.zeros((1, 3), dtype=float)

            with patch(
                    "tools.evaluate_pallet_multiview_prompt_opportunities.read_xyz_pcd",
                    side_effect=mutate_after_read), \
                 self.assertRaisesRegex(RuntimeError, "changed while being parsed"):
                load_center_frame(root, "010", "100")

    def test_evaluate_reads_shared_center_pcd_once_and_binds_all_inputs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            overlap_path = root / "overlap.json"
            overlap_path.write_text(json.dumps(overlap([
                unresolved(obj_id="a"), unresolved(obj_id="b")
            ])), encoding="utf-8")
            calibration = root / "calib"; calibration.mkdir()
            for name in ["lidar2camera.yaml"] + [f"camera{i}_calib.yaml" for i in range(4)]:
                (calibration / name).write_text(name, encoding="utf-8")
            output = root / "output"
            points = np.array([[0, 0, 5]], dtype=float)
            objects = [pallet("a"), pallet("b", center=(10, 0, 5))]
            hashes = {str((root / "label.json").resolve()): "0" * 64,
                      str((root / "point.pcd").resolve()): "1" * 64}
            # Use real files so the final frozen-input readback remains meaningful.
            for path in map(Path, hashes):
                path.write_text(path.name, encoding="utf-8")
                hashes[str(path)] = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
            result = {"evidence_status": "insufficient", "best_camera": None,
                      "best_camera_minimum_grid_point_count": None,
                      "exclusive_target_raw_index_count": 0,
                      "ambiguous_3d_box_raw_index_count": 0, "cameras": []}
            with patch("tools.evaluate_pallet_multiview_prompt_opportunities.load_camera",
                       side_effect=lambda path, camera: {"camera": camera}), \
                 patch("tools.evaluate_pallet_multiview_prompt_opportunities.load_center_frame",
                       return_value=(points, objects, hashes)) as loader, \
                 patch("tools.evaluate_pallet_multiview_prompt_opportunities.analyze_target",
                       return_value=result):
                report = evaluate(overlap_path, root, calibration, output)
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(report["center_frame_pcd_read_count"], 1)
            self.assertEqual(len(report["input_sha256"]), 8)
            self.assertTrue(report["readback_verified"])
            self.assertFalse(report["approved_for_training"])
            self.assertEqual(report["human_truth_writes"], 0)


if __name__ == "__main__":
    unittest.main()
