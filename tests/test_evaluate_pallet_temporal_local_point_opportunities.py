import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from tools.evaluate_pallet_temporal_local_point_opportunities import (
    analyze_target, evaluate, load_sequence_window,
    remap_exclusive_target_points, select_target_object,
    voxelize_mapped_points,
)


def pallet(obj_id, center=(0, 0, 5), scale=(4, 4, 4), yaw=0, ignore=False):
    return {
        "obj_id": obj_id, "obj_type": "Pallet", "ignore": ignore,
        "obj_attr": "empty",
        "obj_psr": {
            "position": dict(zip(("x", "y", "z"), center)),
            "rotation": {"roll": 0, "pitch": 0, "yaw": yaw},
            "scale": dict(zip(("x", "y", "z"), scale)),
        },
    }


def overlap(rows):
    return {
        "schema_version": "labelagent.pallet_box3d_overlap_pairs.v1",
        "status": "offline_instance_diagnostic_not_deployed",
        "human_pixel_truth": False, "approved_for_training": False,
        "production_deployed": False, "workbench_writes": 0,
        "input_sha256": {"frozen": "hash"}, "rows": rows,
    }


def unresolved(obj_ids=("target",), sequence="010", frame="105", camera=1):
    return {
        "sequence": sequence, "frame_id": frame, "camera": camera,
        "depth_split_metrics": [
            {"obj_id": obj_id, "handled_candidate": False}
            for obj_id in obj_ids],
    }


def frames_for(target="target"):
    frames = []
    for index in range(11):
        frames.append({
            "frame_id": str(100 + index),
            "objects": [pallet(target)],
            "points": np.array([
                [-1.5 + index * .04, 0, 5],
                [0 + index * .04, 0, 5],
                [1.5 + index * .04, 0, 5]], dtype=float),
        })
    return frames


class TemporalLocalPointOpportunitiesTest(unittest.TestCase):
    def test_normalized_local_mapping_handles_pose_rotation_and_scale(self):
        source = pallet("target", center=(10, 2, 5), scale=(4, 2, 2),
                        yaw=np.pi / 2)
        center = pallet("target", center=(0, 0, 5), scale=(8, 4, 2), yaw=0)
        # Normalized local coordinate [0.25, 0, 0] in the source box.
        source_box_point = np.array([[10, 3, 5]], dtype=float)
        mapped, metrics = remap_exclusive_target_points(
            source_box_point, [source], "target", center)
        np.testing.assert_allclose(mapped, [[2, 0, 5]], atol=1e-8)
        self.assertEqual(metrics["source_exclusive_point_count"], 1)

    def test_overlapping_box_point_is_excluded_before_remapping(self):
        target = pallet("target")
        peer = pallet("peer", scale=(1, 1, 1))
        points = np.array([[0, 0, 5], [1.5, 0, 5], [-1.5, 0, 5]], dtype=float)
        mapped, metrics = remap_exclusive_target_points(
            points, [target, peer], "target", target)
        self.assertEqual(len(mapped), 2)
        self.assertEqual(metrics["source_ambiguous_point_count"], 1)

    def test_center_global_voxelization_is_deterministic_and_deduplicates(self):
        first = np.array([[.001, .001, 5], [.009, .009, 5]], dtype=float)
        second = np.array([[.002, .002, 5], [.041, 0, 5]], dtype=float)
        unique, origins, keys = voxelize_mapped_points([first, second], [0, 1])
        self.assertEqual(len(unique), 2)
        self.assertEqual(origins[0], (0, 1))
        self.assertEqual(origins[1], (1,))
        reverse, reverse_origins, reverse_keys = voxelize_mapped_points(
            [second, first], [1, 0])
        np.testing.assert_allclose(unique, reverse)
        np.testing.assert_array_equal(keys, reverse_keys)
        self.assertEqual(origins, reverse_origins)

    def test_prompt_gate_requires_stable_unique_noncenter_pixels(self):
        frames = frames_for()

        def projected(points, camera, cell, slack):
            indices = np.arange(len(points), dtype=np.int64)
            # Each frame contributes three distinct pixels.
            pixels = np.column_stack((indices % 3, indices // 3))
            return indices, pixels

        with patch(
                "tools.evaluate_pallet_temporal_local_point_opportunities.project_visible",
                side_effect=projected):
            result = analyze_target(frames, "105", {}, "target")
        self.assertEqual(
            result["evidence_status"],
            "label_conditioned_temporal_local_point_prompt_opportunity")
        self.assertEqual(result["noncenter_contributing_frame_count"], 10)
        self.assertGreaterEqual(result["noncenter_shared_unique_pixel_count"], 3)

    def test_repeated_same_pixel_points_do_not_create_an_opportunity(self):
        frames = frames_for()

        def projected(points, camera, cell, slack):
            indices = np.arange(len(points), dtype=np.int64)
            return indices, np.zeros((len(indices), 2), dtype=np.int64)

        with patch(
                "tools.evaluate_pallet_temporal_local_point_opportunities.project_visible",
                side_effect=projected):
            result = analyze_target(frames, "105", {}, "target")
        self.assertEqual(result["evidence_status"], "insufficient")
        self.assertEqual(result["shared_across_grids_unique_pixel_count"], 1)

    def test_one_noncenter_contributing_frame_is_insufficient(self):
        frames = frames_for()
        center_points = np.array([[-1.5, 0, 5], [0, 0, 5], [1.5, 0, 5]],
                                 dtype=float)
        for row in frames:
            row["points"] = center_points.copy()
        frames[0]["points"] = np.array(
            [[-1.25, 0, 5], [.25, 0, 5], [1.75, 0, 5]], dtype=float)

        def projected(points, camera, cell, slack):
            indices = np.arange(len(points), dtype=np.int64)
            pixels = np.column_stack((indices, np.zeros(len(indices), dtype=int)))
            return indices, pixels

        with patch(
                "tools.evaluate_pallet_temporal_local_point_opportunities.project_visible",
                side_effect=projected):
            result = analyze_target(frames, "105", {}, "target")
        self.assertEqual(result["noncenter_contributing_frame_count"], 1)
        self.assertEqual(result["evidence_status"], "insufficient")

    def test_inconsistent_attribute_or_scale_cannot_open_gate(self):
        frames = frames_for()
        frames[0]["objects"][0]["obj_attr"] = "loaded"

        def projected(points, camera, cell, slack):
            indices = np.arange(len(points), dtype=np.int64)
            return indices, np.column_stack((indices % 3, indices // 3))

        with patch(
                "tools.evaluate_pallet_temporal_local_point_opportunities.project_visible",
                side_effect=projected):
            result = analyze_target(frames, "105", {}, "target")
        self.assertFalse(result["target_attr_consistent"])
        self.assertEqual(result["evidence_status"], "insufficient")

        frames = frames_for()
        frames[0]["objects"][0]["obj_psr"]["scale"]["x"] = 4.1
        with patch(
                "tools.evaluate_pallet_temporal_local_point_opportunities.project_visible",
                side_effect=projected):
            result = analyze_target(frames, "105", {}, "target")
        self.assertGreater(result["target_scale_max_range"], 0)
        self.assertEqual(result["evidence_status"], "insufficient")

    def test_missing_duplicate_ignored_and_nonpositive_scale_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "missing or duplicated"):
            select_target_object([], "target")
        with self.assertRaisesRegex(ValueError, "missing or duplicated"):
            select_target_object([pallet("target"), pallet("target")], "target")
        with self.assertRaisesRegex(ValueError, "ignored"):
            select_target_object([pallet("target", ignore=True)], "target")
        with self.assertRaisesRegex(ValueError, "Invalid cuboid geometry"):
            select_target_object([pallet("target", scale=(1, 0, 1))], "target")

    def test_window_identity_and_runtime_drift_fail_closed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); labels = root / "010" / "labels"
            pcd = root / "010" / "pcd"; labels.mkdir(parents=True); pcd.mkdir()
            for index in range(11):
                frame = str(100 + index)
                label_frame = "wrong" if index == 0 else frame
                (labels / f"{frame}.json").write_text(json.dumps({
                    "frame_id": label_frame, "objects": [pallet("target")]}),
                    encoding="utf-8")
                (pcd / f"{frame}.pcd").write_bytes(b"pcd")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                load_sequence_window(root, "010")
            (labels / "100.json").write_text(json.dumps({
                "frame_id": "100", "objects": [pallet("target")]}),
                encoding="utf-8")

            def mutate(path):
                Path(path).write_bytes(b"changed")
                return np.zeros((1, 3), dtype=float)

            with patch(
                    "tools.evaluate_pallet_temporal_local_point_opportunities.read_xyz_pcd",
                    side_effect=mutate), \
                 self.assertRaisesRegex(RuntimeError, "PCD changed"):
                load_sequence_window(root, "010")

    def test_evaluate_reads_each_window_pcd_once_and_binds_actual_inputs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); window = root / "windows"
            labels = window / "010" / "labels"; pcd = window / "010" / "pcd"
            labels.mkdir(parents=True); pcd.mkdir()
            for index in range(11):
                frame = str(100 + index)
                (labels / f"{frame}.json").write_text(json.dumps({
                    "frame_id": frame,
                    "objects": [pallet("a"), pallet("b", center=(8, 0, 5))],
                }), encoding="utf-8")
                (pcd / f"{frame}.pcd").write_bytes(f"pcd-{frame}".encode())
            overlap_path = root / "overlap.json"
            overlap_path.write_text(json.dumps(overlap([
                unresolved(("a", "b"))])), encoding="utf-8")
            calibration = root / "calib"; calibration.mkdir()
            (calibration / "lidar2camera.yaml").write_text("ext", encoding="utf-8")
            (calibration / "camera1_calib.yaml").write_text("int", encoding="utf-8")
            output = root / "output"
            points = np.array([[-1, 0, 5], [0, 0, 5], [1, 0, 5]], dtype=float)

            def projected(values, camera, cell, slack):
                indices = np.arange(len(values), dtype=np.int64)
                pixels = np.column_stack((indices % 5, indices // 5))
                return indices, pixels

            with patch(
                    "tools.evaluate_pallet_temporal_local_point_opportunities.read_xyz_pcd",
                    return_value=points) as reader, \
                 patch(
                    "tools.evaluate_pallet_temporal_local_point_opportunities.load_camera",
                    return_value={}), \
                 patch(
                    "tools.evaluate_pallet_temporal_local_point_opportunities.project_visible",
                    side_effect=projected):
                report = evaluate(overlap_path, window, calibration, output)
            self.assertEqual(reader.call_count, 11)
            self.assertEqual(report["total_pcd_reads"], 11)
            self.assertTrue(all(count == 1 for count in report["pcd_read_counts"]["010"].values()))
            self.assertEqual(len(report["input_sha256"]), 25)
            self.assertTrue(report["readback_verified"])
            self.assertEqual(report["images_read"], 0)
            self.assertEqual(report["sam_calls"], 0)
            self.assertEqual(report["human_truth_writes"], 0)
            raw = (output / "results.json").read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), report["report_sha256"])


if __name__ == "__main__":
    unittest.main()
