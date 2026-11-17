import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools import slope_lidar_geometry as geometry
from tools.evaluate_slope_lidar_geometry import distance_to_hints
from tools.evaluate_slope_lidar_sam import (
    geometry_attempt_rank,
    geometry_mask_metrics,
    polygon_mask,
    prompt_points,
    ring_constrained_points,
)
from tools.freeze_slope_geometry_external import evenly_spaced, freeze
from tools.evaluate_slope_lidar_external import verify_manifest
from tools.evaluate_projected_support import camera_parameters as existing_camera_parameters
from tools.evaluate_projected_support import project_visible as existing_project_visible


class SlopeLidarGeometryTests(unittest.TestCase):
    def test_binary_pcd_reader_keeps_finite_nonzero_xyz(self):
        header = (
            b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\n"
            b"SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
            b"WIDTH 4\nHEIGHT 1\nPOINTS 4\nDATA binary\n"
        )
        rows = [(1, 2, 3, 4), (0, 0, 0, 1), (float("nan"), 2, 3, 1), (-1, -2, -3, 2)]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.pcd"
            path.write_bytes(header + b"".join(struct.pack("<ffff", *row) for row in rows))
            points = geometry.read_xyz_pcd(path)
        np.testing.assert_allclose(points, [[1, 2, 3], [-1, -2, -3]])

    def test_projection_matches_existing_validated_implementation(self):
        intrinsic = {
            "model_type": "PINHOLE",
            "image_width": 100,
            "image_height": 80,
            "projection_parameters": {"fx": 100, "fy": 80, "cx": 50, "cy": 40},
            "distortion_parameters": {"k1": 0, "k2": 0, "p1": 0, "p2": 0},
        }
        extrinsic = {"calibrated": True, "transform:": np.eye(4).ravel().tolist()}
        first = geometry.camera_parameters(intrinsic, extrinsic)
        second = existing_camera_parameters(intrinsic, extrinsic)
        np.testing.assert_allclose(first["intrinsics"], second["intrinsics"])
        np.testing.assert_allclose(first["transform"], second["transform"])
        points = np.array([[0, 0, 1], [0.1, 0, 1], [0, 0, 2], [-0.1, 0.1, 1]], dtype=float)
        a_source, a_pixels = geometry.project_visible(points, first, cell_size=4, depth_slack=0.1)
        b_source, b_pixels = existing_project_visible(points, second, cell_size=4, depth_slack=0.1)
        np.testing.assert_array_equal(a_source, b_source)
        np.testing.assert_array_equal(a_pixels, b_pixels)

    def test_camera_parameters_accepts_both_observed_transform_spellings(self):
        intrinsic = {
            "model_type": "PINHOLE",
            "image_width": 640,
            "image_height": 512,
            "projection_parameters": {"fx": 400, "fy": 400, "cx": 320, "cy": 256},
            "distortion_parameters": {"k1": 0, "k2": 0, "p1": 0, "p2": 0},
        }
        legacy = geometry.camera_parameters(
            intrinsic, {"calibrated": True, "transform:": np.eye(4).ravel().tolist()}
        )
        external = geometry.camera_parameters(
            intrinsic, {"calibrated": True, "transform": np.eye(4).ravel().tolist()}
        )
        np.testing.assert_array_equal(legacy["transform"], external["transform"])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            geometry.camera_parameters(
                intrinsic,
                {
                    "calibrated": True,
                    "transform:": np.eye(4).ravel().tolist(),
                    "transform": np.eye(4).ravel().tolist(),
                },
            )

    def test_tilted_samples_find_synthetic_ramp_not_flat_floor(self):
        xs = np.linspace(0.6, 2.5, 40)
        ys = np.linspace(-0.8, 0.8, 35)
        floor = np.array([(x, y, -1.0) for x in xs for y in ys])
        ramp = np.array([(x, y, -1.0 + 0.25 * (x - 1.0)) for x in np.linspace(1.0, 2.0, 30) for y in np.linspace(-0.35, 0.35, 24)])
        points = np.vstack([floor, ramp])
        support = {"usable": True, "plane": [0.0, 0.0, 1.0, 1.0]}
        samples = geometry.tilted_planar_samples(points, support)
        components = geometry.cluster_tilted_samples(samples, support)
        self.assertGreater(len(samples["selected_points"]), 20)
        self.assertTrue(components)
        self.assertGreater(components[0]["median_tilt_degrees"], 8)
        self.assertLess(components[0]["median_tilt_degrees"], 25)

    def test_flat_floor_has_no_tilted_components(self):
        points = np.array([(x, y, -1.0) for x in np.linspace(0.6, 2.5, 40) for y in np.linspace(-0.8, 0.8, 35)])
        support = {"usable": True, "plane": [0.0, 0.0, 1.0, 1.0]}
        samples = geometry.tilted_planar_samples(points, support)
        self.assertEqual(len(samples["selected_points"]), 0)
        self.assertEqual(geometry.cluster_tilted_samples(samples, support), [])

    def test_diagnostic_distance_does_not_change_generated_hints(self):
        hints = [{"x": 10, "y": 15}, {"x": 30, "y": 40}]
        original = [dict(row) for row in hints]
        self.assertAlmostEqual(distance_to_hints({"x": 13, "y": 19}, hints), 5.0)
        self.assertEqual(hints, original)
        self.assertIsNone(distance_to_hints({"x": 13, "y": 19}, []))

    def test_traversable_component_gate_rejects_steeper_object_surface(self):
        components = [
            {"median_tilt_degrees": 7.2, "name": "shallow"},
            {"median_tilt_degrees": 11.9, "name": "steeper"},
        ]
        self.assertEqual(
            [row["name"] for row in geometry.select_traversable_components(components, 10)], ["shallow"]
        )
        with self.assertRaises(ValueError):
            geometry.select_traversable_components(components, 25)

    def test_representative_pixels_are_deterministic_and_in_component(self):
        pixels = np.array([[10, 10], [11, 10], [30, 10], [10, 30], [30, 30]], dtype=np.int32)
        first = geometry._representative_pixels(pixels, 3)
        second = geometry._representative_pixels(pixels, 3)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(all(any(np.array_equal(row, source) for source in pixels) for row in first))

    def test_geometry_sam_points_are_unique_and_positive(self):
        hints = [
            {"points": [{"x": 1, "y": 2, "label": 1}, {"x": 3, "y": 4, "label": 1}]},
            {"points": [{"x": 1, "y": 2, "label": 1}]},
        ]
        self.assertEqual(prompt_points(hints), [
            {"x": 1, "y": 2, "label": 1}, {"x": 3, "y": 4, "label": 1}
        ])
        with self.assertRaises(ValueError):
            prompt_points([{"points": [{"x": 1, "y": 2, "label": 0}]}])

    def test_polygon_mask_preserves_parts_and_holes(self):
        result = {
            "points": [[1, 1], [8, 1], [8, 8], [1, 8]],
            "parts": [{"points": [[12, 12], [16, 12], [16, 16], [12, 16]],
                       "holes": [[[13, 13], [14, 13], [14, 14], [13, 14]]]}],
            "holes": [[[3, 3], [6, 3], [6, 6], [3, 6]]],
        }
        mask = polygon_mask(result, 20, 20)
        self.assertEqual(mask[2, 2], 1)
        self.assertEqual(mask[4, 4], 0)
        self.assertEqual(mask[12, 12], 1)
        self.assertEqual(mask[13, 13], 0)

    def test_ring_constraints_use_component_centers_and_outside_negatives(self):
        hints = [
            {"x": 50, "y": 50, "pixel_bbox": [40, 40, 60, 60]},
            {"x": 80, "y": 55, "pixel_bbox": [70, 45, 90, 65]},
        ]
        points = ring_constrained_points(hints, width=140, height=120, margin=20)
        self.assertEqual(points[:2], [
            {"x": 50, "y": 50, "label": 1}, {"x": 80, "y": 55, "label": 1}
        ])
        self.assertGreaterEqual(len(points[2:]), 2)
        for point in points[2:]:
            self.assertEqual(point["label"], 0)
            self.assertFalse(any(box[0] <= point["x"] <= box[2] and box[1] <= point["y"] <= box[3]
                                 for box in ([40, 40, 60, 60], [70, 45, 90, 65])))

    def test_geometry_mask_metrics_reject_spill_and_rank_local_mask_first(self):
        hints = [{"x": 50, "y": 50, "pixel_bbox": [40, 40, 60, 60],
                  "points": [{"x": 50, "y": 50, "label": 1}]}]
        local = np.zeros((100, 100), dtype=np.uint8)
        local[40:61, 40:61] = 1
        spilled = np.ones((100, 100), dtype=np.uint8)
        local_metrics = geometry_mask_metrics(local, hints, padding=10)
        spilled_metrics = geometry_mask_metrics(spilled, hints, padding=10)
        self.assertTrue(local_metrics["geometry_gate_passed"])
        self.assertFalse(spilled_metrics["geometry_gate_passed"])
        first = {"metrics": local_metrics, "warnings": []}
        second = {"metrics": spilled_metrics, "warnings": []}
        self.assertGreater(geometry_attempt_rank(first), geometry_attempt_rank(second))

    def test_external_freeze_uses_exact_intersection_and_even_endpoints(self):
        self.assertEqual(evenly_spaced([str(value) for value in range(10)], 4), ["0", "3", "6", "9"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for camera in range(4):
                directory = root / "cameras_640x512" / f"camera_{camera}"
                directory.mkdir(parents=True)
                for frame in ("1", "2", "3", "4"):
                    (directory / f"{frame}.png").write_bytes(f"{camera}-{frame}".encode())
            lidar = root / "lidar"
            lidar.mkdir()
            for frame in ("1", "2", "3", "4", "5"):
                (lidar / f"{frame}.pcd").write_bytes(frame.encode())
            calibration = root / "calib_640x512"
            calibration.mkdir()
            for name in [*(f"camera{camera}_calib.yaml" for camera in range(4)), "lidar2camera.yaml"]:
                (calibration / name).write_text(name, encoding="utf-8")
            manifest = freeze(root, "dataset", "001", 4)
        self.assertEqual(manifest["selection"]["population_frames"], 4)
        self.assertEqual(manifest["selection"]["frame_ids"], ["1", "2", "3", "4"])
        self.assertFalse(manifest["boundaries"]["labels_read"])

    def test_external_manifest_verifier_rehashes_every_bound_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for camera in range(4):
                directory = root / "cameras_640x512" / f"camera_{camera}"
                directory.mkdir(parents=True)
                for frame in ("1", "2", "3", "4"):
                    (directory / f"{frame}.png").write_bytes(f"{camera}-{frame}".encode())
            lidar = root / "lidar"
            lidar.mkdir()
            for frame in ("1", "2", "3", "4"):
                (lidar / f"{frame}.pcd").write_bytes(frame.encode())
            calibration = root / "calib_640x512"
            calibration.mkdir()
            for name in [*(f"camera{camera}_calib.yaml" for camera in range(4)), "lidar2camera.yaml"]:
                (calibration / name).write_text(name, encoding="utf-8")
            manifest = freeze(root, "dataset", "001", 4)
            manifest_path = root / "manifest.json"
            import json
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            _, verified = verify_manifest(manifest_path)
            self.assertEqual(len(verified), 25)
            (lidar / "2.pcd").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed after selection"):
                verify_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()
