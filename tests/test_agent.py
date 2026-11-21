"""Regression tests for bounded editing and persisted agent decisions."""
import io
import base64
import copy
import json
import logging
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from source.agent import mcap_video, server, workflow
from source.agent.edit_agent import (_hole_regions, _candidate_gap_regions, _shape_from_mask,
                                     _rasterize, _valid_region, apply_operations, valid_shape)


def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(output, "JPEG")
    return output.getvalue()


def sized_image_bytes(width, height):
    output = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(output, "JPEG")
    return output.getvalue()


def shape(shape_id=1, label="a", left=2, right=18):
    points = [[left, 2], [right, 2], [right, 18], [left, 18]]
    part = {"points": points, "holes": []}
    return {"id": shape_id, "label": label, "points": points,
            "holes": [], "parts": [part]}


def sam_result():
    return {"parts": shape()["parts"], "score": 0.9}


class EditTests(unittest.TestCase):
    def test_candidate_class_audit_progress_metrics_are_user_facing_counts(self):
        audits = [
            {"execution_mode": "independent_batch_3", "route": "candidate_preserved",
             "protocol_retry_count": 0},
            {"execution_mode": "independent_batch_3", "route": "human_review_required",
             "protocol_retry_count": 0},
            {"execution_mode": "sequential", "route": "human_review_required",
             "protocol_retry_count": 1, "batch_fallback_reason": "invalid"},
        ]
        self.assertEqual(server.candidate_class_audit_progress_metrics(audits), {
            "audit_batch_completed": 2,
            "audit_sequential_completed": 1,
            "audit_review_required": 2,
            "audit_retry_count": 2,
        })

    def test_routine_werkzeug_access_lines_are_suppressed_but_warnings_remain(self):
        self.assertEqual(logging.getLogger("werkzeug").level, logging.WARNING)

    def test_run_status_exposes_current_file_activation_generation(self):
        with server.app.test_client() as client:
            payload = client.get("/api/run/status").get_json()
        self.assertEqual(
            payload["selected_file_activation_generation"],
            server.file_activation_generation,
        )

    def test_process_file_activation_generation_is_positive_and_javascript_safe(self):
        self.assertGreater(server.file_activation_generation, 0)
        self.assertLessEqual(server.file_activation_generation, (1 << 53) - 1)

    def test_open_line_gap_is_detected_and_fill_preserves_excluded_object(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[5:95, 5:95] = 1
        mask[35:65, 35:65] = 0  # Excluded robot.
        mask[5:40, 48:52] = 0  # Thin line connected to robot and background.
        candidate = _shape_from_mask(mask, "地面", 1)
        self.assertFalse(_hole_regions(candidate, 100, 100))
        gaps = _candidate_gap_regions(candidate, 100, 100)
        gap = next(region for region in gaps if region["mask"][20, 50])
        self.assertEqual(gap["kind"], "open_gap")
        updated, applied, errors = apply_operations([candidate], [{
            "operation": "fill_gap", "shape_ids": [1], "hole_id": gap["id"],
            "polygon": [[0, 0], [99, 0], [99, 99], [0, 99]]}], ["地面"], 100, 100)
        self.assertFalse(errors)
        result = _rasterize(updated[0], 100, 100)
        self.assertEqual(result[20, 50], 1)
        self.assertEqual(result[50, 50], 0)
        added = (result != 0) & (mask == 0)
        self.assertTrue(np.all(~added | (gap["mask"] != 0)))
        self.assertTrue(np.all(result[mask != 0] == 1))
        self.assertEqual(applied[0]["filled_pixels"], int(added.sum()))

    def test_partial_gap_edits_share_observation_ids_and_preserve_other_holes(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[15:30, 15:30] = 0
        mask[60:80, 60:80] = 0
        candidate = _shape_from_mask(mask, "地面", 1)
        self.assertTrue(np.array_equal(mask, _rasterize(candidate, 100, 100)))
        gap = _hole_regions(candidate, 100, 100)[0]
        operations = [{"operation": "fill_gap", "shape_ids": [1], "hole_id": gap["id"],
                       "polygon": polygon} for polygon in (
                           [[15, 15], [21, 15], [21, 29], [15, 29]],
                           [[22, 15], [29, 15], [29, 29], [22, 29]])]
        result, applied, errors = apply_operations([candidate], operations, ["地面"], 100, 100)
        self.assertFalse(errors)
        self.assertEqual(len(applied), 2)
        expected = mask.copy()
        expected[15:30, 15:30] = 1
        self.assertTrue(np.array_equal(expected, _rasterize(result[0], 100, 100)))

    def test_shape_and_polygon_validation(self):
        self.assertTrue(valid_shape(shape(), 40, 40))
        invalid = shape()
        invalid["parts"][0]["holes"] = [[[0, 0], [float("inf"), 0], [0, 2]]]
        self.assertFalse(valid_shape(invalid, 40, 40))
        updated, applied, queue = apply_operations([shape()], [
            {"operation": "add_region", "shape_ids": [1],
             "polygon": [[0, 0], [40, 0], [0, 2]]}], ["a"], 40, 40)
        self.assertEqual(updated, [shape()])
        self.assertFalse(applied)
        self.assertTrue(queue)
        self.assertFalse(_valid_region([[5, 5], [5, 5], [5, 5]], 40, 40))
        self.assertFalse(_valid_region([[2, 2], [5, 5], [8, 8]], 40, 40))
        self.assertTrue(_valid_region([[2, 2], [8, 2], [8, 8]], 40, 40))

    def test_merge_and_split_preserve_pixels(self):
        original = [shape(), shape(2, left=22, right=35)]
        merged, applied, queue = apply_operations(original, [
            {"operation": "merge", "shape_ids": [1, 2], "label": "a"}],
            ["a"], 40, 40, next_id=3)
        self.assertFalse(queue)
        self.assertEqual([item["id"] for item in merged], [3])
        self.assertEqual(int(_rasterize(merged[0], 40, 40).sum()),
                         sum(int(_rasterize(item, 40, 40).sum()) for item in original))
        split, applied, queue = apply_operations([shape()], [
            {"operation": "split", "shape_ids": [1],
             "polygon": [[0, 0], [10, 0], [10, 39], [0, 39]]}],
            ["a"], 40, 40, next_id=2)
        self.assertFalse(queue)
        self.assertEqual([item["id"] for item in split], [1, 2])
        self.assertEqual(sum(int(_rasterize(item, 40, 40).sum()) for item in split),
                         int(_rasterize(shape(), 40, 40).sum()))

    def test_fill_gap_polygon_is_clipped_to_the_named_hole(self):
        donut = {"id": 1, "label": "地面", "points": [[2, 2], [30, 2], [30, 30], [2, 30]],
                 "holes": [], "parts": [{"points": [[2, 2], [30, 2], [30, 30], [2, 30]],
                                           "holes": []}]}
        donut["holes"] = [[[10, 10], [20, 10], [20, 20], [10, 20]]]
        donut["parts"][0]["holes"] = donut["holes"]
        hole = _hole_regions(donut, 40, 40)[0]
        updated, applied, queue = apply_operations([donut], [{
            "operation": "fill_gap", "shape_ids": [1], "hole_id": hole["id"],
            "polygon": [[5, 5], [25, 5], [25, 25], [5, 25]]}], ["地面"], 40, 40)
        self.assertFalse(queue)
        self.assertEqual(applied[0]["filled_pixels"], int(hole["mask"].sum()))
        result = _rasterize(updated[0], 40, 40)
        self.assertEqual(result[15, 15], 1)
        before = _rasterize(donut, 40, 40)
        added = (result != 0) & (before == 0)
        expected_added = (hole["mask"] != 0) & (before == 0)
        self.assertEqual(int(added.sum()), int(expected_added.sum()))
        self.assertTrue((added <= (hole["mask"] != 0)).all())

    def test_rejected_edits_leave_original(self):
        for operation in (
                {"operation": "review", "shape_ids": [1]},
                {"operation": "merge", "shape_ids": [1, 1], "label": "a"},
                {"operation": "erase_region", "shape_ids": [1],
                 "polygon": [[0, 0], [39, 0], [39, 39], [0, 39]]},
                {"operation": "remove", "shape_ids": [99]}):
            with self.subTest(operation=operation):
                updated, applied, queue = apply_operations([shape()], [operation],
                    ["a"], 40, 40)
                self.assertEqual(updated, [shape()])
                self.assertFalse(applied)
                self.assertTrue(queue)

    def test_large_traversable_surface_is_not_reclassified_as_instance(self):
        large_floor = shape(label="floor", left=1, right=38)
        large_floor["points"] = [[1, 1], [38, 1], [38, 38], [1, 38]]
        large_floor["parts"] = [{"points": large_floor["points"], "holes": []}]
        updated, applied, queue = apply_operations([large_floor], [{
            "operation": "reclassify", "shape_ids": [1], "label": "ego",
            "reason": "Foreground chassis is visible",
        }], ["floor", "ego"], 40, 40)
        self.assertEqual(updated[0]["label"], "floor")
        self.assertFalse(applied)
        self.assertEqual(queue[0]["reason_code"],
                         "global_surface_reclassify_requires_human_review")
        self.assertGreater(queue[0]["area_ratio"], 0.35)

    def test_small_surface_candidate_can_still_be_reclassified(self):
        candidate = shape(label="floor", left=2, right=12)
        updated, applied, queue = apply_operations([candidate], [{
            "operation": "reclassify", "shape_ids": [1], "label": "ego",
            "reason": "The entire small candidate is host chassis",
        }], ["floor", "ego"], 40, 40)
        self.assertEqual(updated[0]["label"], "ego")
        self.assertTrue(applied)
        self.assertFalse(queue)

    def test_large_floor_erase_is_preserved_for_human_review(self):
        large_floor = shape(label="floor", left=1, right=38)
        large_floor["points"] = [[1, 1], [38, 1], [38, 38], [1, 38]]
        large_floor["parts"] = [{"points": large_floor["points"], "holes": []}]
        updated, applied, queue = apply_operations([large_floor], [{
            "operation": "erase_region", "shape_ids": [1],
            "polygon": [[0, 0], [39, 0], [39, 20], [0, 20]],
            "reason": "remove a suspected foreground object",
        }], ["floor"], 40, 40)
        self.assertEqual(updated, [large_floor])
        self.assertFalse(applied)
        self.assertEqual(queue[0]["reason_code"],
                         "global_surface_large_erase_requires_human_review")
        self.assertGreater(queue[0]["removed_fraction"], 0.25)

    def test_small_floor_erase_remains_available(self):
        large_floor = shape(label="floor", left=1, right=38)
        large_floor["points"] = [[1, 1], [38, 1], [38, 38], [1, 38]]
        large_floor["parts"] = [{"points": large_floor["points"], "holes": []}]
        updated, applied, queue = apply_operations([large_floor], [{
            "operation": "erase_region", "shape_ids": [1],
            "polygon": [[1, 1], [5, 1], [5, 5], [1, 5]],
            "reason": "remove a bounded local foreground object",
        }], ["floor"], 40, 40)
        self.assertNotEqual(updated, [large_floor])
        self.assertTrue(applied)
        self.assertFalse(queue)

    def test_unreferenced_high_risk_reclassification_requires_human_review(self):
        person = shape(label="person")
        updated, applied, queue = apply_operations([person], [{
            "operation": "reclassify", "shape_ids": [1], "label": "ego",
            "reason": "foreground host structure",
        }], ["person", "ego"], 40, 40, allow_high_risk_reclassify=False)
        self.assertEqual(updated, [person])
        self.assertFalse(applied)
        self.assertEqual(queue[0]["reason_code"],
                         "unreferenced_high_risk_reclassify_requires_human_review")

    def test_reference_backed_high_risk_reclassification_remains_available(self):
        person = shape(label="person")
        updated, applied, queue = apply_operations([person], [{
            "operation": "reclassify", "shape_ids": [1], "label": "ego",
            "reason": "human reference identifies host chassis",
        }], ["person", "ego"], 40, 40, allow_high_risk_reclassify=True)
        self.assertEqual(updated[0]["label"], "ego")
        self.assertTrue(applied)
        self.assertFalse(queue)

    def test_unreferenced_floor_to_slope_reclassification_requires_human_review(self):
        floor = shape(label="floor", left=1, right=38)
        updated, applied, queue = apply_operations([floor], [{
            "operation": "reclassify", "shape_ids": [1], "label": "slope",
            "reason": "overlay color looked like a ramp",
        }], ["floor", "slope"], 40, 40, allow_high_risk_reclassify=False)
        self.assertEqual(updated, [floor])
        self.assertFalse(applied)
        self.assertEqual(
            queue[0]["reason_code"],
            "unreferenced_global_surface_reclassify_requires_human_review")
        self.assertEqual(queue[0]["source_label"], "floor")
        self.assertEqual(queue[0]["target_label"], "slope")

    def test_reference_backed_floor_to_slope_reclassification_remains_available(self):
        floor = shape(label="floor")
        updated, applied, queue = apply_operations([floor], [{
            "operation": "reclassify", "shape_ids": [1], "label": "slope",
            "reason": "reviewed reference confirms the inclined ramp",
        }], ["floor", "slope"], 40, 40, allow_high_risk_reclassify=True)
        self.assertEqual(updated[0]["label"], "slope")
        self.assertTrue(applied)
        self.assertFalse(queue)

    def test_unreferenced_large_floor_expansion_requires_human_review(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[20:39, 2:39] = 1
        floor = _shape_from_mask(mask, "floor", 1)
        updated, applied, queue = apply_operations([floor], [{
            "operation": "add_region", "shape_ids": [1],
            "polygon": [[0, 0], [39, 0], [39, 25], [0, 25]],
            "reason": "expand floor across the upper image",
        }], ["floor"], 40, 40,
            allow_large_global_surface_expansion=False)
        self.assertEqual(updated, [floor])
        self.assertFalse(applied)
        self.assertEqual(
            queue[0]["reason_code"],
            "unreferenced_global_surface_large_add_requires_human_review")
        self.assertEqual(queue[0]["route"], "human_review_required")
        self.assertGreater(queue[0]["image_fraction"], 0.05)

    def test_unreferenced_small_local_floor_repair_remains_available(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:39, 2:39] = 1
        floor = _shape_from_mask(mask, "floor", 1)
        updated, applied, queue = apply_operations([floor], [{
            "operation": "add_region", "shape_ids": [1],
            "polygon": [[5, 7], [10, 7], [10, 11], [5, 11]],
            "reason": "repair a small visible floor boundary notch",
        }], ["floor"], 40, 40,
            allow_large_global_surface_expansion=False)
        self.assertNotEqual(updated, [floor])
        self.assertTrue(applied)
        self.assertFalse(queue)

    def test_broad_floor_polygon_cannot_fill_small_excluded_object_hole(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[1:39, 1:39] = 1
        mask[16:21, 16:21] = 0
        floor = _shape_from_mask(mask, "floor", 1)
        updated, applied, queue = apply_operations([floor], [{
            "operation": "add_region", "shape_ids": [1],
            "polygon": [[1, 1], [38, 1], [38, 38], [1, 38]],
            "reason": "fill the entire floor envelope",
        }], ["floor"], 40, 40,
            allow_large_global_surface_expansion=False)
        self.assertEqual(updated, [floor])
        self.assertFalse(applied)
        self.assertEqual(
            queue[0]["reason_code"],
            "unreferenced_global_surface_large_add_requires_human_review")
        self.assertGreater(queue[0]["region_fraction"], 0.25)
        self.assertLess(queue[0]["image_fraction"], 0.05)

    def test_reference_backed_large_floor_expansion_remains_available(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[20:39, 2:39] = 1
        floor = _shape_from_mask(mask, "floor", 1)
        updated, applied, queue = apply_operations([floor], [{
            "operation": "add_region", "shape_ids": [1],
            "polygon": [[0, 0], [39, 0], [39, 25], [0, 25]],
            "reason": "human reference confirms the missing surface",
        }], ["floor"], 40, 40,
            allow_large_global_surface_expansion=True)
        self.assertNotEqual(updated, [floor])
        self.assertTrue(applied)
        self.assertFalse(queue)

    def test_instance_candidate_cannot_be_reclassified_wholesale_as_floor(self):
        pallet = shape(label="pallet")
        updated, applied, queue = apply_operations([pallet], [{
            "operation": "reclassify", "shape_ids": [1], "label": "floor",
            "reason": "single visual review guessed that the close pallet is floor",
        }], ["pallet", "floor"], 40, 40, allow_high_risk_reclassify=True)
        self.assertEqual(updated, [pallet])
        self.assertFalse(applied)
        self.assertEqual(
            queue[0]["reason_code"],
            "instance_to_global_surface_requires_human_review")
        self.assertEqual(queue[0]["source_label"], "pallet")
        self.assertEqual(queue[0]["target_label"], "floor")


class WorkflowTests(unittest.TestCase):
    def test_pallet_cleanup_fills_texture_holes_and_preserves_structural_opening(self):
        mask = np.zeros((100, 140), dtype=np.uint8)
        mask[10:90, 10:130] = 1
        mask[30:35, 30:35] = 0       # texture-scale enclosed hole
        mask[62:75, 92:105] = 0      # larger compact texture hole
        mask[55:61, 45:75] = 0       # structural elongated slot
        mask[9, 60] = 1              # one-pixel outer spike
        mask[10, 90] = 0             # one-pixel boundary notch
        candidate = _shape_from_mask(mask, "pallet", 7)
        candidate["score"] = 0.875

        refined, audit = workflow.refine_pallet_candidate_geometry(
            candidate, 140, 100)
        output = _rasterize(refined, 140, 100)

        self.assertIsNotNone(audit)
        self.assertEqual(audit["reason_code"], "pallet_texture_geometry_cleanup")
        self.assertEqual(audit["route"], "candidate_only")
        self.assertEqual(output[32, 32], 1)
        self.assertEqual(output[68, 98], 1)
        self.assertEqual(output[58, 60], 0)
        self.assertEqual(output[9, 60], 0)
        self.assertEqual(output[10, 90], 1)
        self.assertEqual(refined["score"], candidate["score"])
        self.assertTrue(audit["smoothing_applied"])
        self.assertEqual(audit["smoothing_kernel"], 5)
        self.assertEqual(audit["filled_texture_hole_count"], 1)

    def test_pallet_cleanup_preserves_imported_string_candidate_id(self):
        mask = np.zeros((100, 140), dtype=np.uint8)
        mask[10:90, 10:130] = 1
        mask[25:30, 25:30] = 0
        shape = _shape_from_mask(mask, "pallet", 1)
        shape["id"] = "sam3-camera_0-7-pallet-2"

        refined, audit = workflow.refine_pallet_candidate_geometry(
            shape, 140, 100)

        self.assertIsNotNone(audit)
        self.assertEqual(refined["id"], "sam3-camera_0-7-pallet-2")
        self.assertGreaterEqual(audit["filled_hole_count"], 1)
        self.assertEqual(audit["geometry_cleanup_version"], 5)
        self.assertEqual(audit["route"], "candidate_only")

    def test_pallet_cleanup_keeps_safe_hole_fill_when_smoothing_exceeds_budget(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[40:440, 80:560] = 1
        placed = 0
        for y in range(60, 420, 24):
            for x in range(100, 540, 24):
                if placed >= 120:
                    break
                mask[y:y + 5, x:x + 5] = 0
                placed += 1
            if placed >= 120:
                break
        # Optional median smoothing changes these high-frequency outer-edge
        # notches.  Combined with the safe enclosed-hole fills it exceeds the
        # 2.5 percent total budget, so only the hole cleanup may survive.
        for x in range(80, 560, 8):
            mask[40:44, x:x + 4] = 0
            mask[436:440, x + 4:x + 8] = 0
        shape = _shape_from_mask(mask, "pallet", 17)

        refined, audit = workflow.refine_pallet_candidate_geometry(shape, 640, 480)

        self.assertIsNotNone(audit)
        self.assertTrue(audit["smoothing_rejected_by_budget"])
        self.assertFalse(audit["smoothing_applied"])
        self.assertEqual(audit["filled_hole_count"], 120)
        self.assertLessEqual(audit["change_ratio"], .025)
        after = _rasterize(refined, 640, 480)
        self.assertEqual(after[62, 102], 1)
        # The optional outer-edge smoothing was rejected, so the attached
        # background notch remains unchanged.
        self.assertEqual(after[41, 81], 0)
        self.assertLessEqual(audit["change_ratio"], 0.025)

    def test_pallet_cleanup_fills_compact_surface_breaks_and_keeps_long_slots(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[245:480, :] = 1
        # Six compact, irregular surface drop-outs from the close-pallet case.
        compact_holes = [
            (274, 247, 10, 22), (292, 397, 27, 17), (299, 229, 22, 20),
            (301, 174, 19, 51), (301, 335, 19, 51), (303, 125, 19, 21),
        ]
        # Four long horizontal board gaps must remain excluded.
        structural_slots = [
            (378, 384, 13, 59), (380, 340, 10, 39),
            (381, 134, 10, 48), (381, 190, 11, 66),
        ]
        for y, x, h, w in compact_holes:
            cv2.ellipse(mask, (x + w // 2, y + h // 2),
                        (max(w // 2 - 1, 1), max(h // 2 - 1, 1)),
                        0, 0, 360, 0, thickness=cv2.FILLED)
        for y, x, h, w in structural_slots:
            mask[y:y + h, x:x + w] = 0
        candidate = _shape_from_mask(mask, "pallet", 24)

        refined, audit = workflow.refine_pallet_candidate_geometry(
            candidate, 640, 480)
        output = _rasterize(refined, 640, 480)

        self.assertIsNotNone(audit)
        self.assertEqual(audit["geometry_cleanup_version"], 5)
        self.assertEqual(audit["filled_hole_count"], 6)
        self.assertGreaterEqual(audit["filled_compact_texture_hole_count"], 4)
        for y, x, h, w in compact_holes:
            self.assertEqual(output[y + h // 2, x + w // 2], 1)
        for y, x, h, w in structural_slots:
            self.assertEqual(output[y + h // 2, x + w // 2], 0)
        self.assertGreaterEqual(audit["minimum_structural_hole_retention"], .75)
        self.assertLessEqual(audit["change_ratio"], .025)

    def test_pallet_cleanup_preserves_compact_opening_on_medium_plastic_instance(self):
        mask = np.zeros((300, 420), dtype=np.uint8)
        mask[60:240, 120:300] = 1
        cv2.circle(mask, (210, 150), 12, 0, thickness=cv2.FILLED)
        candidate = _shape_from_mask(mask, "pallet", 25)

        refined, audit = workflow.refine_pallet_candidate_geometry(
            candidate, 420, 300)
        output = _rasterize(refined, 420, 300)

        self.assertEqual(output[150, 210], 0)
        self.assertIsNotNone(audit)
        self.assertEqual(audit["geometry_cleanup_version"], 5)
        self.assertEqual(audit["filled_compact_texture_hole_count"], 0)
        self.assertGreaterEqual(audit["minimum_structural_hole_retention"], .75)

    def test_image_guided_pallet_cleanup_fills_wood_texture_and_keeps_dark_slots(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[245:480, :] = 1
        cv2.ellipse(mask, (290, 280), (42, 13), 0, 0, 360, 0,
                    thickness=cv2.FILLED)
        mask[245:267, 405:415] = 0
        mask[378:392, 120:250] = 0
        mask[378:392, 340:455] = 0
        image = np.full((480, 640, 3), 24, dtype=np.uint8)
        image[245:480, :] = np.array([205, 190, 170], dtype=np.uint8)
        # The SAM drop-out is still visibly the same wooden surface.
        image[267:294, 248:333] = np.array([198, 183, 164], dtype=np.uint8)
        # Real openings expose a much darker background.
        image[378:392, 120:250] = 18
        image[378:392, 340:455] = 18
        candidate = _shape_from_mask(mask, "pallet", 26)

        refined, audit = workflow.refine_pallet_candidate_geometry(
            candidate, 640, 480, image_rgb=image)
        output = _rasterize(refined, 640, 480)

        self.assertIsNotNone(audit)
        self.assertEqual(audit["geometry_cleanup_version"], 5)
        self.assertGreaterEqual(audit["filled_image_texture_hole_count"], 1)
        self.assertGreaterEqual(audit["filled_image_boundary_notch_count"], 1)
        self.assertEqual(output[280, 290], 1)
        self.assertEqual(output[255, 410], 1)
        self.assertEqual(output[385, 180], 0)
        self.assertEqual(output[385, 400], 0)
        self.assertGreaterEqual(audit["minimum_structural_hole_retention"], .75)
        self.assertLessEqual(audit["change_ratio"], .06)

    def test_image_guided_pallet_cleanup_preserves_dark_plastic_grid_openings(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[180:460, 80:560] = 1
        openings = [(180, 260), (270, 260), (360, 260), (450, 260)]
        for x, y in openings:
            cv2.circle(mask, (x, y), 12, 0, thickness=cv2.FILLED)
        mask[180:205, 315:325] = 0
        image = np.full((480, 640, 3), 20, dtype=np.uint8)
        image[180:460, 80:560] = np.array([25, 90, 165], dtype=np.uint8)
        for x, y in openings:
            cv2.circle(image, (x, y), 12, (8, 12, 18), thickness=cv2.FILLED)
        image[180:205, 315:325] = np.array([8, 12, 18], dtype=np.uint8)
        candidate = _shape_from_mask(mask, "pallet", 27)

        refined, audit = workflow.refine_pallet_candidate_geometry(
            candidate, 640, 480, image_rgb=image)
        output = _rasterize(refined, 640, 480)

        for x, y in openings:
            self.assertEqual(output[y, x], 0)
        self.assertEqual(output[190, 320], 0)
        self.assertIsNotNone(audit)
        self.assertEqual(audit["geometry_cleanup_version"], 5)
        self.assertEqual(audit["filled_image_texture_hole_count"], 0)
        self.assertEqual(audit["filled_image_boundary_notch_count"], 0)
        self.assertEqual(audit["filled_compact_texture_hole_count"], 0)
        self.assertGreaterEqual(audit["minimum_structural_hole_retention"], .75)

    def test_image_guided_boundary_notch_can_merge_two_local_wood_fragments(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:300, 40:320] = 1
        mask[220:470, 330:600] = 1
        image = np.full((480, 640, 3), 24, dtype=np.uint8)
        image[100:300, 40:320] = np.array([205, 190, 170], dtype=np.uint8)
        image[220:470, 330:600] = np.array([205, 190, 170], dtype=np.uint8)
        image[220:300, 320:330] = np.array([202, 188, 169], dtype=np.uint8)
        candidate = _shape_from_mask(mask, "pallet", 28)

        refined, audit = workflow.refine_pallet_candidate_geometry(
            candidate, 640, 480, image_rgb=image)
        output = _rasterize(refined, 640, 480)

        self.assertIsNotNone(audit)
        self.assertEqual(output[260, 325], 1)
        self.assertGreaterEqual(audit["filled_image_boundary_notch_count"], 1)
        self.assertEqual(audit["component_count_before"], 2)
        self.assertEqual(audit["component_count_after"], 1)
        self.assertEqual(audit["component_merge_count"], 1)
        self.assertLessEqual(audit["component_merge_count"], 2)

    def test_image_guided_boundary_notch_skips_distant_small_pallet(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[180:260, 100:220] = 1
        mask[180:260, 230:350] = 1
        image = np.full((480, 640, 3), 24, dtype=np.uint8)
        image[180:260, 100:220] = np.array([205, 190, 170], dtype=np.uint8)
        image[180:260, 230:350] = np.array([205, 190, 170], dtype=np.uint8)
        image[180:260, 220:230] = np.array([202, 188, 169], dtype=np.uint8)
        candidate = _shape_from_mask(mask, "pallet", 29)

        refined, audit = workflow.refine_pallet_candidate_geometry(
            candidate, 640, 480, image_rgb=image)
        output = _rasterize(refined, 640, 480)

        self.assertIsNotNone(audit)
        self.assertEqual(output[220, 225], 0)
        self.assertEqual(audit["filled_image_boundary_notch_count"], 0)
        self.assertEqual(audit["component_merge_count"], 0)

    def test_image_guided_boundary_notch_never_merges_more_than_two_components(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        regions = [
            (20, 20, 160, 200),
            (170, 120, 310, 300),
            (320, 220, 460, 400),
            (470, 120, 610, 300),
        ]
        image = np.full((480, 640, 3), 24, dtype=np.uint8)
        for left, top, right, bottom in regions:
            mask[top:bottom, left:right] = 1
            image[top:bottom, left:right] = np.array(
                [205, 190, 170], dtype=np.uint8)
        image[120:200, 160:170] = np.array([202, 188, 169], dtype=np.uint8)
        image[220:300, 310:320] = np.array([202, 188, 169], dtype=np.uint8)
        image[220:300, 460:470] = np.array([202, 188, 169], dtype=np.uint8)
        candidate = _shape_from_mask(mask, "pallet", 30)

        refined, audit = workflow.refine_pallet_candidate_geometry(
            candidate, 640, 480, image_rgb=image)
        output = _rasterize(refined, 640, 480)

        self.assertIsNotNone(audit)
        self.assertEqual(audit["component_count_before"], 4)
        self.assertEqual(audit["component_count_after"], 2)
        self.assertEqual(audit["component_merge_count"], 2)
        self.assertLessEqual(audit["component_merge_count"], 2)
        self.assertEqual(workflow._significant_component_count(output), 2)

    def test_close_pallet_cleanup_uses_stronger_bounded_boundary_smoothing(self):
        mask = np.zeros((260, 360), dtype=np.uint8)
        mask[20:240, 20:340] = 1
        mask[80:88, 80:110] = 0      # long structural slit remains
        mask[120:135, 150:165] = 0   # compact texture hole is filled
        mask[19, 200] = 1            # shallow outer spike is smoothed
        candidate = _shape_from_mask(mask, "pallet", 8)

        refined, audit = workflow.refine_pallet_candidate_geometry(
            candidate, 360, 260)
        output = _rasterize(refined, 360, 260)

        self.assertIsNotNone(audit)
        self.assertEqual(audit["smoothing_kernel"], 7)
        self.assertEqual(output[126, 156], 1)
        self.assertEqual(output[84, 95], 0)
        self.assertEqual(output[19, 200], 0)
        self.assertLessEqual(audit["change_ratio"], 0.025)

    def test_pallet_boundary_simplification_reduces_pixel_staircase_vertices(self):
        mask = np.zeros((300, 420), dtype=np.uint8)
        for y in range(30, 270):
            left = 30 + (y % 6)
            right = 390 - ((y + 2) % 6)
            mask[y, left:right] = 1
        mask[110:120, 120:180] = 0  # structural long slot remains excluded
        candidate = _shape_from_mask(mask, "pallet", 21)

        refined, audit = workflow.refine_pallet_candidate_geometry(
            candidate, 420, 300)
        output = _rasterize(refined, 420, 300)

        self.assertIsNotNone(audit)
        self.assertTrue(audit["boundary_simplification_applied"])
        self.assertIn(audit["boundary_simplification_epsilon"], (.75, .5))
        self.assertLess(audit["boundary_vertices_after"],
                        audit["boundary_vertices_before"])
        self.assertGreater(audit["boundary_vertex_reduction_ratio"], .1)
        self.assertLessEqual(audit["change_ratio"], .025)
        self.assertGreaterEqual(audit["minimum_structural_hole_retention"], .75)
        self.assertEqual(output[115, 150], 0)

    def test_pallet_boundary_simplification_fails_closed_when_budget_would_break(self):
        mask = np.zeros((240, 320), dtype=np.uint8)
        cv2.circle(mask, (160, 120), 90, 1, thickness=cv2.FILLED)
        candidate = _shape_from_mask(mask, "pallet", 22)

        def triangle(contour, epsilon, closed):
            points = np.asarray(contour).reshape(-1, 2)
            minimum = points.min(axis=0)
            maximum = points.max(axis=0)
            return np.asarray([
                [[minimum[0], minimum[1]]],
                [[maximum[0], minimum[1]]],
                [[minimum[0], maximum[1]]],
            ], dtype=np.float32)

        with patch.object(workflow.cv2, "approxPolyDP", side_effect=triangle):
            refined, audit = workflow.refine_pallet_candidate_geometry(
                candidate, 320, 240)

        self.assertIsNotNone(audit)
        self.assertFalse(audit["boundary_simplification_applied"])
        self.assertLessEqual(audit["change_ratio"], .025)
        self.assertGreaterEqual(
            np.logical_and(_rasterize(refined, 320, 240) != 0, mask != 0).sum()
            / max(int(mask.sum()), 1), .975)

    def test_pallet_cleanup_rejects_polygon_rebuild_that_loses_thin_component(self):
        large = {"points": [[10, 10], [290, 10], [290, 210], [10, 210]],
                 "holes": []}
        # Twelve raster pixels but only five contour-area pixels: this is a
        # significant thin part that _shape_from_mask would otherwise omit.
        thin = {"points": [[300, 10], [301, 10], [301, 15], [300, 15]],
                "holes": []}
        candidate = {"id": 23, "label": "pallet",
                     "points": large["points"], "holes": [],
                     "parts": [large, thin]}
        before = _rasterize(candidate, 320, 240)

        refined, audit = workflow.refine_pallet_candidate_geometry(
            candidate, 320, 240)

        self.assertIs(refined, candidate)
        self.assertIsNone(audit)
        self.assertTrue(np.array_equal(_rasterize(refined, 320, 240), before))

    def test_pallet_cleanup_removes_micro_hole_left_by_boundary_smoothing(self):
        mask = np.zeros((100, 140), dtype=np.uint8)
        mask[10:90, 10:130] = 1
        candidate = _shape_from_mask(mask, "pallet", 9)
        real_median_blur = workflow.cv2.medianBlur

        def smoothing_with_micro_hole(image, kernel):
            output = real_median_blur(image, kernel)
            output[40:45, 40:45] = 0
            return output

        with patch.object(workflow.cv2, "medianBlur",
                          side_effect=smoothing_with_micro_hole):
            refined, audit = workflow.refine_pallet_candidate_geometry(
                candidate, 140, 100)
        output = _rasterize(refined, 140, 100)

        self.assertEqual(output[42, 42], 1)
        self.assertEqual(audit["post_smoothing_filled_hole_count"], 1)
        self.assertEqual(audit["post_smoothing_filled_hole_pixels"], 25)

    def test_pallet_cleanup_skips_small_distant_instances(self):
        candidate = shape(shape_id=3, label="pallet", left=2, right=12)
        refined, audit = workflow.refine_pallet_candidate_geometry(candidate, 40, 40)
        self.assertIs(refined, candidate)
        self.assertIsNone(audit)

    def test_non_ground_review_does_not_generate_irrelevant_gap_evidence(self):
        agent = workflow.AnnotationAgent(
            {}, image_bytes(), {"prompts": []}, ["pallet"], max_rounds=1,
            reference={"image": image_bytes(), "marked_image": image_bytes(),
                       "annotations": [shape(label="pallet")]})
        agent.shapes = [shape(label="pallet")]
        with patch.object(workflow, "candidate_holes") as holes, \
             patch.object(workflow, "review", return_value={
                 "status": "done", "reason": "No supported edit",
                 "hole_assessment": [], "operations": []}):
            result = agent.execute(skip_generation=True)
        holes.assert_not_called()
        self.assertEqual(result["agent"]["stop_reason"], "done")

    def test_agent_cannot_silently_delete_every_generated_candidate(self):
        agent = workflow.AnnotationAgent(
            {}, image_bytes(), {"prompts": []}, ["a"], max_rounds=1)

        def generate():
            agent.shapes = [shape()]
            agent.generated = 1

        decision = {"status": "continue", "operations": [{
            "operation": "remove", "shape_ids": [1],
            "reason": "model requested full deletion"}],
            "protocol_repairs": []}
        with patch.object(agent, "generate", side_effect=generate), \
             patch.object(workflow, "review", return_value=decision):
            result = agent.execute()
        self.assertEqual(result["shapes"], [shape()])
        self.assertEqual(result["agent"]["stop_reason"],
                         "empty_result_rollback")
        rollback = next(item for item in result["review_queue"]
                        if item.get("reason_code") == "agent_empty_result_rollback")
        self.assertEqual(rollback["restored_candidate_count"], 1)
        self.assertEqual(rollback["route"], "human_review_required")

    def test_unreferenced_review_forbids_conflicting_same_candidate_operations(self):
        rules = workflow.COMPACT_UNREFERENCED_RULES
        self.assertIn("each candidate ID in at most one operation", rules)
        self.assertIn("Never both split and reclassify the same ID", rules)
        self.assertIn("mixed semantics must not be reclassified wholesale", rules)

    def test_local_gap_evidence_shows_original_binary_and_bounded_image_count(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[5:95, 48:52] = 0
        candidate = _shape_from_mask(mask, "地面", 1)
        gaps = workflow.candidate_holes([candidate], ["地面"], 100, 100)
        output = io.BytesIO()
        Image.new("RGB", (100, 100), "yellow").save(output, "JPEG")
        sheets = workflow.gap_detail_images(output.getvalue(), [candidate], gaps)
        self.assertEqual(len(sheets), 1)
        sheet, regions = sheets[0]
        self.assertTrue(regions)
        self.assertEqual(regions[0]["hole_id"], gaps[0]["id"])
        self.assertLessEqual(len(regions), 4)
        picture = Image.open(io.BytesIO(sheet))
        self.assertEqual(picture.width, 1080)

    def test_local_gap_evidence_batches_eight_regions_into_two_pages(self):
        image = io.BytesIO()
        Image.new("RGB", (100, 100), "gray").save(image, "JPEG")
        holes = []
        for index in range(8):
            mask = np.zeros((100, 100), dtype=np.uint8)
            x = 4 + index * 10
            mask[10:16, x:x + 6] = 1
            holes.append({"id": f"1:hole:{index}", "shape_id": 1,
                          "area": int(mask.sum()), "bbox": [x, 10, x + 5, 15],
                          "centroid": [x + 2, 12], "mask": mask})
        pages = workflow.gap_detail_images(
            image.getvalue(), [shape(label="floor")], holes)
        self.assertEqual([len(reports) for _, reports in pages], [4, 4])

    def test_description_uses_original_image_and_separate_outline(self):
        original = image_bytes()
        marked = io.BytesIO()
        Image.new("RGB", (40, 40), (255, 0, 255)).save(marked, "JPEG")
        with patch.object(workflow, "chat", return_value={"description": "地面可见边界。",
                "prompts": [{"label": "地面", "text": "floor"}]}) as chat:
            workflow.describe({}, original, [shape(label="地面")], ["地面"], marked.getvalue())
        self.assertEqual(chat.call_args.args[2], original)
        self.assertEqual(chat.call_args.kwargs["extra_images"], (marked.getvalue(),))
        self.assertIn("轮廓颜色只是界面标记", chat.call_args.args[1])
        self.assertIn("仅依据图像 1", chat.call_args.args[1])

    def test_description_retries_a_schema_bound_string_instead_of_saving_it(self):
        truncated = {"description": "x" * 699,
                     "prompts": [{"label": "地面", "text": "floor"}]}
        complete = {"description": "The visible floor follows the confirmed reference boundary.",
                    "prompts": [{"label": "地面", "text": "floor"}]}
        with patch.object(workflow, "chat",
                          side_effect=[truncated, complete]) as chat:
            result = workflow.describe(
                {}, image_bytes(), [shape(label="地面")], ["地面"])
        self.assertEqual(result, complete)
        self.assertEqual(chat.call_count, 2)
        self.assertIn("360 characters", chat.call_args_list[0].args[1])
        self.assertIn("达到长度上限", chat.call_args_list[1].args[1])

    def test_description_uses_truth_bound_fallback_if_every_retry_hits_schema_bound(self):
        truncated = {"description": "x" * 699,
                     "prompts": [{"label": "地面", "text": "floor"}]}
        with patch.object(workflow, "chat", return_value=truncated) as chat:
            result = workflow.describe(
                {}, image_bytes(), [shape(label="地面")], ["地面"])
        self.assertEqual(chat.call_count, workflow.MAX_PROTOCOL_RETRIES + 1)
        self.assertEqual(result["prompt_source"],
                         "deterministic_reference_geometry_fallback")
        self.assertEqual(result["description_fallback_reason"],
                         "vlm_protocol_retries_exhausted")
        self.assertTrue(result["description"].endswith("."))
        self.assertLessEqual(len(result["description"]),
                             workflow.MAX_REFERENCE_DESCRIPTION_CHARS)
        self.assertIn("1 地面 instance", result["description"])
        self.assertEqual(result["prompts"],
                         [{"label": "地面", "text": "floor"}])
        self.assertTrue(result["candidate_only"])
        self.assertTrue(result["human_reference_available"])

    def test_reference_description_fallback_rejects_unknown_class_prompt(self):
        truncated = {"description": "x" * 699,
                     "prompts": [{"label": "unknown_widget", "text": "widget"}]}
        with patch.object(workflow, "chat", return_value=truncated):
            with self.assertRaisesRegex(ValueError, "没有受控的 SAM 3 兜底提示"):
                workflow.describe({}, image_bytes(),
                                  [shape(label="unknown_widget")],
                                  ["unknown_widget"])

    def test_visible_label_blind_review_keeps_operator_person_and_viewpoint_non_ego(self):
        labels = ["ego", "other_robot", "person"]
        answer = {"assessments": [
            {"label": label, "evidence": "absent", "confidence": 0.9,
             "reason": "No visible object."} for label in labels], "unknown_candidates": []}
        with patch.object(workflow, "chat", return_value=answer) as chat:
            workflow.discover_visible_labels({}, image_bytes(), labels)
        instruction = chat.call_args.args[1]
        self.assertIn("distinct from any human operator", instruction)
        self.assertIn("a visible human body", instruction)
        self.assertIn("camera viewpoint or camera position alone is never evidence", instruction)

    def test_visible_label_discovery_filters_known_labels_and_routes_unknown_to_review(self):
        answer = {"assessments": [
            {"label": "person", "evidence": "present", "confidence": 0.94,
             "reason": "A standing person is visible."},
            {"label": "pallet", "evidence": "absent", "confidence": 0.91,
             "reason": "No pallet is visible."},
            {"label": "slope", "evidence": "uncertain", "confidence": 0.52,
             "reason": "The distant surface is partly occluded."}],
            "unknown_candidates": [{"candidate_name": "rack", "confidence": 0.72,
                                    "evidence": "A large storage rack is visible."},
                                   {"candidate_name": "person", "confidence": 0.91,
                                    "evidence": "A person is visible."}]}
        with patch.object(workflow, "chat", return_value=answer) as chat:
            result = workflow.discover_visible_labels({}, image_bytes(), ["person", "pallet", "slope"])
        self.assertEqual(result["visible_labels"], ["person"])
        self.assertEqual(result["uncertain_labels"], ["slope"])
        self.assertEqual(result["candidate_labels"], ["person", "slope"])
        self.assertEqual(result["consistency_flags"], [])
        self.assertFalse(result["registry_mutated"])
        self.assertEqual(result["unknown_candidates"][0]["route"], "human_review_required")
        self.assertFalse(result["unknown_candidates"][0]["ontology_mutation_allowed"])
        self.assertEqual(result["registry_conflicts"][0]["matching_known_label"], "person")
        self.assertEqual(result["registry_conflicts"][0]["reason_code"], "known_label_alias_conflict")
        self.assertIn("assess every known label", chat.call_args.args[1])
        discovery_schema = chat.call_args.kwargs["response_schema"]
        keyed = discovery_schema["properties"]["assessments"]
        self.assertEqual(keyed["required"], ["person", "pallet", "slope"])
        self.assertFalse(keyed["additionalProperties"])
        self.assertEqual(keyed["properties"]["person"]
                         ["properties"]["reason"]["maxLength"], 72)

    def test_visible_label_discovery_normalizes_keyed_assessments_in_registry_order(self):
        answer = {"assessments": {
            "pallet": {"evidence": "absent", "confidence": 0.91,
                       "reason": "No pallet is visible."},
            "person": {"evidence": "present", "confidence": 0.94,
                       "reason": "A standing person is visible."}},
            "unknown_candidates": []}
        with patch.object(workflow, "chat", return_value=answer):
            result = workflow.discover_visible_labels({}, image_bytes(),
                                                      ["person", "pallet"])
        self.assertEqual([item["label"] for item in result["assessments"]],
                         ["person", "pallet"])
        self.assertEqual(result["visible_labels"], ["person"])
        self.assertEqual(result["uncertain_labels"], [])

    def test_visible_label_discovery_rejects_incomplete_keyed_assessments(self):
        answer = {"assessments": {
            "person": {"evidence": "present", "confidence": 0.94,
                       "reason": "A standing person is visible."}},
            "unknown_candidates": []}
        with patch.object(workflow, "chat", return_value=answer):
            with self.assertRaisesRegex(ValueError, "逐项判断"):
                workflow.discover_visible_labels({}, image_bytes(),
                                                 ["person", "pallet"])

    def test_complete_json_object_recovers_only_a_closed_assessment_mapping(self):
        raw = ('{"assessments":{"person":{"evidence":"present",'
               '"confidence":0.9,"reason":"visible { body }"}},'
               '"unknown_candidates":[{"candidate_name":"unfinished')
        recovered = workflow._complete_json_object(raw, "assessments")
        self.assertEqual(recovered["person"]["evidence"], "present")
        self.assertIsNone(workflow._complete_json_object(
            '{"assessments":{"person":{"evidence":"present"}',
            "assessments"))

    def test_floor_visibility_adds_auditable_slope_recall_probe(self):
        registry = ["pallet", "floor", "slope", "ego"]
        discovery = {"visible_labels": ["floor", "ego"],
                     "uncertain_labels": [],
                     "candidate_labels": ["floor", "ego"]}
        selected, probes = workflow.candidate_labels_with_recall_probes(
            discovery, registry)
        self.assertEqual(selected, ["floor", "slope", "ego"])
        self.assertEqual(probes, ["slope"])
        self.assertNotIn("slope", discovery["visible_labels"])
        self.assertEqual(discovery["candidate_labels"], ["floor", "ego"])

    def test_slope_recall_probe_neither_expands_without_floor_nor_duplicates(self):
        registry = ["floor", "slope", "ego"]
        selected, probes = workflow.candidate_labels_with_recall_probes({
            "visible_labels": ["ego"], "uncertain_labels": [],
            "candidate_labels": ["ego"]}, registry)
        self.assertEqual((selected, probes), (["ego"], []))
        selected, probes = workflow.candidate_labels_with_recall_probes({
            "visible_labels": ["floor", "slope"], "uncertain_labels": [],
            "candidate_labels": ["floor", "slope"]}, registry)
        self.assertEqual((selected, probes), (["floor", "slope"], []))

    def test_reference_coverage_probes_every_missing_registry_label_without_mutating_discovery(self):
        registry = ["person", "pallet", "floor", "slope"]
        discovery = {"visible_labels": ["person"], "uncertain_labels": ["floor"],
                     "candidate_labels": ["person", "floor"]}
        selected, probes = workflow.candidate_labels_with_recall_probes(
            discovery, registry, probe_missing_selected=True)
        self.assertEqual(selected, registry)
        self.assertEqual(probes, ["slope", "pallet"])
        self.assertEqual(discovery["candidate_labels"], ["person", "floor"])
        self.assertNotIn("pallet", discovery["visible_labels"])

    def test_visible_label_discovery_uses_registry_sized_compact_budget(self):
        labels = ["车辆", "pallet", "person", "floor", "slope", "ego",
                  "other_robot", "cargo_floor"]
        answer = {"assessments": [
            {"label": label, "evidence": "absent", "confidence": 0.9,
             "reason": "No visible instance."} for label in labels],
            "unknown_candidates": []}
        config = {"llm_prompt_profile": "compact"}
        with patch.object(workflow, "chat", return_value=answer) as chat:
            workflow.discover_visible_labels(config, image_bytes(), labels)
        self.assertEqual(chat.call_args.args[0]["llm_max_tokens"], 384)
        self.assertNotIn("llm_max_tokens", config)
        self.assertIn("at most 10 words", chat.call_args.args[1])
        self.assertIn("no indentation or extra whitespace", chat.call_args.args[1])
        self.assertEqual(workflow.compact_discovery_token_budget(config, 8), 384)
        self.assertEqual(workflow.compact_discovery_token_budget(config, 20), 1024)
        self.assertEqual(workflow.compact_discovery_fallback_budget(config, 8), 640)
        self.assertEqual(workflow.compact_discovery_fallback_budget(config, 20), 1024)
        self.assertEqual(workflow.compact_discovery_token_budget(
            {**config, "llm_max_tokens": 704}, 8), 704)

    def test_visible_label_discovery_retries_full_budget_after_short_truncation(self):
        labels = ["车辆", "pallet", "person", "floor", "slope", "ego",
                  "other_robot", "cargo_floor"]
        answer = {"assessments": {
            label: {"evidence": "absent", "confidence": 0.9,
                    "reason": "No visible instance."} for label in labels},
            "unknown_candidates": []}
        truncated = workflow.ModelOutputError(
            "大模型未返回有效 JSON：unterminated string",
            '{"assessments":{', finish_reason="length")
        with patch.object(workflow, "chat", side_effect=[truncated, answer]) as chat:
            result = workflow.discover_visible_labels(
                {"llm_prompt_profile": "compact"}, image_bytes(), labels)
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(chat.call_args_list[0].args[0]["llm_max_tokens"], 384)
        self.assertEqual(chat.call_args_list[1].args[0]["llm_max_tokens"], 640)
        self.assertEqual(result["visible_labels"], [])
        self.assertTrue(result["open_set_scan_complete"])

    def test_visible_label_discovery_does_not_retry_non_truncation_error(self):
        labels = ["pallet", "floor"]
        invalid = workflow.ModelOutputError(
            "大模型未返回有效 JSON", '{"not_assessments":true}',
            finish_reason="stop")
        with patch.object(workflow, "chat", side_effect=invalid) as chat:
            with self.assertRaises(workflow.ModelOutputError):
                workflow.discover_visible_labels(
                    {"llm_prompt_profile": "compact"}, image_bytes(), labels)
        self.assertEqual(chat.call_count, 1)

    def test_visible_label_discovery_routes_contradictory_absent_to_uncertain_candidate(self):
        answer = {"assessments": [
            {"label": "person", "evidence": "absent", "confidence": 0.8,
             "reason": "A person is visible in the image, but the label is not present in the list of known labels."},
            {"label": "pallet", "evidence": "absent", "confidence": 0.9,
             "reason": "No pallet is visible."}], "unknown_candidates": []}
        with patch.object(workflow, "chat", return_value=answer):
            result = workflow.discover_visible_labels({}, image_bytes(), ["person", "pallet"])
        self.assertEqual(result["visible_labels"], [])
        self.assertEqual(result["uncertain_labels"], ["person"])
        self.assertEqual(result["candidate_labels"], ["person"])
        self.assertEqual(result["assessments"][0]["evidence"], "uncertain")
        self.assertEqual(result["consistency_flags"], [{
            "label": "person", "original_evidence": "absent", "normalized_evidence": "uncertain",
            "reason_code": "vlm_visibility_contradiction", "route": "human_review_required",
            "reason": "A person is visible in the image, but the label is not present in the list of known labels."}])

    def test_visible_label_discovery_routes_duplicate_and_missing_registry_labels(self):
        answer = {"assessments": [
            {"label": "person", "evidence": "present", "confidence": 0.9, "reason": "visible"},
            {"label": "person", "evidence": "absent", "confidence": 0.8, "reason": "duplicate"}],
            "unknown_candidates": []}
        with patch.object(workflow, "chat", return_value=answer):
            result = workflow.discover_visible_labels({}, image_bytes(), ["person", "pallet"])
        self.assertEqual(result["uncertain_labels"], ["person", "pallet"])
        self.assertEqual(result["visible_labels"], [])
        self.assertTrue(any(flag["reason_code"] == "vlm_duplicate_registry_assessment"
                            for flag in result["consistency_flags"]))

    def test_visible_label_discovery_downgrades_shared_vehicle_robot_anchor(self):
        labels = ["车辆", "ego", "other_robot"]
        answer = {"assessments": [
            {"label": "车辆", "evidence": "present", "confidence": 0.9,
             "reason": "A robot is visible near [700,450]."},
            {"label": "ego", "evidence": "absent", "confidence": 0.9,
             "reason": "No ego robot is visible."},
            {"label": "other_robot", "evidence": "present", "confidence": 0.9,
             "reason": "Another robot is visible at [700,450]."}], "unknown_candidates": []}
        with patch.object(workflow, "chat", return_value=answer):
            result = workflow.discover_visible_labels({}, image_bytes(), labels)
        self.assertEqual(result["visible_labels"], [])
        self.assertEqual(result["uncertain_labels"], ["车辆", "other_robot"])
        flags = [flag for flag in result["consistency_flags"]
                 if flag["reason_code"] == "vlm_cross_label_entity_conflict"]
        self.assertEqual({flag["label"] for flag in flags}, {"车辆", "other_robot"})
        self.assertEqual(flags[0]["conflicting_labels"], ["车辆", "other_robot"])

    def test_visible_label_discovery_ignores_other_object_and_negated_anchors(self):
        labels = ["车辆", "other_robot"]
        answers = [
            ["A vehicle is visible at [300,450], not the robot at [700,450].",
             "A robot is visible at [700,450]."],
            ["A vehicle at [300,450] is left of the robot at [700,450].",
             "A robot is visible at [700,450]."],
        ]
        for reasons in answers:
            with self.subTest(reasons=reasons):
                answer = {"assessments": [
                    {"label": label, "evidence": "present", "confidence": 0.9, "reason": reason}
                    for label, reason in zip(labels, reasons)], "unknown_candidates": []}
                with patch.object(workflow, "chat", return_value=answer):
                    result = workflow.discover_visible_labels({}, image_bytes(), labels)
                self.assertEqual(result["visible_labels"], labels)
                self.assertFalse(any(flag["reason_code"] == "vlm_cross_label_entity_conflict"
                                     for flag in result["consistency_flags"]))

    def test_visible_label_discovery_keeps_distinct_vehicle_robot_anchors(self):
        labels = ["车辆", "other_robot"]
        answer = {"assessments": [
            {"label": "车辆", "evidence": "present", "confidence": 0.9,
             "reason": "A vehicle is visible at [300,450]."},
            {"label": "other_robot", "evidence": "present", "confidence": 0.9,
             "reason": "A robot is visible at [700,450]."}], "unknown_candidates": []}
        with patch.object(workflow, "chat", return_value=answer):
            result = workflow.discover_visible_labels({}, image_bytes(), labels)
        self.assertEqual(result["visible_labels"], labels)
        self.assertFalse(any(flag["reason_code"] == "vlm_cross_label_entity_conflict"
                             for flag in result["consistency_flags"]))

    def test_visible_label_discovery_downgrades_conventional_forklift_robot_claim(self):
        labels = ["车辆", "other_robot"]
        answer = {"assessments": {
            "车辆": {"evidence": "present", "confidence": 0.95,
                     "reason": "A parked forklift body is visible."},
            "other_robot": {"evidence": "present", "confidence": 0.95,
                            "reason": "A visible forklift is distinct from ego."}},
            "unknown_candidates": []}
        with patch.object(workflow, "chat", return_value=answer):
            result = workflow.discover_visible_labels({}, image_bytes(), labels)
        self.assertEqual(result["visible_labels"], ["车辆"])
        self.assertEqual(result["uncertain_labels"], ["other_robot"])
        self.assertEqual(result["candidate_labels"], labels)
        flag = next(flag for flag in result["consistency_flags"]
                    if flag["reason_code"]
                    == "other_robot_conventional_vehicle_conflict")
        self.assertEqual(flag["route"], "human_review_required")
        self.assertEqual(flag["original_reason"],
                         "A visible forklift is distinct from ego.")

    def test_visible_label_discovery_keeps_autonomous_robot_hardware_claim(self):
        answer = {"assessments": {
            "other_robot": {"evidence": "present", "confidence": 0.92,
                            "reason": "Autonomous forklift AGV chassis is visible."}},
            "unknown_candidates": []}
        with patch.object(workflow, "chat", return_value=answer):
            result = workflow.discover_visible_labels(
                {}, image_bytes(), ["other_robot"])
        self.assertEqual(result["visible_labels"], ["other_robot"])
        self.assertEqual(result["uncertain_labels"], [])
        self.assertFalse(any(
            flag["reason_code"] == "other_robot_conventional_vehicle_conflict"
            for flag in result["consistency_flags"]))

    def test_visible_label_discovery_routes_english_vehicle_alias_to_registry_conflict(self):
        answer = {"assessments": [{"label": "车辆", "evidence": "present",
            "confidence": 0.96, "reason": "A vehicle is visible."}],
            "unknown_candidates": [{"candidate_name": "vehicle", "confidence": 0.95,
                                    "evidence": "A vehicle is visible."}]}
        with patch.object(workflow, "chat", return_value=answer):
            result = workflow.discover_visible_labels({}, image_bytes(), ["车辆"])
        self.assertEqual(result["unknown_candidates"], [])
        self.assertEqual(result["registry_conflicts"], [{
            "candidate_name": "vehicle", "confidence": 0.95, "evidence": "A vehicle is visible.",
            "route": "human_review_required", "ontology_mutation_allowed": False,
            "matching_known_label": "车辆", "reason_code": "known_label_alias_conflict"}])
        self.assertFalse(result["registry_mutated"])

    def test_description_only_uses_reference_truth_label_and_short_prompt(self):
        answer = {"description": "地面边界，不包括障碍物。", "prompts": [
            {"label": "person", "text": "A warehouse person in clothing"},
            {"label": "pallet", "text": "wooden warehouse pallet"},
            {"label": "地面", "text": "The visible warehouse floor surface around pallets and people"}]}
        with patch.object(workflow, "chat", return_value=answer) as chat:
            result = workflow.describe({}, image_bytes(), [shape(label="地面")],
                                       ["person", "pallet", "地面"])
        self.assertEqual(result["prompts"], [{"label": "地面", "text": "floor"}])
        self.assertEqual(result["description"], "地面边界，不包括障碍物。")
        instruction = chat.call_args.args[1]
        self.assertIn('允许类别：["地面"]', instruction)
        self.assertNotIn('允许类别：["person"', instruction)

    def test_description_grounds_rules_in_confirmed_reference_geometry(self):
        reference = shape(label="地面")
        reference["source"] = "automatic"
        reference["parts"][0]["holes"] = [[[5, 5], [8, 5], [8, 8], [5, 8]]]
        with patch.object(workflow, "chat", return_value={"description": "参考地面区域。",
                "prompts": [{"label": "地面", "text": "floor"}]}) as chat:
            workflow.describe({}, image_bytes(), [reference], ["地面"])
        instruction = chat.call_args.args[1]
        for principle in ("参考帧真值优先", "先比对，再描述", "必须有标注证据",
                          "不擅自改变区域归属", "谨慎解释未标注区域", "提炼可迁移规则",
                          "输出前检查一致性", "分别生成完整描述和 SAM 3 关键词"):
            self.assertIn(principle, instruction)
        self.assertIn("地面框线归属规则", instruction)
        self.assertIn("无论颜色", instruction)
        self.assertIn("当前可见且没有被物体遮挡", instruction)
        self.assertIn("无论最初来自人工标注还是自动标注", instruction)
        self.assertIn("不得预设所有标线都包含或排除", instruction)
        self.assertIn("证据不足时省略未经证实的规则", instruction)
        self.assertIn("参考帧未出现的独立物体默认按非地面语义处理", instruction)
        self.assertIn("黄色地面标线纳入地面、将机器人排除在地面之外", instruction)
        self.assertIn(json.dumps([reference], ensure_ascii=False), instruction)

    def test_review_has_original_reference_and_geometry_on_every_round(self):
        marked = io.BytesIO()
        Image.new("RGB", (40, 40), "magenta").save(marked, "JPEG")
        reference = {"image": image_bytes(), "marked_image": marked.getvalue(),
                     "annotations": [shape(label="a")]}
        decisions = [{"status": "continue", "operations": [{"operation": "reclassify",
            "shape_ids": [1], "label": "b", "reason": "reference class"}]},
            {"status": "done", "operations": []}]
        with patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "chat", side_effect=decisions) as chat:
            result = workflow.run({}, image_bytes(), {"description": "reference rule",
                "prompts": [{"label": "a", "text": "target"}]}, ["a", "b"],
                reference=reference)
        self.assertTrue(result["agent"]["converged"])
        self.assertEqual(chat.call_count, 2)
        for call in chat.call_args_list:
            self.assertEqual(call.kwargs["extra_images"][2:],
                             (reference["image"], reference["marked_image"]))
            self.assertIn(json.dumps(workflow.reference_geometry(reference["annotations"]), ensure_ascii=False), call.args[1])
            self.assertIn("如与参考真值证据冲突，以参考真值为准", call.args[1])
            self.assertIn("不得直接照搬参考帧坐标", call.args[1])

    def test_review_prompt_allows_local_sam_points_and_opencv(self):
        with patch.object(workflow, "chat", return_value={"status": "done", "operations": []}) as chat:
            workflow.review({}, image_bytes(), image_bytes(),
                {"description": "floor only", "prompts": [{"label": "a", "text": "floor"}]},
                [], ["a"], [])
        instruction = chat.call_args.args[1]
        for principle in ("复查阶段允许使用 SAM 3", "只能对现有候选的局部问题使用 sam_points",
                          "禁止使用 sam_text 对整幅图像重新分割", "保留已正确标注的区域",
                          "不得仅凭参考帧与当前帧坐标或面积不同",
                          "修改范围限于需要修复的区域", "无法可靠判断或准确定位时",
                          "SAM 3 点提示结果必须由下一轮 GPT"):
            self.assertIn(principle, instruction)
        self.assertNotIn("- sam_text:", instruction)
        self.assertIn("- sam_points:", instruction)
        self.assertIn("不修改候选并返回 done", instruction)
        self.assertIn("发现漏标时使用局部 sam_points 或 add_region 修复，不能返回 done", instruction)
        self.assertIn("线条颜色不能作为扣除依据", instruction)
        self.assertNotIn('"status":"continue|done|needs_review"', instruction)

    def test_review_requires_assessment_for_each_detected_gap(self):
        donut = {"id": 1, "label": "地面", "points": [[2, 2], [30, 2], [30, 30], [2, 30]],
                 "holes": [[[10, 10], [20, 10], [20, 20], [10, 20]]],
                 "parts": [{"points": [[2, 2], [30, 2], [30, 30], [2, 30]],
                            "holes": [[[10, 10], [20, 10], [20, 20], [10, 20]]]}]}
        holes = workflow.candidate_holes([donut], ["地面"], 40, 40)
        with patch.object(workflow, "chat", return_value={"status": "done", "operations": []}) as chat:
            with self.assertRaisesRegex(ValueError, "逐一判断"):
                workflow.review({}, image_bytes(), image_bytes(),
                    {"description": "floor", "prompts": []}, [], ["地面"], [], holes=holes)
        self.assertEqual(chat.call_count, 3)
        self.assertIn("上一份输出未通过 Agent 协议校验", chat.call_args.args[1])

    def test_review_retries_protocol_error_and_returns_corrected_decision(self):
        donut = {"id": 1, "label": "地面", "points": [[2, 2], [30, 2], [30, 30], [2, 30]],
                 "holes": [[[10, 10], [20, 10], [20, 20], [10, 20]]],
                 "parts": [{"points": [[2, 2], [30, 2], [30, 30], [2, 30]],
                            "holes": [[[10, 10], [20, 10], [20, 20], [10, 20]]]}]}
        holes = workflow.candidate_holes([donut], ["地面"], 40, 40)
        invalid = {"status": "done", "hole_assessment": [{"hole_id": "#5",
            "semantic": "ground", "reason": "floor"}], "operations": []}
        valid = {"status": "done", "hole_assessment": [{"hole_id": hole["id"],
            "semantic": "non_ground", "reason": "visible object"} for hole in holes],
            "operations": []}
        with patch.object(workflow, "chat", side_effect=[invalid, valid]) as chat:
            decision = workflow.review({}, image_bytes(), image_bytes(),
                {"description": "floor", "prompts": []}, [], ["地面"], [], holes=holes)
        self.assertEqual(decision, valid)
        self.assertEqual(chat.call_count, 2)
        self.assertIn("GPT 空洞判断必须覆盖全部 hole_id", chat.call_args.args[1])
        self.assertIn('"hole_id": "#5"', chat.call_args.args[1])

    def test_degenerate_polygon_retry_feedback_requires_distinct_non_collinear_vertices(self):
        invalid = {"status": "continue", "reason": "Repair boundary", "operations": [{
            "operation": "add_region", "shape_ids": [1],
            "polygon": [[3, 4], [3, 4], [3, 4]], "reason": "Repair local edge"}]}
        valid = {"status": "done", "reason": "No reliable local repair remains", "operations": []}
        with patch.object(workflow, "chat", side_effect=[invalid, valid]) as chat:
            decision = workflow.review({}, image_bytes(), image_bytes(),
                {"description": "a", "prompts": []}, [shape()], ["a"], [])
        self.assertEqual(decision, valid)
        self.assertEqual(chat.call_count, 2)
        correction = chat.call_args.args[1]
        self.assertIn("至少 3 个彼此不同且不共线", correction)
        self.assertIn("不得重复同一坐标", correction)
        self.assertIn("返回 done 和空 operations", correction)

    def test_omitted_hole_is_preserved_and_routed_to_human_review(self):
        holes = [
            {"id": "1:hole:1:1:2:2", "shape_id": 1},
            {"id": "1:hole:3:3:4:4", "shape_id": 1},
        ]
        decision = workflow._validate_review_result({
            "status": "done",
            "reason": "Only one region was clear",
            "hole_assessment": [{"hole_id": holes[0]["id"],
                                 "semantic": "non_ground", "reason": "Visible obstacle"}],
            "operations": [],
        }, holes, [shape()], ["a"], 40, 40)
        repaired = {item["hole_id"]: item for item in decision["hole_assessment"]}
        self.assertEqual(repaired[holes[1]["id"]]["semantic"], "non_ground")
        self.assertEqual(decision["protocol_repairs"], [{
            "reason_code": "vlm_hole_assessment_omitted",
            "hole_id": holes[1]["id"], "route": "human_review_required"}])

    def test_exact_duplicate_hole_assessment_is_dropped_before_safe_omission_repair(self):
        holes = [
            {"id": "1:hole:1:1:2:2", "shape_id": 1},
            {"id": "1:hole:3:3:4:4", "shape_id": 1},
        ]
        repeated = {"hole_id": holes[0]["id"], "semantic": "non_ground",
                    "reason": "Visible obstacle"}
        decision = workflow._validate_review_result({
            "status": "done",
            "reason": "One assessment was repeated",
            "hole_assessment": [repeated, copy.deepcopy(repeated)],
            "operations": [],
        }, holes, [shape()], ["a"], 40, 40)
        self.assertEqual(decision["protocol_normalizations"], [{
            "reason_code": "duplicate_hole_assessment_dropped",
            "hole_id": holes[0]["id"],
        }])
        self.assertEqual(decision["protocol_repairs"], [{
            "reason_code": "vlm_hole_assessment_omitted",
            "hole_id": holes[1]["id"], "route": "human_review_required"}])
        self.assertEqual({item["hole_id"] for item in decision["hole_assessment"]},
                         {item["id"] for item in holes})

    def test_conflicting_duplicate_hole_assessment_remains_protocol_error(self):
        hole = {"id": "1:hole:1:1:2:2", "shape_id": 1}
        assessment = [
            {"hole_id": hole["id"], "semantic": "ground", "reason": "Floor"},
            {"hole_id": hole["id"], "semantic": "non_ground", "reason": "Obstacle"},
        ]
        with self.assertRaisesRegex(ValueError, "恰好一次"):
            workflow._validate_review_result({
                "status": "done", "reason": "Conflicting duplicate",
                "hole_assessment": assessment, "operations": [],
            }, [hole], [shape()], ["a"], 40, 40)

    def test_omitted_hole_cannot_be_filled_and_unknown_ids_still_fail(self):
        holes = [
            {"id": "1:hole:1:1:2:2", "shape_id": 1},
            {"id": "1:hole:3:3:4:4", "shape_id": 1},
        ]
        fill = {"operation": "fill_gap", "shape_ids": [1],
                "hole_id": holes[1]["id"],
                "polygon": [[3, 3], [4, 3], [4, 4]], "reason": "Fill floor"}
        partial = {"status": "continue", "reason": "Partial answer",
                   "hole_assessment": [{"hole_id": holes[0]["id"],
                                        "semantic": "non_ground", "reason": "Obstacle"}],
                   "operations": [fill]}
        with self.assertRaisesRegex(ValueError, "只能修补判定为 ground"):
            workflow._validate_review_result(partial, holes, [shape()], ["a"], 40, 40)
        unknown = copy.deepcopy(partial)
        unknown["operations"] = []
        unknown["hole_assessment"].append(
            {"hole_id": "unknown", "semantic": "non_ground", "reason": "Unknown"})
        with self.assertRaisesRegex(ValueError, "无效 ID"):
            workflow._validate_review_result(unknown, holes, [shape()], ["a"], 40, 40)

    def test_omitted_hole_repair_finishes_as_review_required_without_discarding_candidates(self):
        decision = {"status": "done", "reason": "One region was omitted",
                    "hole_assessment": [], "operations": [],
                    "protocol_repairs": [{
                        "reason_code": "vlm_hole_assessment_omitted",
                        "hole_id": "1:hole:3:3:4:4",
                        "route": "human_review_required"}]}
        with patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "review", return_value=decision):
            result = workflow.run({}, image_bytes(), {"description": "class a",
                "prompts": [{"label": "a", "text": "square"}]}, ["a"], 2)
        self.assertEqual(result["agent"]["stop_reason"], "review_required")
        self.assertFalse(result["agent"]["converged"])
        self.assertEqual(len(result["shapes"]), 1)
        self.assertEqual(result["review_queue"][0]["reason_code"],
                         "vlm_hole_assessment_omitted")

    def test_empty_continue_is_normalized_to_done_but_unfilled_ground_is_not(self):
        empty = workflow._validate_review_result(
            {"status": "continue", "reason": "No edit remains", "operations": []},
            [], [shape()], ["a"], 40, 40)
        self.assertEqual(empty["status"], "done")
        self.assertEqual(empty["protocol_normalizations"][0]["reason_code"],
                         "empty_continue_normalized_to_done")
        hole = {"id": "1:hole:1:1:2:2", "shape_id": 1}
        pending = workflow._validate_review_result({
            "status": "continue", "reason": "Ground remains",
            "hole_assessment": [{"hole_id": hole["id"], "semantic": "ground",
                                 "reason": "Visible ground"}],
            "operations": [],
        }, [hole], [shape()], ["a"], 40, 40)
        self.assertEqual(pending["status"], "continue")
        self.assertNotIn("protocol_normalizations", pending)

    def test_explicit_ground_assessments_can_fill_every_exact_gap_once(self):
        floor = {
            "id": 1, "label": "floor",
            "points": [[2, 2], [36, 2], [36, 36], [2, 36]],
            "holes": [
                [[8, 8], [13, 8], [13, 13], [8, 13]],
                [[22, 20], [28, 20], [28, 27], [22, 27]],
            ],
            "parts": [{
                "points": [[2, 2], [36, 2], [36, 36], [2, 36]],
                "holes": [
                    [[8, 8], [13, 8], [13, 13], [8, 13]],
                    [[22, 20], [28, 20], [28, 27], [22, 27]],
                ],
            }],
        }
        holes = [item for item in workflow.candidate_holes([floor], ["floor"], 40, 40)
                 if item["kind"] == "hole"]
        explicit = {
            "operation": "fill_gap", "shape_ids": [1],
            "hole_id": holes[0]["id"],
            "polygon": [[7, 7], [14, 7], [14, 14], [7, 14]],
            "reason": "Visible floor",
        }
        decision = workflow._validate_review_result({
            "status": "continue", "reason": "Both gaps are visible floor",
            "hole_assessment": [{"hole_id": hole["id"], "semantic": "ground",
                                 "reason": "Visible floor in target and reference"}
                                for hole in holes],
            "operations": [explicit],
        }, holes, [floor], ["floor"], 40, 40,
            synthesize_ground_fills=True)
        self.assertEqual(len(decision["operations"]), 2)
        self.assertEqual({item["hole_id"] for item in decision["operations"]},
                         {item["id"] for item in holes})
        normalizations = [item for item in decision["protocol_normalizations"]
                          if item["reason_code"] == "explicit_ground_fill_synthesized"]
        self.assertEqual([item["hole_id"] for item in normalizations], [holes[1]["id"]])
        updated, applied, queue = apply_operations(
            [floor], decision["operations"], ["floor"], 40, 40, gap_regions=holes)
        self.assertEqual(queue, [])
        self.assertEqual(len(applied), 2)
        self.assertEqual([item for item in workflow.candidate_holes(
            updated, ["floor"], 40, 40) if item["kind"] == "hole"], [])

    def test_compact_reference_review_synthesizes_ground_fill_without_model_polygon(self):
        floor = {
            "id": 1, "label": "floor",
            "points": [[2, 2], [36, 2], [36, 36], [2, 36]],
            "holes": [[[8, 8], [14, 8], [14, 14], [8, 14]]],
            "parts": [{"points": [[2, 2], [36, 2], [36, 36], [2, 36]],
                       "holes": [[[8, 8], [14, 8], [14, 14], [8, 14]]]}],
        }
        holes = [item for item in workflow.candidate_holes(
            [floor], ["floor"], 40, 40) if item["kind"] == "hole"]
        reply = {
            "status": "continue", "reason": "Visible reference-backed ground",
            "hole_assessment": [{"hole_id": holes[0]["id"],
                                 "semantic": "ground",
                                 "reason": "Visible floor in target and reference"}],
            "operations": [],
        }
        reference = {"image": image_bytes(), "marked_image": image_bytes(),
                     "annotations": [floor]}
        with patch.object(workflow, "chat", return_value=reply) as call:
            decision = workflow.review(
                {"llm_prompt_profile": "compact"}, image_bytes(), image_bytes(),
                {"description": "Reference floor", "prompts": [{
                    "label": "floor", "text": "floor"}]},
                [floor], ["floor"], [], reference=reference,
                hole_image=image_bytes(), holes=holes)
        self.assertEqual([item["operation"] for item in decision["operations"]],
                         ["fill_gap"])
        variants = call.call_args.kwargs["response_schema"]["properties"][
            "operations"]["items"]["anyOf"]
        self.assertNotIn("fill_gap", {
            name for variant in variants
            for name in variant["properties"]["operation"]["enum"]})
        self.assertIn("不要输出fill_gap", call.call_args.args[1])

    def test_ground_fill_synthesis_never_promotes_omitted_or_non_ground_gaps(self):
        holes = [
            {"id": "1:hole:4:4:8:8", "shape_id": 1,
             "bbox": [4, 4, 8, 8]},
            {"id": "1:hole:12:12:16:16", "shape_id": 1,
             "bbox": [12, 12, 16, 16]},
        ]
        decision = workflow._validate_review_result({
            "status": "continue", "reason": "Only the first gap is verified floor",
            "hole_assessment": [{"hole_id": holes[0]["id"], "semantic": "ground",
                                 "reason": "Visible ground"}],
            "operations": [],
        }, holes, [shape()], ["a"], 40, 40,
            synthesize_ground_fills=True)
        self.assertEqual([item["hole_id"] for item in decision["operations"]],
                         [holes[0]["id"]])
        self.assertEqual(decision["protocol_repairs"], [{
            "reason_code": "vlm_hole_assessment_omitted",
            "hole_id": holes[1]["id"], "route": "human_review_required"}])
        repaired = {item["hole_id"]: item for item in decision["hole_assessment"]}
        self.assertEqual(repaired[holes[1]["id"]]["semantic"], "non_ground")

    def test_ground_fill_synthesis_requires_reference_and_explicit_opt_in(self):
        hole = {"id": "1:hole:4:4:8:8", "shape_id": 1,
                "bbox": [4, 4, 8, 8]}
        payload = {
            "status": "continue", "reason": "Visible ground",
            "hole_assessment": [{"hole_id": hole["id"], "semantic": "ground",
                                 "reason": "Visible ground"}],
            "operations": [],
        }
        default = workflow._validate_review_result(
            payload, [hole], [shape()], ["a"], 40, 40)
        unreferenced = workflow._validate_review_result(
            payload, [hole], [shape()], ["a"], 40, 40,
            reference_provided=False, synthesize_ground_fills=True)
        self.assertEqual(default["operations"], [])
        self.assertEqual(unreferenced["operations"], [])

    def test_no_op_reclassify_is_dropped_without_a_model_retry(self):
        decision = workflow._validate_review_result({
            "status": "continue",
            "reason": "Keep the existing class",
            "operations": [{"operation": "reclassify", "shape_ids": [1],
                            "label": "a", "reason": "Candidate already matches"}],
        }, [], [shape()], ["a"], 40, 40, reference_provided=False)
        self.assertEqual(decision["status"], "done")
        self.assertEqual(decision["operations"], [])
        self.assertEqual(
            [item["reason_code"] for item in decision["protocol_normalizations"]],
            ["no_op_reclassify_dropped", "empty_continue_normalized_to_done"])

    def test_remove_without_negative_evidence_is_dropped(self):
        decision = workflow._validate_review_result({
            "status": "continue",
            "reason": "Candidate is correct",
            "operations": [{"operation": "remove", "shape_ids": [1],
                            "reason": "Person instance matches reference geometry"}],
        }, [], [shape()], ["a"], 40, 40)
        self.assertEqual(decision["status"], "done")
        self.assertEqual(decision["operations"], [])
        self.assertEqual(
            [item["reason_code"] for item in decision["protocol_normalizations"]],
            ["remove_without_negative_evidence_dropped",
             "empty_continue_normalized_to_done"])

    def test_remove_with_explicit_source_class_negative_is_retained(self):
        decision = workflow._validate_review_result({
            "status": "continue",
            "reason": "Reject the false pallet candidate",
            "operations": [{"operation": "remove", "shape_ids": [1],
                            "reason": "Candidate is not a pallet; it is a cardboard box."}],
        }, [], [shape(label="pallet")], ["pallet"], 40, 40)
        self.assertEqual(len(decision["operations"]), 1)
        self.assertEqual(decision["operations"][0]["operation"], "remove")
        self.assertNotIn("protocol_normalizations", decision)

    def test_review_retries_unknown_operation_and_missing_tool_fields(self):
        operation = {"operation": "add_region", "shape_ids": [1],
                     "polygon": [[16, 2], [25, 2], [25, 18], [16, 18]], "reason": "repair floor"}
        corrected = {"status": "continue", "operations": [operation]}
        cases = [
            ({"status": "continue", "operations": [{**operation, "operation": "fill_hole"}]}, "操作名无效"),
            ({"status": "continue", "operations": [{key: value for key, value in operation.items()
                                                    if key != "shape_ids"}]}, "shape_ids"),
            ({"status": "continue", "operations": [{key: value for key, value in operation.items()
                                                    if key != "polygon"}]}, "polygon"),
            ({"status": "continue", "operations": [{**operation, "shape_ids": [99]}]}, "shape_ids"),
            ({"status": "done", "operations": None}, "operations"),
            ({"status": "done", "operations": {}}, "operations"),
        ]
        for invalid, error in cases:
            with self.subTest(error=error), \
                 patch.object(workflow, "sam_text", return_value=[sam_result()]), \
                 patch.object(workflow, "chat", side_effect=[
                     invalid, corrected, {"status": "done", "operations": []}]) as chat:
                result = workflow.run({}, image_bytes(), {"description": "floor",
                    "prompts": [{"label": "a", "text": "floor"}]}, ["a"], 2)
                self.assertTrue(result["agent"]["converged"])
                self.assertEqual(result["agent"]["rounds"], 2)
                self.assertEqual(len(result["applied"]), 1)
                self.assertIn(error, chat.call_args_list[1].args[1])
                self.assertIn(json.dumps(invalid, ensure_ascii=False), chat.call_args_list[1].args[1])
                self.assertEqual(chat.call_args_list[0].kwargs["extra_images"],
                                 chat.call_args_list[1].kwargs["extra_images"])
                mask = _rasterize(result["shapes"][0], 40, 40)
                self.assertEqual(mask[5, 23], 1)

    def test_review_retries_invalid_json_but_not_transport_errors(self):
        def response(content):
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())
        with patch.object(workflow.urllib.request, "urlopen", side_effect=[
                response("invalid json"), response('{"status":"done","operations":[]}')]) as request:
            result = workflow.review({"llm_url": "http://example.test", "llm_model": "test"},
                image_bytes(), image_bytes(), {"description": "a", "prompts": []}, [], ["a"], [])
        self.assertEqual(result["status"], "done")
        self.assertEqual(request.call_count, 2)
        sent = json.loads(request.call_args.args[0].data)["messages"][0]["content"][0]["text"]
        self.assertIn("invalid json", sent)
        self.assertIn("未返回有效 JSON", sent)
        for error in (ValueError("大模型请求超时"), workflow.RunCancelled("用户已停止")):
            with self.subTest(error=error), patch.object(workflow, "chat", side_effect=error) as chat:
                with self.assertRaises(type(error)):
                    workflow.review({}, image_bytes(), image_bytes(),
                        {"description": "a", "prompts": []}, [], ["a"], [])
                self.assertEqual(chat.call_count, 1)
                self.assertIsNone(workflow._review_call.get())

    def test_exhausted_protocol_retries_preserve_candidates_and_trace_rejections(self):
        events = []
        invalid = {"status": "done", "operations": None}
        with workflow.trace_calls(events.append), \
             patch.object(workflow, "sam_text", return_value=[sam_result()]) as sam, \
             patch.object(workflow, "_chat", return_value=invalid) as chat:
            result = workflow.run({}, image_bytes(), {"description": "a",
                "prompts": [{"label": "a", "text": "square"}]}, ["a"])
        self.assertEqual(chat.call_count, 3)
        self.assertEqual(sam.call_count, 1)
        self.assertFalse(result["agent"]["converged"])
        self.assertEqual(result["agent"]["rounds"], 0)
        self.assertEqual(result["agent"]["stop_reason"], "invalid_model_output")
        self.assertIn("协议校验重试 2 次后仍失败", result["review_queue"][0]["error"])
        self.assertEqual(result["review_queue"][0]["reason_code"],
                         "agent_invalid_model_output")
        self.assertEqual(result["review_queue"][0]["route"], "human_review_required")
        self.assertEqual(result["shapes"][0]["parts"], shape()["parts"])
        rejected = [event for event in events if event["status"] == "rejected"]
        self.assertEqual([event["review_attempt"] for event in rejected], [1, 2, 3])
        self.assertEqual([event["retrying"] for event in rejected], [True, True, False])
        self.assertEqual(len({event["id"] for event in rejected}), 3)
        self.assertEqual(rejected[-1]["output"]["parsed"], invalid)
        self.assertIn("operations", rejected[-1]["error"])
        self.assertIsNone(workflow._review_call.get())

    def test_repeated_degenerate_polygons_stop_after_one_correction(self):
        events = []
        invalid = {"status": "continue", "reason": "Repair boundary", "operations": [{
            "operation": "add_region", "shape_ids": [1],
            "polygon": [[3, 4], [3, 4], [3, 4]], "reason": "Repair local edge"}]}
        with workflow.trace_calls(events.append), \
             patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "_chat", return_value=invalid) as chat:
            result = workflow.run({}, image_bytes(), {"description": "a",
                "prompts": [{"label": "a", "text": "square"}]}, ["a"])
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(result["agent"]["stop_reason"], "invalid_model_output")
        self.assertEqual(result["agent"]["rounds"], 0)
        self.assertEqual(result["shapes"][0]["parts"], shape()["parts"])
        self.assertIn("连续 2 次返回相同的退化多边形", result["review_queue"][0]["error"])
        rejected = [event for event in events if event["status"] == "rejected"]
        self.assertEqual([event["review_attempt"] for event in rejected], [1, 2])
        self.assertEqual([event["retrying"] for event in rejected], [True, False])
        self.assertEqual(rejected[-1]["protocol_retry_stop_reason"],
                         "repeated_degenerate_polygon")
        self.assertIn("唯一一次协议纠错机会", chat.call_args_list[1].args[1])

    def test_degenerate_polygon_with_other_protocol_error_keeps_full_retry_budget(self):
        events = []
        invalid = {"status": "continue", "operations": [{
            "operation": "add_region", "shape_ids": [999],
            "polygon": [[3, 4], [3, 4], [3, 4]]}]}
        with workflow.trace_calls(events.append), \
             patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "_chat", return_value=invalid) as chat:
            result = workflow.run({}, image_bytes(), {"description": "a",
                "prompts": [{"label": "a", "text": "square"}]}, ["a"])
        self.assertEqual(chat.call_count, 3)
        self.assertEqual(result["agent"]["stop_reason"], "invalid_model_output")
        rejected = [event for event in events if event["status"] == "rejected"]
        self.assertEqual([event["retrying"] for event in rejected], [True, True, False])
        self.assertNotIn("protocol_retry_stop_reason", rejected[-1])

    def test_missing_sam_assessment_is_sent_back_for_correction(self):
        with patch.object(workflow, "chat", side_effect=[
                {"status": "done", "operations": []},
                {"status": "done", "operations": [],
                 "sam_assessment": {"improved": False, "reason": "lost correct region"}}]) as chat:
            result = workflow.review({}, image_bytes(), image_bytes(),
                {"description": "a", "prompts": []}, [shape()], ["a"], [],
                sam_comparison={"shape_id": 1, "before_image": image_bytes()})
        self.assertFalse(result["sam_assessment"]["improved"])
        self.assertIn("未评估 SAM 3 点提示结果", chat.call_args.args[1])

    def test_legacy_review_request_ends_without_human_review(self):
        original = shape()
        result = self.run_agent([{"status": "needs_review", "reason": "uncertain",
            "operations": [{"operation": "review", "shape_ids": [1]}]}])
        self.assertEqual(result["agent"]["stop_reason"], "done")
        self.assertTrue(result["agent"]["converged"])
        self.assertEqual(result["review_queue"], [])
        self.assertEqual(result["shapes"][0]["parts"], original["parts"])
        self.assertEqual(result["agent"]["audit"][0]["decision"]["operations"], [])

    def test_legacy_review_with_valid_edit_applies_edit(self):
        result = self.run_agent([{"status": "needs_review", "operations": [
            {"operation": "review", "shape_ids": [1]},
            {"operation": "reclassify", "shape_ids": [1], "label": "b", "reason": "reference class"}]},
            {"status": "done", "operations": []}])
        self.assertEqual(result["agent"]["stop_reason"], "done")
        self.assertEqual(result["shapes"][0]["label"], "b")
        self.assertEqual(result["review_queue"], [])

    def test_multiple_opencv_rounds_change_only_requested_regions_without_sam_reentry(self):
        decisions = [
            {"status": "continue", "operations": [{"operation": "add_region", "shape_ids": [1],
                "polygon": [[16, 2], [25, 2], [25, 18], [16, 18]], "reason": "extend right edge"}]},
            {"status": "continue", "operations": [{"operation": "erase_region", "shape_ids": [1],
                "polygon": [[2, 2], [5, 2], [5, 18], [2, 18]], "reason": "trim left edge"}]},
            {"status": "done", "operations": []}]
        with patch.object(workflow, "sam_request", return_value={"results": [sam_result()]}) as sam, \
             patch.object(workflow, "chat", side_effect=decisions):
            result = workflow.run({}, image_bytes(),
                {"description": "class a", "prompts": [{"label": "a", "text": "square"}]}, ["a"], 3)
        sam.assert_called_once_with({}, "/segment_text", image_bytes(), prompt="square")
        self.assertTrue(result["agent"]["converged"])
        self.assertEqual(result["agent"]["rounds"], 3)
        self.assertEqual(result["candidates_added"], 1)
        expected = _rasterize(shape(), 40, 40)
        expected[2:19, 16:26] = 1
        expected[2:19, 2:6] = 0
        self.assertEqual(_rasterize(result["shapes"][0], 40, 40).tolist(), expected.tolist())

    def test_multi_prompt_generation_reuses_one_image_request(self):
        response = {"prompts": {
            "a": {"results": [sam_result()]},
            "b": {"results": []},
        }, "diagnostics": {"image_encoded_once": True, "total_seconds": 0.2}}
        agent = workflow.AnnotationAgent({}, image_bytes(), {
            "description": "classes a and b",
            "prompts": [{"label": "a", "text": "square"},
                        {"label": "b", "text": "circle"}],
        }, ["a", "b"], 1)
        with patch.object(workflow, "sam_request", return_value=response) as request:
            agent.generate()
        request.assert_called_once_with({}, "/segment_text_multi", image_bytes(),
                                        prompts={"a": "square", "b": "circle"})
        self.assertEqual([item["label"] for item in agent.shapes], ["a"])
        self.assertEqual(agent.sam_generation["mode"], "multi_prompt")
        self.assertTrue(agent.sam_generation["diagnostics"]["image_encoded_once"])

    def test_multi_prompt_adapts_legacy_list_contract_and_caches_protocol(self):
        prompts = {"a": "square", "b": "circle"}
        config = {"sam3_url": "http://legacy-sam:18766"}
        legacy = {"items": [
            {"prompt": "square", "results": [sam_result()]},
            {"prompt": "circle", "results": []},
        ], "diagnostics": {"feature_reuse": True, "image_feature_passes": 1}}
        with workflow._sam_multi_protocol_lock:
            workflow._sam_multi_protocols.pop(config["sam3_url"], None)
        with patch.object(workflow, "sam_request", side_effect=[
                ValueError("SAM 3 请求失败：image (base64) and prompts are required"),
                legacy, legacy]) as request:
            first, first_diagnostics = workflow.sam_text_multi(
                image_bytes(), prompts, config)
            second, second_diagnostics = workflow.sam_text_multi(
                image_bytes(), prompts, config)
        self.assertEqual(request.call_args_list[0].kwargs["prompts"], prompts)
        self.assertEqual(request.call_args_list[1].kwargs["prompts"], ["square", "circle"])
        self.assertEqual(request.call_args_list[2].kwargs["prompts"], ["square", "circle"])
        self.assertEqual(set(first), set(prompts))
        self.assertEqual(set(second), set(prompts))
        self.assertTrue(first_diagnostics["legacy_protocol"])
        self.assertTrue(first_diagnostics["image_encoded_once"])
        self.assertTrue(second_diagnostics["image_encoded_once"])

    def test_multi_prompt_does_not_cache_malformed_legacy_response(self):
        prompts = {"a": "square", "b": "circle"}
        config = {"sam3_url": "http://malformed-legacy-sam:18766"}
        with workflow._sam_multi_protocol_lock:
            workflow._sam_multi_protocols.pop(config["sam3_url"], None)
        with patch.object(workflow, "sam_request", side_effect=[
                ValueError("SAM 3 请求失败：image (base64) and prompts are required"),
                {"items": [{"prompt": "square", "results": []}]}]):
            with self.assertRaisesRegex(ValueError, "缺少完整类别结果"):
                workflow.sam_text_multi(image_bytes(), prompts, config)
        with workflow._sam_multi_protocol_lock:
            self.assertNotIn(config["sam3_url"], workflow._sam_multi_protocols)

    def test_multi_prompt_does_not_hide_unrelated_server_error(self):
        config = {"sam3_url": "http://unavailable-sam:18766"}
        with patch.object(workflow, "sam_request",
                          side_effect=ValueError("SAM 3 请求失败：service busy")) as request:
            with self.assertRaisesRegex(ValueError, "service busy"):
                workflow.sam_text_multi(
                    image_bytes(), {"a": "square", "b": "circle"}, config)
        request.assert_called_once()

    def test_multi_prompt_generation_falls_back_to_sequential_requests(self):
        agent = workflow.AnnotationAgent({}, image_bytes(), {
            "description": "classes a and b",
            "prompts": [{"label": "a", "text": "square"},
                        {"label": "b", "text": "circle"}],
        }, ["a", "b"], 1)
        with patch.object(workflow, "sam_request", side_effect=[
                ValueError("SAM 3 请求失败：Not Found"),
                {"results": [sam_result()]}, {"results": []}]) as request:
            agent.generate()
        self.assertEqual([call.args[1] for call in request.call_args_list],
                         ["/segment_text_multi", "/segment_text", "/segment_text"])
        self.assertEqual(agent.sam_generation["mode"], "sequential_fallback")
        self.assertIn("Not Found", agent.sam_generation["error"])
        self.assertEqual([item["label"] for item in agent.shapes], ["a"])

    def test_multi_prompt_prefetch_accepts_ready_and_busy_contracts(self):
        ready = {"prompts": {
            "a": {"results": [sam_result()]},
            "b": {"results": []},
        }, "diagnostics": {"response_cache": {"hit": False}}}
        with patch.object(workflow, "sam_request", return_value=ready) as request:
            result = workflow.sam_text_multi_prefetch(
                image_bytes(), {"a": "square", "b": "circle"}, {})
        request.assert_called_once_with(
            {}, "/prefetch_text_multi", image_bytes(),
            prompts={"a": "square", "b": "circle"})
        self.assertTrue(result["prefetched"])
        with patch.object(workflow, "sam_request", return_value={
                "prefetched": False, "code": "sam_busy"}):
            busy = workflow.sam_text_multi_prefetch(
                image_bytes(), {"a": "square", "b": "circle"}, {})
        self.assertFalse(busy["prefetched"])
        self.assertEqual(busy["code"], "sam_busy")

    def test_multi_prompt_prefetch_rejects_incomplete_success(self):
        with patch.object(workflow, "sam_request", return_value={
                "prompts": {"a": {"results": []}}}):
            with self.assertRaisesRegex(ValueError, "缺少类别结果"):
                workflow.sam_text_multi_prefetch(
                    image_bytes(), {"a": "square", "b": "circle"}, {})

    def test_single_prompt_prefetch_accepts_ready_and_busy_contracts(self):
        ready = {"results": [sam_result()],
                 "diagnostics": {"response_cache": {"hit": False}}}
        with patch.object(workflow, "sam_request", return_value=ready) as request:
            result = workflow.sam_text_prefetch(image_bytes(), "square", {})
        request.assert_called_once_with(
            {}, "/prefetch_text", image_bytes(), prompt="square")
        self.assertTrue(result["prefetched"])
        with patch.object(workflow, "sam_request", return_value={
                "prefetched": False, "code": "sam_busy"}):
            busy = workflow.sam_text_prefetch(image_bytes(), "square", {})
        self.assertEqual(busy["code"], "sam_busy")

    def test_generate_only_returns_guarded_candidates_without_agent_call(self):
        with patch.object(workflow, "sam_request", return_value={"results": [sam_result()]}) as sam, \
             patch.object(workflow, "chat") as chat:
            result = workflow.generate_only({}, image_bytes(),
                {"description": "class a", "prompts": [{"label": "a", "text": "square"}]},
                ["a"], max_rounds=3)
        sam.assert_called_once()
        chat.assert_not_called()
        self.assertEqual(result["agent"]["stop_reason"], "review_deferred")
        self.assertEqual(result["agent"]["rounds"], 0)
        self.assertEqual(result["agent"]["max_rounds"], 3)
        self.assertEqual(result["shapes"][0]["label"], "a")

    def test_review_existing_runs_agent_without_regenerating_sam(self):
        existing = shape(label="a")
        with patch.object(workflow, "sam_request") as sam, \
             patch.object(workflow, "chat", return_value={
                 "status": "done", "reason": "candidate is supported",
                 "hole_assessment": [], "operations": []}) as chat:
            result = workflow.review_existing({}, image_bytes(),
                {"description": "class a", "prompts": [{"label": "a", "text": "square"}]},
                ["a"], [existing], max_rounds=2,
                candidate_filters=[{"reason_code": "kept"}],
                sam_generation={"mode": "multi_prompt"})
        sam.assert_not_called()
        chat.assert_called_once()
        self.assertEqual(result["agent"]["stop_reason"], "done")
        self.assertEqual(result["agent"]["rounds"], 1)
        self.assertEqual(result["shapes"], [existing])
        self.assertEqual(result["agent"]["candidate_filters"], [{"reason_code": "kept"}])
        self.assertEqual(result["agent"]["sam_generation"], {"mode": "multi_prompt"})

    def test_keyword_from_gpt_is_preserved_when_short_and_relevant(self):
        answer = {"description": "标注参考帧里的可见车辆实例。", "prompts": [
            {"label": "车辆", "text": "forklift"},
            {"label": "person", "text": "worker"}]}
        result = workflow.restrict_description(answer, ["车辆"])
        self.assertEqual(result["description"], answer["description"])
        self.assertEqual(result["prompts"], [{"label": "车辆", "text": "forklift"}])

    def test_pallet_prompt_cannot_drift_to_stacked_cargo(self):
        for text in ("", " stacked boxes ", "wooden pallet", "pallet"):
            with self.subTest(text=text):
                answer = {"description": "只标注可见托盘结构，不把堆叠箱体当作托盘。",
                          "prompts": [{"label": "pallet", "text": text}]}
                result = workflow.restrict_description(answer, ["pallet"])
                self.assertEqual(result["description"], answer["description"])
                self.assertEqual(result["prompts"], [{"label": "pallet", "text": "pallet"}])
                self.assertEqual(answer["prompts"][0]["text"], text)

    def test_known_workbench_labels_have_stable_sam_prompt_fallbacks(self):
        labels = ["车辆", "pallet", "person", "floor", "slope", "ego", "other_robot", "cargo_floor"]
        result = workflow.restrict_description(
            {"description": "Review every configured class.", "prompts": []}, labels)
        self.assertEqual(result["prompts"], [
            {"label": "车辆", "text": "vehicle"},
            {"label": "pallet", "text": "pallet"},
            {"label": "person", "text": "person"},
            {"label": "floor", "text": "floor"},
            {"label": "slope", "text": "traversable surface"},
            {"label": "ego", "text": "ego robot"},
            {"label": "other_robot", "text": "robot"},
            {"label": "cargo_floor", "text": "robot cargo platform"},
        ])

    def test_slope_prompt_uses_evidence_backed_traversable_query(self):
        result = workflow.restrict_description(
            {"description": "Review the ramp.",
             "prompts": [{"label": "slope", "text": "yellow ramp"}]}, ["slope"])
        self.assertEqual(result["prompts"], [
            {"label": "slope", "text": "traversable surface"}])

    def test_slope_geometry_filter_keeps_broad_plane_and_routes_tall_strip(self):
        def polygon(candidate_id, label, points):
            part = {"points": points, "holes": []}
            return {"id": candidate_id, "label": label, "points": points,
                    "holes": [], "parts": [part], "score": 0.8}
        broad = polygon(1, "slope", [[1, 2], [31, 2], [31, 10], [1, 10]])
        tall = polygon(2, "slope", [[2, 1], [10, 1], [10, 31], [2, 31]])
        floor = polygon(3, "floor", [[1, 1], [5, 1], [5, 30], [1, 30]])
        retained, suppressed = workflow.filter_slope_candidate_geometry(
            [broad, tall, floor], 40)
        self.assertEqual([item["id"] for item in retained], [1, 3])
        self.assertEqual([item["id"] for item in suppressed], [2])
        self.assertEqual(suppressed[0]["reason_code"],
                         "slope_candidate_not_broad_planar")
        self.assertEqual(suppressed[0]["route"], "human_review_required")

    def test_slope_geometry_filter_routes_thin_horizontal_strip(self):
        points = [[1, 2], [35, 2], [35, 3], [1, 3]]
        part = {"points": points, "holes": []}
        candidate = {"id": 1, "label": "slope", "points": points,
                     "holes": [], "parts": [part], "score": 0.8}
        retained, suppressed = workflow.filter_slope_candidate_geometry(
            [candidate], 40)
        self.assertEqual(retained, [])
        self.assertEqual([item["id"] for item in suppressed], [1])
        self.assertEqual(suppressed[0]["reason_code"],
                         "slope_candidate_below_minimum_height")
        self.assertEqual(suppressed[0]["minimum_height"], 4)

    def test_slope_geometry_filter_rejects_compact_floor_patch_without_losing_wide_ramp(self):
        def polygon(candidate_id, width, height):
            points = [[0, 0], [width - 1, 0],
                      [width - 1, height - 1], [0, height - 1]]
            part = {"points": points, "holes": []}
            return {"id": candidate_id, "label": "slope", "points": points,
                    "holes": [], "parts": [part], "score": 0.8}
        compact_floor_patch = polygon(1, 169, 100)
        narrowest_preserved_ramp_proxy = polygon(2, 195, 100)
        retained, suppressed = workflow.filter_slope_candidate_geometry(
            [compact_floor_patch, narrowest_preserved_ramp_proxy], 480)
        self.assertEqual([item["id"] for item in retained], [2])
        self.assertEqual([item["id"] for item in suppressed], [1])
        self.assertEqual(suppressed[0]["reason_code"],
                         "slope_candidate_not_broad_planar")
        self.assertEqual(suppressed[0]["minimum_width_height_ratio"], 1.75)

    def test_slope_generation_uses_overlap_crops_only_after_full_frame_rejection(self):
        def result(points, score=0.8):
            return {"parts": [{"points": points, "holes": []}], "score": score}
        tall = result([[2, 1], [10, 1], [10, 31], [2, 31]])
        crop_wide = result([[1, 1], [20, 1], [20, 8], [1, 8]], 0.7)
        crop_tall = result([[1, 1], [8, 1], [8, 20], [1, 20]])
        agent = workflow.AnnotationAgent({}, image_bytes(), {
            "description": "slope",
            "prompts": [{"label": "slope", "text": "traversable surface"}],
        }, ["slope"], 1)
        with patch.object(workflow, "sam_text", side_effect=[
                [tall], [crop_wide], [crop_tall], [crop_tall], [crop_tall]]) as segment:
            agent.generate()
        self.assertEqual(segment.call_count, 5)
        self.assertEqual(len(agent.shapes), 1)
        self.assertEqual(agent.shapes[0]["source"], "sam_text_overlap_crop_fallback")
        self.assertEqual(agent.shapes[0]["crop_index"], 0)
        self.assertEqual(len(agent.candidate_filters), 4)

    def test_slope_pallet_conflict_filter_preserves_pallet_and_disjoint_slope(self):
        pallet = shape(1, "pallet", 2, 18)
        overlapping = shape(2, "slope", 3, 17)
        disjoint = shape(3, "slope", 22, 38)
        retained, suppressed = workflow.filter_slope_pallet_conflicts(
            [pallet, overlapping, disjoint], 40, 40)
        self.assertEqual([item["id"] for item in retained], [1, 3])
        self.assertEqual([item["id"] for item in suppressed], [2])
        self.assertEqual(suppressed[0]["pallet_id"], 1)
        self.assertEqual(suppressed[0]["reason_code"],
                         "slope_candidate_pallet_conflict")

    def test_slope_pallet_conflict_uses_occlusion_envelope_when_masks_are_disjoint(self):
        pallet_points = [[0, 0], [39, 0], [39, 5], [5, 5], [5, 39], [0, 39]]
        pallet_part = {"points": pallet_points, "holes": []}
        pallet = {"id": 1, "label": "pallet", "points": pallet_points,
                  "holes": [], "parts": [pallet_part], "score": 0.9}
        slope_points = [[20, 20], [30, 20], [30, 30], [20, 30]]
        slope_part = {"points": slope_points, "holes": []}
        slope = {"id": 2, "label": "slope", "points": slope_points,
                 "holes": [], "parts": [slope_part], "score": 0.8}
        retained, suppressed = workflow.filter_slope_pallet_conflicts(
            [pallet, slope], 40, 40)
        self.assertEqual([item["id"] for item in retained], [1])
        self.assertEqual([item["id"] for item in suppressed], [2])
        self.assertEqual(suppressed[0]["slope_covered_by_pallet"], 0.0)
        self.assertEqual(suppressed[0]["slope_envelope_covered_by_pallet"], 1.0)

    def test_slope_floor_conflict_routes_only_nearly_covered_slope(self):
        floor = shape(1, "floor", 1, 30)
        covered = shape(2, "slope", 3, 20)
        disjoint = shape(3, "slope", 31, 38)
        retained, suppressed = workflow.filter_slope_floor_conflicts(
            [floor, covered, disjoint], 40, 40)
        self.assertEqual([item["id"] for item in retained], [1, 3])
        self.assertEqual([item["id"] for item in suppressed], [2])
        self.assertEqual(suppressed[0]["floor_id"], 1)
        self.assertEqual(suppressed[0]["reason_code"],
                         "slope_candidate_floor_conflict")
        self.assertEqual(suppressed[0]["route"], "human_review_required")
        self.assertGreaterEqual(suppressed[0]["slope_covered_by_floor"], 0.9)

    def test_slope_reference_ego_guard_requires_same_resolution_reviewed_ego(self):
        ego = shape(1, "ego", 2, 24)
        covered = shape(2, "slope", 3, 20)
        disjoint = shape(3, "slope", 27, 38)
        reference = {"image": image_bytes(), "annotations": [ego]}
        retained, suppressed = workflow.filter_slope_reference_ego_conflicts(
            [ego, covered, disjoint], reference, 40, 40)
        self.assertEqual([item["id"] for item in retained], [1, 3])
        self.assertEqual([item["id"] for item in suppressed], [2])
        self.assertEqual(suppressed[0]["reason_code"],
                         "slope_candidate_reference_ego_conflict")
        self.assertEqual(suppressed[0]["route"], "human_review_required")
        self.assertGreaterEqual(suppressed[0]["slope_covered_by_reference_ego"], 0.8)
        smaller = io.BytesIO()
        Image.new("RGB", (20, 20), "white").save(smaller, "JPEG")
        unchanged, warnings = workflow.filter_slope_reference_ego_conflicts(
            [covered], {"image": smaller.getvalue(), "annotations": [ego]}, 40, 40)
        self.assertEqual(unchanged, [covered])
        self.assertEqual(warnings, [])

    def test_slope_current_ego_guard_requires_strong_nearly_covering_candidate(self):
        ego = shape(1, "ego", 2, 24)
        ego["score"] = 0.95
        covered = shape(2, "slope", 3, 20)
        disjoint = shape(3, "slope", 27, 38)
        retained, suppressed = workflow.filter_slope_current_ego_conflicts(
            [ego, covered, disjoint], 40, 40)
        self.assertEqual([item["id"] for item in retained], [1, 3])
        self.assertEqual([item["id"] for item in suppressed], [2])
        self.assertEqual(suppressed[0]["reason_code"],
                         "slope_candidate_current_ego_conflict")
        self.assertEqual(suppressed[0]["route"], "human_review_required")
        self.assertFalse(suppressed[0]["human_truth"])
        self.assertFalse(suppressed[0]["approved_for_training"])
        self.assertGreaterEqual(suppressed[0]["slope_covered_by_current_ego"], 0.8)

        weak_ego = copy.deepcopy(ego)
        weak_ego["score"] = 0.84
        unchanged, warnings = workflow.filter_slope_current_ego_conflicts(
            [weak_ego, covered], 40, 40)
        self.assertEqual(unchanged, [weak_ego, covered])
        self.assertEqual(warnings, [])

    def test_robot_and_person_prompts_use_exact_canonical_queries(self):
        labels = ["ego", "other_robot", "person"]
        expected = [
            {"label": "ego", "text": "ego robot"},
            {"label": "other_robot", "text": "robot"},
            {"label": "person", "text": "person"}]
        for prompts in ([], [
                {"label": "ego", "text": "camera viewpoint"},
                {"label": "other_robot", "text": "human operator"},
                {"label": "person", "text": "operator"}]):
            with self.subTest(prompts=prompts):
                result = workflow.restrict_description(
                    {"description": "区分相机载体、其他机器人与人员。", "prompts": prompts}, labels)
                self.assertEqual(result["prompts"], expected)

    def test_other_robot_semantic_fallback_is_bounded_and_candidate_only(self):
        candidates = []
        for candidate_id, score in enumerate((0.4, 0.9, 0.8, 0.7), 1):
            candidate = shape(candidate_id, "other_robot")
            candidate["score"] = score
            candidates.append(candidate)
        baseline = copy.deepcopy(candidates)
        retained, filtered = workflow.select_other_robot_semantic_fallback_candidates(
            candidates, maximum=3)
        self.assertEqual([item["id"] for item in retained], [2, 3, 4])
        self.assertEqual([item["candidate_id"] for item in filtered], [1])
        self.assertTrue(all(item["candidate_only"] for item in retained))
        self.assertTrue(all(item["human_truth"] is False for item in retained))
        self.assertTrue(all(item["approved_for_training"] is False
                            for item in retained))
        self.assertEqual([item["points"] for item in candidates],
                         [item["points"] for item in baseline])

    def test_other_robot_semantic_fallback_runs_only_after_visible_empty_baseline(self):
        agent = workflow.AnnotationAgent(
            {"other_robot_semantic_fallback_visible": True}, image_bytes(),
            {"prompts": [{"label": "other_robot", "text": "robot"}]},
            ["other_robot"], 1)
        with patch.object(workflow, "sam_text", side_effect=[[], [sam_result()]]) as call:
            agent.generate()
        self.assertEqual([item.args[1] for item in call.call_args_list],
                         ["robot", "autonomous pallet truck"])
        self.assertEqual(len(agent.shapes), 1)
        self.assertEqual(agent.shapes[0]["source"],
                         "sam_text_other_robot_semantic_fallback")
        self.assertEqual(
            agent.sam_generation["semantic_fallbacks"][0]["trigger"],
            "visible_label_baseline_empty")

    def test_other_robot_semantic_fallback_stays_off_without_visible_evidence(self):
        agent = workflow.AnnotationAgent(
            {}, image_bytes(),
            {"prompts": [{"label": "other_robot", "text": "robot"}]},
            ["other_robot"], 1)
        with patch.object(workflow, "sam_text", return_value=[]) as call:
            agent.generate()
        self.assertEqual(call.call_count, 1)
        self.assertEqual(agent.shapes, [])
        self.assertNotIn("semantic_fallbacks", agent.sam_generation)

    def test_other_robot_semantic_fallback_does_not_duplicate_baseline(self):
        agent = workflow.AnnotationAgent(
            {"other_robot_semantic_fallback_visible": True}, image_bytes(),
            {"prompts": [{"label": "other_robot", "text": "robot"}]},
            ["other_robot"], 1)
        with patch.object(workflow, "sam_text", return_value=[sam_result()]) as call:
            agent.generate()
        self.assertEqual(call.call_count, 1)
        self.assertEqual(len(agent.shapes), 1)
        self.assertNotIn("semantic_fallbacks", agent.sam_generation)

    def test_cargo_floor_prompt_is_overridden_even_when_gpt_returns_ambiguous_text(self):
        result = workflow.restrict_description(
            {"description": "cargo area", "prompts": [{"label": "cargo_floor", "text": "cargo floor"}]},
            ["cargo_floor"])
        self.assertEqual(result["prompts"], [
            {"label": "cargo_floor", "text": "robot cargo platform"}])

    def test_person_prompt_uses_evidence_backed_canonical_query(self):
        result = workflow.restrict_description(
            {"description": "people", "prompts": [
                {"label": "person", "text": "worker operator worker worker"}]},
            ["person"])
        self.assertEqual(result["prompts"], [{"label": "person", "text": "person"}])

    def test_candidate_filter_removes_same_label_and_floor_conflicts(self):
        floor = shape(label="floor", shape_id=1)
        floor_copy = shape(label="floor", shape_id=2)
        cargo_copy = shape(label="cargo_floor", shape_id=3)
        kept, filtered = workflow.suppress_near_duplicate_candidates(
            [floor, floor_copy, cargo_copy], 40, 40)
        self.assertEqual([(item["id"], item["label"]) for item in kept], [(1, "floor")])
        self.assertEqual({item["reason"] for item in filtered},
                         {"near_duplicate_same_label", "overlaps_warehouse_floor"})

    def test_unreferenced_draft_preserves_floor_deck_ambiguity_and_same_class_dedup(self):
        originals = [shape(1, "floor"), shape(2, "floor"), shape(3, "cargo_floor")]
        baseline = copy.deepcopy(originals)
        kept, filtered = workflow.suppress_near_duplicate_candidates(
            originals, 40, 40, preserve_class_conflicts=True)
        self.assertEqual({item["id"] for item in kept}, {1, 3})
        self.assertEqual([item["reason"] for item in filtered], ["near_duplicate_same_label"])
        self.assertEqual(originals, baseline)

    def test_class_overlap_flags_only_conflicting_high_iou_pairs(self):
        cases = [
            ([shape(1, "ego"), shape(2, "other_robot")], 1),
            ([shape(1, "车辆"), shape(2, "ego")], 1),
            ([shape(1, "floor"), shape(2, "cargo_floor")], 1),
            ([shape(1, "person"), shape(2, "ego")], 0),
            ([shape(1, "ego"), shape(2, "ego")], 0),
            ([shape(1, "ego"), shape(2, "other_robot", 23, 35)], 0),
        ]
        for shapes, expected in cases:
            with self.subTest(labels=[item["label"] for item in shapes], expected=expected):
                baseline = copy.deepcopy(shapes)
                conflicts = workflow.candidate_class_overlap_conflicts(shapes, 40, 40)
                self.assertEqual(len(conflicts), expected)
                self.assertEqual(shapes, baseline)
                if expected:
                    self.assertEqual(conflicts[0]["shape_ids"], [1, 2])
                    self.assertEqual(conflicts[0]["route"], "human_review_required")
                    self.assertEqual(conflicts[0]["iou"], 1.0)

    def test_overlap_conflict_ids_follow_merged_candidate_ids(self):
        generated = [shape(1, "ego"), shape(2, "other_robot")]
        combined, id_map = server.merge_inference_candidates(
            [shape(1, "pallet")], generated, ["ego", "other_robot"], return_id_map=True)
        result = {"agent": {}, "applied": [], "review_queue":
                  workflow.candidate_class_overlap_conflicts(generated, 40, 40)}
        server.bind_merged_candidate_references(result, id_map)
        by_id = {item["id"]: item["label"] for item in combined}
        conflict = result["review_queue"][0]
        self.assertEqual([by_id[item] for item in conflict["shape_ids"]],
                         ["ego", "other_robot"])
        self.assertEqual(conflict["original_shape_ids"], [1, 2])

    def test_unreferenced_pipeline_retains_conflict_despite_model_done(self):
        multi = ({"floor": {"results": [sam_result()]},
                  "cargo_floor": {"results": [sam_result()]}},
                 {"image_encoded_once": True})
        with patch.object(workflow, "sam_text_multi", return_value=multi), \
             patch.object(workflow, "chat", return_value={"status": "done", "operations": []}):
            result = workflow.run({}, image_bytes(), {"description": "draft", "prompts": [
                {"label": "floor", "text": "floor"},
                {"label": "cargo_floor", "text": "robot cargo platform"}]},
                ["floor", "cargo_floor"])
        self.assertEqual({item["label"] for item in result["shapes"]}, {"floor", "cargo_floor"})
        self.assertEqual(len(result["review_queue"]), 1)
        self.assertEqual(result["review_queue"][0]["reason_code"], "candidate_cross_class_overlap")
        self.assertEqual(result["review_queue"][0]["shape_ids"], [1, 2])

    def test_unknown_non_ascii_label_still_requires_an_explicit_mapping(self):
        with self.assertRaisesRegex(ValueError, "缺少简短英文"):
            workflow.restrict_description(
                {"description": "unknown class", "prompts": []}, ["未知类别"])

    def test_gpt_reviews_initial_sam_candidates_against_detailed_rule(self):
        prompts = []
        def inspect(config, instruction, image, extra_images=()):
            prompts.append((instruction, len(extra_images)))
            return {"status": "done", "reason": "边界符合参考帧规范", "operations": []}
        with patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "chat", side_effect=inspect):
            result = workflow.run({}, image_bytes(), {"description": "只标注可见目标边界，不补全遮挡部分。",
                "prompts": [{"label": "a", "text": "target"}]}, ["a"], 2)
        self.assertTrue(result["agent"]["converged"])
        self.assertEqual(len(prompts), 1)
        self.assertIn("只标注可见目标边界", prompts[0][0])
        self.assertIn('"id": 1', prompts[0][0])
        self.assertEqual(prompts[0][1], 2)

    def test_unreferenced_batch_skips_reference_dependent_floor_gap_analysis(self):
        decision = {"status": "done", "reason": "No supported edit.",
                    "hole_assessment": [], "operations": []}
        with patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "candidate_holes") as holes, \
             patch.object(workflow, "chat", return_value=decision):
            result = workflow.run({}, image_bytes(),
                {"description": "generic floor", "prompts": [{"label": "floor", "text": "floor"}]},
                ["floor"], 2, reference=None)
        holes.assert_not_called()
        self.assertTrue(result["agent"]["converged"])
        self.assertEqual(result["review_queue"], [])

    def test_model_trace_records_empty_sam_response_and_bad_gpt_json(self):
        events = []
        with workflow.trace_calls(events.append), \
             patch.object(workflow.urllib.request, "urlopen", side_effect=[
                 io.BytesIO(b'{"results": [], "diagnostics": {"model_mask_count": 0}}'),
                 io.BytesIO(b'{"choices":[{"message":{"content":"invalid json"}}]}')]):
            self.assertEqual(workflow.sam_text(image_bytes(), "person", {}), [])
            with self.assertRaises(ValueError):
                workflow.chat({"llm_url": "http://example.test", "llm_model": "test",
                               "llm_key": "secret-token"}, "inspect", image_bytes())
        self.assertEqual([event["status"] for event in events],
                         ["running", "succeeded", "running", "failed"])
        self.assertEqual(events[1]["output"]["returned_count"], 0)
        self.assertEqual(events[1]["input"]["image"]["width"], 40)
        self.assertEqual(events[3]["output"]["raw_text"], "invalid json")
        self.assertNotIn("secret-token", json.dumps(events))
        self.assertNotIn("base64", json.dumps(events))

    def test_compact_review_uses_bounded_generation_budget(self):
        captured = {}
        def respond(request, timeout):
            captured.update(json.loads(request.data))
            return io.BytesIO(b'{"choices":[{"message":{"content":"{\\"status\\":\\"done\\"}"}}]}')
        token = workflow._review_call.set({"attempt": 1, "feedback": None})
        try:
            with patch.object(workflow.urllib.request, "urlopen", side_effect=respond):
                workflow._chat({"llm_url": "http://example.test", "llm_model": "test",
                                "llm_prompt_profile": "compact"}, "inspect", image_bytes())
        finally:
            workflow._review_call.reset(token)
        self.assertEqual(captured["max_tokens"], 768)
        self.assertEqual(workflow.compact_review_token_budget(
            {"llm_prompt_profile": "compact"}, 0), 768)
        self.assertEqual(workflow.compact_review_token_budget(
            {"llm_prompt_profile": "compact"}, 4), 1536)
        self.assertEqual(workflow.compact_review_token_budget(
            {"llm_prompt_profile": "compact", "llm_max_tokens": 1024}, 4), 1024)

    def test_chat_preserves_length_finish_reason_on_invalid_json(self):
        response = io.BytesIO(
            b'{"choices":[{"finish_reason":"length","message":{"content":'
            b'"{\\"assessments\\":{\\"floor\\":{\\"evidence\\":\\"visible\\""}}]}')
        with patch.object(workflow.urllib.request, "urlopen", return_value=response):
            with self.assertRaises(workflow.ModelOutputError) as caught:
                workflow._chat(
                    {"llm_url": "http://example.test", "llm_model": "test"},
                    "inspect", image_bytes())
        self.assertEqual(caught.exception.finish_reason, "length")
        self.assertIn('"assessments"', caught.exception.raw_text)

        from source.agent.protocol_schema import review_schema
        schema = review_schema(["floor"], [shape(label="floor")], [{
            "id": "1:hole:1:1:2:2", "area": 4, "bbox": [1, 1, 2, 2]}])
        properties = schema["properties"]
        self.assertEqual(properties["reason"]["maxLength"], 120)
        self.assertEqual(properties["hole_assessment"]["items"]["properties"]
                         ["reason"]["maxLength"], 100)
        operation_variants = properties["operations"]["items"]["anyOf"]
        self.assertTrue(all(item["properties"]["reason"]["maxLength"] == 120
                            for item in operation_variants))
        synthesized = review_schema(
            ["floor"], [shape(label="floor")], [{
                "id": "1:hole:1:1:2:2", "area": 4,
                "bbox": [1, 1, 2, 2]}], synthesize_ground_fills=True)
        synthesized_operations = synthesized["properties"]["operations"]["items"]["anyOf"]
        self.assertNotIn("fill_gap", {
            name for variant in synthesized_operations
            for name in variant["properties"]["operation"]["enum"]})

    def test_review_gap_selection_excludes_open_image_border_artifacts(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        holes = [
            {"id": "edge-right", "kind": "open_gap", "area": 1000,
             "bbox": [25, 4, 29, 14], "mask": mask},
            {"id": "edge-bottom", "kind": "open_gap", "area": 900,
             "bbox": [4, 18, 18, 19], "mask": mask},
            {"id": "internal-gap", "kind": "open_gap", "area": 400,
             "bbox": [5, 5, 12, 12], "mask": mask},
            {"id": "enclosed-hole", "kind": "hole", "area": 300,
             "bbox": [0, 0, 3, 3], "mask": mask},
        ]
        selected = workflow.select_review_gaps(holes, maximum=4)
        self.assertEqual({item["id"] for item in selected},
                         {"internal-gap", "enclosed-hole"})

    def test_review_gap_selection_suppresses_resolution_scaled_tail_noise(self):
        mask = np.zeros((1536, 1920), dtype=np.uint8)
        holes = [
            {"id": "tiny", "kind": "hole", "area": 50,
             "bbox": [10, 10, 20, 20], "mask": mask},
            {"id": "meaningful", "kind": "hole", "area": 70,
             "bbox": [30, 30, 40, 40], "mask": mask},
        ]
        selected = workflow.select_review_gaps(holes)
        self.assertEqual([item["id"] for item in selected], ["meaningful"])
        self.assertEqual([item["id"] for item in workflow.select_review_gaps(
            holes, minimum_area=40)], ["tiny", "meaningful"])

    def test_truncated_compact_review_doubles_budget_once_for_protocol_repair(self):
        budgets, events = [], []
        def fake_chat(config, instruction, image, extra_images=(), raw=None,
                      response_schema=None):
            budgets.append(int(config.get("llm_max_tokens") or 768))
            if len(budgets) == 1:
                raw.update(text='{"status":"done","operations":[{"reason":"unfinished',
                           finish_reason="length")
                raise workflow.ModelOutputError(
                    "大模型未返回有效 JSON：Unterminated string", raw["text"])
            raw.update(text='{"status":"done","operations":[]}', finish_reason="stop")
            return {"status": "done", "operations": []}
        with workflow.trace_calls(events.append), patch.object(workflow, "_chat", side_effect=fake_chat):
            result = workflow.review({"llm_prompt_profile": "compact"}, image_bytes(),
                image_bytes(), {"description": "a", "prompts": []}, [], ["a"], [])
        self.assertEqual(result["status"], "done")
        self.assertEqual(budgets, [768, 1536])
        gpt_events = [event for event in events if event["service"] == "GPT"
                      and event["status"] in ("rejected", "succeeded")]
        self.assertEqual(gpt_events[0]["output"]["finish_reason"], "length")
        self.assertEqual(gpt_events[-1]["input"]["max_tokens"], 1536)

    def test_local_vlm_busy_waits_and_retries_without_repeating_sam(self):
        busy = workflow.urllib.error.HTTPError(
            "http://example.test", 503, "Service Unavailable", None,
            io.BytesIO(b'{"error":{"message":"Local VLM is busy"}}'))
        success = io.BytesIO(
            b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"status\\":\\"done\\"}"}}]}')
        raw, events = {}, []
        token = workflow._review_call.set({"record": {
            "id": "busy-call", "service": "GPT", "status": "running"}})
        try:
            with workflow.trace_calls(events.append), \
                 patch.object(workflow.urllib.request, "urlopen", side_effect=[busy, success]) as request, \
                 patch.object(workflow, "_wait_cancellable") as wait:
                result = workflow._chat({"llm_url": "http://example.test",
                    "llm_model": "test"}, "inspect", image_bytes(), raw=raw)
        finally:
            workflow._review_call.reset(token)
        self.assertEqual(result, {"status": "done"})
        self.assertEqual(request.call_count, 2)
        wait.assert_called_once_with(1)
        self.assertEqual(raw["busy_retry_count"], 1)
        self.assertEqual(events[0]["status"], "waiting")
        self.assertEqual(events[0]["retry_in_seconds"], 1)
        self.assertNotIn("error", events[0])
        self.assertIn("已排队", events[0]["waiting_reason"])

    def test_small_image_fast_route_is_bounded_and_does_not_mutate_config(self):
        workflow._reset_fast_route_circuits()
        self.addCleanup(workflow._reset_fast_route_circuits)
        config = {"llm_url": "http://primary/v1", "llm_model": "primary",
                  "llm_key": "primary-key", "llm_fast_url": "http://fast/v1",
                  "llm_fast_model": "fast", "llm_fast_key": "fast-key",
                  "llm_fast_max_pixels": 640 * 512,
                  "llm_fast_max_long_edge": 640}
        original = copy.deepcopy(config)
        selected, route = workflow.select_vlm_route(
            config, sized_image_bytes(640, 512))
        self.assertEqual(route, "small_image_fast")
        self.assertEqual(selected["llm_url"], "http://fast/v1")
        self.assertEqual(selected["llm_model"], "fast")
        self.assertEqual(selected["llm_key"], "fast-key")
        self.assertEqual(selected["llm_busy_retries"], 0)
        self.assertEqual(config, original)

        selected, route = workflow.select_vlm_route(
            config, sized_image_bytes(1920, 1536))
        self.assertIs(selected, config)
        self.assertEqual(route, "primary")
        selected, route = workflow.select_vlm_route(
            config, sized_image_bytes(640, 512),
            extra_images=(sized_image_bytes(1280, 512),))
        self.assertEqual(route, "small_image_fast")
        self.assertEqual(selected["llm_url"], "http://fast/v1")
        selected, route = workflow.select_vlm_route(
            config, sized_image_bytes(640, 512),
            extra_images=(sized_image_bytes(1920, 1536),))
        self.assertIs(selected, config)
        self.assertEqual(route, "primary")

    def test_fast_only_chat_never_falls_back_to_busy_primary_lane(self):
        workflow._reset_fast_route_circuits()
        self.addCleanup(workflow._reset_fast_route_circuits)
        busy = workflow.urllib.error.HTTPError(
            "http://fast/v1/chat/completions", 503, "Service Unavailable", None,
            io.BytesIO(b'{"error":{"message":"Local VLM is busy"}}'))
        config = {"llm_url": "http://primary/v1", "llm_model": "primary",
                  "llm_fast_url": "http://fast/v1", "llm_fast_model": "fast"}
        with patch.object(workflow.urllib.request, "urlopen", side_effect=busy) as request:
            with self.assertRaisesRegex(ValueError, "快速通道和主通道均未返回结果"):
                workflow.chat_fast_only(
                    config, "inspect", sized_image_bytes(64, 64))
        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[0].full_url,
                         "http://fast/v1/chat/completions")

    def test_small_image_fast_busy_falls_back_to_primary_without_waiting(self):
        workflow._reset_fast_route_circuits()
        self.addCleanup(workflow._reset_fast_route_circuits)
        busy = workflow.urllib.error.HTTPError(
            "http://fast/v1/chat/completions", 503, "Service Unavailable", None,
            io.BytesIO(b'{"error":{"message":"Local VLM is busy"}}'))
        success = io.BytesIO(
            b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"status\\":\\"done\\"}"}}]}')
        requests = []
        def respond(request, timeout):
            requests.append((request.full_url, timeout))
            response = [busy, success][len(requests) - 1]
            if isinstance(response, Exception):
                raise response
            return response
        raw = {}
        with patch.object(workflow.urllib.request, "urlopen", side_effect=respond), \
             patch.object(workflow, "_wait_cancellable") as wait:
            result = workflow._chat({
                "llm_url": "http://primary/v1", "llm_model": "primary",
                "llm_fast_url": "http://fast/v1", "llm_fast_model": "fast",
            }, "inspect", image_bytes(), raw=raw)
        self.assertEqual(result, {"status": "done"})
        self.assertEqual(requests, [
            ("http://fast/v1/chat/completions", 60),
            ("http://primary/v1/chat/completions", 1200)])
        wait.assert_not_called()
        self.assertEqual(raw["model_route"], "primary_fallback")
        self.assertEqual(raw["fast_fallback_reason"], "fast_busy")
        self.assertEqual(raw["busy_retry_count"], 0)

    def test_small_image_fast_timeout_falls_back_with_bounded_timeout(self):
        workflow._reset_fast_route_circuits()
        self.addCleanup(workflow._reset_fast_route_circuits)
        success = io.BytesIO(
            b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"status\\":\\"done\\"}"}}]}')
        raw = {}
        with patch.object(workflow.urllib.request, "urlopen",
                          side_effect=[TimeoutError(), success]) as request:
            result = workflow._chat({
                "llm_url": "http://primary/v1", "llm_model": "primary",
                "llm_fast_url": "http://fast/v1", "llm_fast_model": "fast",
                "llm_fast_timeout_seconds": 25,
            }, "inspect", image_bytes(), raw=raw)
        self.assertEqual(result, {"status": "done"})
        self.assertEqual([call.kwargs["timeout"] for call in request.call_args_list],
                         [25, 1200])
        self.assertEqual(raw["model_route"], "primary_fallback")
        self.assertEqual(raw["fast_fallback_reason"], "fast_timeout")

    def test_fast_failure_opens_circuit_and_skips_repeated_failed_attempt(self):
        workflow._reset_fast_route_circuits()
        self.addCleanup(workflow._reset_fast_route_circuits)
        responses = [
            workflow.urllib.error.URLError("offline"),
            io.BytesIO(b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"status\\":\\"done\\"}"}}]}'),
            io.BytesIO(b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"status\\":\\"done\\"}"}}]}'),
        ]
        requests = []
        def respond(request, timeout):
            requests.append(request.full_url)
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        config = {"llm_url": "http://primary/v1", "llm_model": "primary",
                  "llm_fast_url": "http://circuit-fast/v1",
                  "llm_fast_model": "fast", "llm_fast_cooldown_seconds": 60}
        first, second = {}, {}
        with patch.object(workflow.urllib.request, "urlopen", side_effect=respond):
            self.assertEqual(workflow._chat(
                config, "inspect", image_bytes(), raw=first), {"status": "done"})
            self.assertEqual(workflow._chat(
                config, "inspect", image_bytes(), raw=second), {"status": "done"})
        self.assertEqual(requests, [
            "http://circuit-fast/v1/chat/completions",
            "http://primary/v1/chat/completions",
            "http://primary/v1/chat/completions",
        ])
        self.assertEqual(first["model_route"], "primary_fallback")
        self.assertEqual(first["fast_fallback_reason"], "fast_unavailable")
        self.assertEqual(second["model_route"], "primary")
        self.assertEqual(second["fast_fallback_reason"], "fast_circuit_open")

    def test_fast_circuit_expires_and_allows_a_recovery_probe(self):
        workflow._reset_fast_route_circuits()
        self.addCleanup(workflow._reset_fast_route_circuits)
        config = {"llm_url": "http://primary/v1", "llm_model": "primary",
                  "llm_fast_url": "http://recover-fast/v1",
                  "llm_fast_model": "fast", "llm_fast_cooldown_seconds": 60}
        with patch.object(workflow.time, "monotonic", return_value=100):
            workflow._open_fast_route_circuit(config, "fast_unavailable")
            self.assertEqual(workflow._fast_route_circuit_reason(config),
                             "fast_circuit_open")
        success = io.BytesIO(
            b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"status\\":\\"done\\"}"}}]}')
        raw = {}
        with patch.object(workflow.time, "monotonic", return_value=161), \
             patch.object(workflow.urllib.request, "urlopen", return_value=success) as request:
            result = workflow._chat(config, "inspect", image_bytes(), raw=raw)
            self.assertIsNone(workflow._fast_route_circuit_reason(config))
        self.assertEqual(result, {"status": "done"})
        self.assertEqual(request.call_args.args[0].full_url,
                         "http://recover-fast/v1/chat/completions")
        self.assertEqual(raw["model_route"], "small_image_fast")
        self.assertNotIn("fast_fallback_reason", raw)

    def test_large_image_never_contacts_fast_route(self):
        workflow._reset_fast_route_circuits()
        self.addCleanup(workflow._reset_fast_route_circuits)
        success = io.BytesIO(
            b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"status\\":\\"done\\"}"}}]}')
        raw = {}
        with patch.object(workflow.urllib.request, "urlopen", return_value=success) as request:
            result = workflow._chat({
                "llm_url": "http://primary/v1", "llm_model": "primary",
                "llm_fast_url": "http://fast/v1", "llm_fast_model": "fast",
            }, "inspect", sized_image_bytes(1920, 1536), raw=raw)
        self.assertEqual(result, {"status": "done"})
        self.assertEqual(request.call_args.args[0].full_url,
                         "http://primary/v1/chat/completions")
        self.assertEqual(raw["model_route"], "primary")
        self.assertNotIn("fast_fallback_reason", raw)

    def test_local_vlm_request_stats_are_preserved_without_unbounded_fields(self):
        response = {
            "local_vlm": {
                "backend": "transformers_local_cuda",
                "image_count": 1,
                "effective_image_token_budget": 256,
                "actual_image_token_counts": [252],
                "image_grid_thw": [[1, 28, 36]],
                "prompt_tokens": 900,
                "completion_tokens": 120,
                "generation_seconds": 12.5,
                "seconds": 12.7,
                "cache_hit": True,
                "cache_key_sha256": "a" * 64,
                "response_cache_entries": 2,
                "coalesced": True,
                "coalesced_wait_seconds": 4.2,
                "inference_stats": {"prompt_tokens": 900,
                    "completion_tokens": 120, "generation_seconds": 12.5,
                    "generation_cache_implementation": "dynamic",
                    "configured_generation_cache_implementation": "static",
                    "generation_cache_fallback_reason": "multi_image_static_cache_guard",
                    "peak_allocated_bytes": 123456789},
                "cache_release": {"large": "not copied"},
            },
            "usage": {"prompt_tokens": 900, "completion_tokens": 120,
                      "total_tokens": 1020},
            "choices": [{"finish_reason": "stop", "message": {
                "content": "{\"status\":\"done\"}"}}],
        }
        raw = {}
        with patch.object(workflow.urllib.request, "urlopen",
                          return_value=io.BytesIO(json.dumps(response).encode())):
            result = workflow._chat({"llm_url": "http://example.test",
                "llm_model": "test"}, "inspect", image_bytes(), raw=raw)
        self.assertEqual(result, {"status": "done"})
        self.assertEqual(raw["local_vlm"]["actual_image_token_counts"], [252])
        self.assertEqual(raw["local_vlm"]["generation_seconds"], 12.5)
        self.assertEqual(raw["local_vlm"]["seconds"], 12.7)
        self.assertTrue(raw["local_vlm"]["cache_hit"])
        self.assertEqual(raw["local_vlm"]["cache_key_sha256"], "a" * 64)
        self.assertEqual(raw["local_vlm"]["response_cache_entries"], 2)
        self.assertTrue(raw["local_vlm"]["coalesced"])
        self.assertEqual(raw["local_vlm"]["coalesced_wait_seconds"], 4.2)
        self.assertEqual(raw["local_vlm"]["inference_stats"], {
            "prompt_tokens": 900, "completion_tokens": 120,
            "generation_seconds": 12.5,
            "generation_cache_implementation": "dynamic",
            "configured_generation_cache_implementation": "static",
            "generation_cache_fallback_reason": "multi_image_static_cache_guard"})
        self.assertEqual(raw["local_vlm"]["usage"]["total_tokens"], 1020)
        self.assertNotIn("cache_release", raw["local_vlm"])

    def test_trace_records_sam_transport_failure_and_restores_context(self):
        events = []
        with workflow.trace_calls(events.append), \
             patch.object(workflow.urllib.request, "urlopen", side_effect=TimeoutError):
            with self.assertRaisesRegex(ValueError, "超时"):
                workflow.sam_text(image_bytes(), "person", {})
        self.assertEqual(events[-1]["status"], "failed")
        self.assertIn("超时", events[-1]["error"])
        self.assertIsNone(workflow._trace_callback.get())

    def run_agent(self, decisions, max_rounds=3):
        with patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "chat", side_effect=decisions):
            return workflow.run({}, image_bytes(),
                {"description": "annotate only class a", "prompts": [{"label": "a", "text": "square"}]},
                ["a", "b"], max_rounds=max_rounds)

    def test_edit_requires_new_observation_before_done(self):
        result = self.run_agent([
            {"status": "done", "operations": [{"operation": "reclassify",
                "shape_ids": [1], "label": "b", "reason": "wrong class"}]},
            {"status": "done", "operations": []}])
        self.assertTrue(result["agent"]["converged"])
        self.assertEqual(result["agent"]["rounds"], 2)
        self.assertEqual(result["shapes"][0]["label"], "b")

    def test_cycle_and_round_limit_require_review(self):
        cycle = self.run_agent([
            {"status": "continue", "operations": [{"operation": "reclassify",
                "shape_ids": [1], "label": "b", "reason": "reference class"}]},
            {"status": "continue", "operations": [{"operation": "reclassify",
                "shape_ids": [1], "label": "a", "reason": "reference class"}]}])
        self.assertEqual(cycle["agent"]["stop_reason"], "cycle")
        self.assertTrue(cycle["review_queue"])
        limit = self.run_agent([{"status": "continue", "operations": [
            {"operation": "reclassify", "shape_ids": [1], "label": "b", "reason": "reference class"}]}], 1)
        self.assertEqual(limit["agent"]["stop_reason"], "max_rounds")
        self.assertFalse(limit["agent"]["converged"])

    def test_invalid_sam_repair_requests_preserve_candidates(self):
        for kind in ("sam_points", "sam_text"):
            for initial in ([sam_result()], []):
                with self.subTest(kind=kind, empty=not initial):
                    decision = {"status": "done", "operations": [{"operation": kind,
                        "shape_ids": [1] if initial else [], "label": "a", "prompt": "square",
                        "points": [{"x": 5, "y": 5, "label": 1}], "reason": "repair boundary"}]}
                    with patch.object(workflow, "sam_request", return_value={"results": initial}) as sam, \
                         patch.object(workflow, "chat", return_value=decision) as chat:
                        result = workflow.run({}, image_bytes(),
                            {"description": "class a", "prompts": [{"label": "a", "text": "square"}]}, ["a"], 2)
                    self.assertEqual(sam.call_args_list[0].args[1], "/segment_text")
                    self.assertEqual(len(result["shapes"]), len(initial))
                    if initial:
                        self.assertEqual(result["shapes"][0]["parts"], shape()["parts"])
                        self.assertEqual(result["shapes"][0]["id"], 1)
                    self.assertFalse(result["agent"]["converged"])
                    self.assertEqual(result["agent"]["stop_reason"], "invalid_model_output")
                    self.assertEqual(result["agent"]["audit"], [])
                    self.assertEqual(chat.call_count, 3)
                    self.assertEqual(sam.call_count, 1)
                    self.assertIn("协议校验重试 2 次后仍失败", result["review_queue"][0]["error"])
                    self.assertEqual(result["review_queue"][0]["reason_code"],
                                     "agent_invalid_model_output")
                    self.assertEqual(result["review_queue"][0]["route"],
                                     "human_review_required")

    def test_sam_points_refine_existing_mask_locally_and_require_next_gpt_assessment(self):
        edit = {"operation": "sam_points", "shape_ids": [1],
                "polygon": [[15, 0], [25, 0], [25, 20], [15, 20]],
                "points": [{"x": 20, "y": 5, "label": 1}, {"x": 28, "y": 5, "label": 0}],
                "reason": "repair right boundary"}
        decisions = [{"status": "continue", "operations": [edit]},
                     {"status": "done", "sam_assessment": {"improved": True,
                        "reason": "missing region filled; correct boundaries preserved"}, "operations": []}]
        with patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "sam_request", return_value={"parts": shape(left=0, right=30)["parts"]}) as sam, \
             patch.object(workflow, "chat", side_effect=decisions) as chat:
            result = workflow.run({}, image_bytes(), {"description": "a only",
                "prompts": [{"label": "a", "text": "square"}]}, ["a"])
        self.assertTrue(result["agent"]["converged"])
        self.assertEqual(result["agent"]["rounds"], 2)
        self.assertEqual(sam.call_args.args[1], "/segment_points")
        self.assertEqual(sam.call_args.kwargs["points"], edit["points"])
        self.assertEqual(sam.call_args.kwargs["polygon"], edit["polygon"])
        import numpy as np
        prior = np.asarray(Image.open(io.BytesIO(base64.b64decode(sam.call_args.kwargs["mask"])))) > 127
        expected = _rasterize(shape(), 40, 40)
        self.assertEqual(prior.tolist(), expected.astype(bool).tolist())
        expected[2:19, 15:26] = 1
        self.assertEqual(_rasterize(result["shapes"][0], 40, 40).tolist(), expected.tolist())
        self.assertEqual(result["shapes"][0]["id"], 1)
        self.assertEqual(len(chat.call_args_list[1].kwargs["extra_images"]), 3)
        self.assertIn("图像 4 是修改前候选", chat.call_args_list[1].args[1])

    def test_duplicate_only_sam_points_and_zero_area_polygon_are_rejected_before_sam(self):
        invalid = {"status": "continue", "operations": [{"operation": "sam_points",
            "shape_ids": [1], "polygon": [[10, 10], [10, 10], [10, 10]],
            "points": [{"x": 10, "y": 10, "label": 1}] * 3,
            "reason": "repair boundary"}]}
        with self.assertRaisesRegex(ValueError, "有效 polygon|重复的同一个点"):
            workflow._validate_review_result(invalid, [], [shape()], ["a"], 40, 40)

    def test_conflicting_and_out_of_bounds_sam_points_are_rejected(self):
        base = {"status": "continue", "operations": [{"operation": "sam_points",
            "shape_ids": [1], "polygon": [[0, 0], [20, 0], [20, 20]],
            "points": [{"x": 5, "y": 5, "label": 1}, {"x": 5, "y": 5, "label": 0}],
            "reason": "repair boundary"}]}
        with self.assertRaisesRegex(ValueError, "冲突"):
            workflow._validate_review_result(base, [], [shape()], ["a"], 40, 40)
        base["operations"][0]["points"] = [{"x": 40, "y": 5, "label": 1}]
        with self.assertRaisesRegex(ValueError, "有效 points"):
            workflow._validate_review_result(base, [], [shape()], ["a"], 40, 40)

    def test_sam_improvement_claim_without_pending_result_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "禁止声明 improved"):
            workflow._validate_review_result({"status": "done", "operations": [],
                "sam_assessment": {"improved": True, "reason": "looks better"}},
                [], [shape()], ["a"], 40, 40)

    def test_sam_regression_rolls_back_before_opencv_fallback(self):
        edit = {"operation": "sam_points", "shape_ids": [1],
                "polygon": [[0, 0], [30, 0], [30, 30], [0, 30]],
                "points": [{"x": 5, "y": 5, "label": 1}], "reason": "repair boundary"}
        decisions = [{"status": "continue", "operations": [edit]},
            {"status": "continue", "sam_assessment": {"improved": False, "reason": "lost correct area"},
             "operations": [{"operation": "add_region", "shape_ids": [1],
                             "polygon": [[16, 2], [25, 2], [25, 18], [16, 18]], "reason": "repair boundary"}]},
            {"status": "done", "operations": []}]
        with patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "sam_request", return_value={"parts": shape(right=10)["parts"]}) as sam, \
             patch.object(workflow, "chat", side_effect=decisions):
            result = workflow.run({}, image_bytes(), {"description": "a only",
                "prompts": [{"label": "a", "text": "square"}]}, ["a"])
        self.assertTrue(result["agent"]["converged"])
        self.assertEqual(sam.call_count, 1)
        expected = _rasterize(shape(), 40, 40)
        expected[2:19, 16:26] = 1
        self.assertEqual(_rasterize(result["shapes"][0], 40, 40).tolist(), expected.tolist())
        self.assertEqual(result["agent"]["audit"][1]["sam_rollback"]["shape_id"], 1)

    def test_sam_no_change_allows_next_round_to_use_opencv(self):
        decisions = [{"status": "continue", "operations": [{"operation": "sam_points", "shape_ids": [1],
            "polygon": [[0, 0], [30, 0], [30, 30], [0, 30]],
            "points": [{"x": 5, "y": 5, "label": 1}], "reason": "repair boundary"}]},
            {"status": "continue", "operations": [{"operation": "add_region", "shape_ids": [1],
             "polygon": [[16, 2], [25, 2], [25, 18], [16, 18]], "reason": "repair boundary"}]},
            {"status": "done", "operations": []}]
        with patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "sam_request", return_value=sam_result()), \
             patch.object(workflow, "chat", side_effect=decisions):
            result = workflow.run({}, image_bytes(), {"description": "a only",
                "prompts": [{"label": "a", "text": "square"}]}, ["a"])
        self.assertTrue(result["agent"]["converged"])
        self.assertIn("OpenCV", result["agent"]["audit"][0]["tool_results"][0]["fallback"])

    def test_unassessed_sam_result_is_rolled_back_at_round_limit(self):
        decision = {"status": "continue", "operations": [{"operation": "sam_points", "shape_ids": [1],
            "polygon": [[0, 0], [30, 0], [30, 30], [0, 30]],
            "points": [{"x": 5, "y": 5, "label": 1}], "reason": "repair boundary"}]}
        with patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "sam_request", return_value={"parts": shape(right=10)["parts"]}), \
             patch.object(workflow, "chat", return_value=decision):
            result = workflow.run({}, image_bytes(), {"description": "a only",
                "prompts": [{"label": "a", "text": "square"}]}, ["a"], 1)
        self.assertFalse(result["agent"]["converged"])
        self.assertEqual(result["shapes"][0]["parts"], shape()["parts"])

    def test_stop_during_model_call_returns_without_waiting_for_response(self):
        stop, entered, release, finished = (threading.Event() for _ in range(4))
        errors = []
        def model():
            entered.set()
            release.wait(5)
            return "late response"
        def caller():
            try:
                with workflow.cancellation(stop):
                    workflow._cancellable_call(model)
            except workflow.RunCancelled:
                errors.append("cancelled")
            finally:
                finished.set()
        worker = threading.Thread(target=caller)
        worker.start()
        try:
            self.assertTrue(entered.wait(2))
            stop.set()
            self.assertTrue(finished.wait(2))
            self.assertEqual(errors, ["cancelled"])
        finally:
            release.set()
            worker.join(2)

    def test_repeated_failed_operation_is_skipped_and_audited(self):
        failed = {"operation": "sam_points", "shape_ids": [1], "label": "a",
                  "polygon": [[0, 0], [30, 0], [30, 30], [0, 30]],
                  "points": [{"x": 5, "y": 5, "label": 1}], "reason": "repair boundary"}
        decision = {"status": "continue", "operations": [failed]}
        first = {"status": "continue", "operations": [failed,
            {"operation": "reclassify", "shape_ids": [1], "label": "b", "reason": "reference class"}]}
        with patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "chat", side_effect=[first, decision]), \
             patch.object(workflow, "sam_request") as sam:
            result = workflow.run({}, image_bytes(),
                {"description": "class a", "prompts": [{"label": "a", "text": "square"}]}, ["a", "b"], 2)
        self.assertEqual([item["status"] for item in result["agent"]["audit"][1]["tool_results"]],
                         ["skipped"])
        self.assertTrue(any("停止重复重试" in item["reason"]
                            for item in result["review_queue"]))

    def test_repeated_no_change_after_repairs_requires_exact_three_round_streak(self):
        repeated = {"operation": "add_region", "shape_ids": [9],
            "polygon": [[1, 1], [8, 1], [8, 8]], "reason": "same region"}
        audit = []
        for number in range(1, 4):
            repair = {"operation": "fill_gap", "shape_ids": [11],
                "hole_id": f"hole-{number}", "polygon": [[2, 2], [4, 2], [4, 4]],
                "reason": "visible floor"}
            audit.append({"round": number,
                "decision": {"status": "continue",
                    "operations": [copy.deepcopy(repeated), repair]},
                "tool_results": [{"status": "no_change"}, {"status": "applied"}]})
        self.assertEqual(workflow.repeated_no_change_after_repairs(audit),
            {"operation": "add_region", "shape_ids": [9],
             "polygon": [[1, 1], [8, 1], [8, 8]]})
        self.assertIsNone(workflow.repeated_no_change_after_repairs(audit[:2]))
        changed = copy.deepcopy(audit)
        changed[1]["decision"]["operations"][0]["polygon"][0] = [0, 0]
        self.assertIsNone(workflow.repeated_no_change_after_repairs(changed))
        no_repair = copy.deepcopy(audit)
        no_repair[-1]["decision"]["operations"] = [repeated]
        no_repair[-1]["tool_results"] = [{"status": "no_change"}]
        self.assertIsNone(workflow.repeated_no_change_after_repairs(no_repair))

    def test_review_stops_after_third_repeated_no_change_when_last_gap_is_repaired(self):
        mask = np.ones((40, 40), dtype=np.uint8)
        for left in (6, 17, 28):
            mask[20:25, left:left + 5] = 0
        floor = _shape_from_mask(mask, "floor", 1)
        repeated = {"operation": "add_region", "shape_ids": [1],
            "polygon": [[1, 1], [4, 1], [4, 4], [1, 4]],
            "reason": "Already included region."}

        def decision(*args, **kwargs):
            hole = kwargs["holes"][0]
            x1, y1, x2, y2 = hole["bbox"]
            repair = {"operation": "fill_gap", "shape_ids": [1],
                "hole_id": hole["id"],
                "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "reason": "Visible floor gap."}
            return {"status": "continue", "reason": "Repair one visible gap.",
                    "operations": [copy.deepcopy(repeated), repair]}

        with patch.object(workflow, "review", side_effect=decision) as review_mock:
            result = workflow.review_existing({}, image_bytes(),
                {"description": "floor", "prompts": []}, ["floor"], [floor],
                max_rounds=6, reference={})
        self.assertEqual(review_mock.call_count, 3)
        self.assertEqual(result["agent"]["rounds"], 3)
        self.assertEqual(result["agent"]["stop_reason"],
                         "repeated_no_change_after_repairs")
        self.assertEqual(result["agent"]["audit"][-1]["early_stop"]["reason_code"],
                         "repeated_no_change_after_repairs")
        self.assertFalse(result["agent"]["converged"])
        self.assertTrue(result["review_queue"])
        self.assertEqual(workflow.candidate_holes(result["shapes"], ["floor"], 40, 40), [])


class FakeVideo:
    path = Path("/tmp/test.mcap")
    topics = {"camera0": {"count": 3}, "camera1": {"count": 3}}
    catalog = topics
    duration = 1

    def frame(self, topic, index):
        return image_bytes()

    def timeline(self, topic, directory):
        return {"status": "ready", "times": [0.0, 0.5, 1.0], "fps": 2,
                "duration": 1.0, "count": 3, "timestamp_source": "capture",
                "progress": 1, "playback_ready": False, "url": None}


def active_request_scope():
    return {"path": str(FakeVideo.path), "topic": "camera0",
            "file_activation_generation": server.current_file_activation_generation()}


def active_label_write_scope():
    return {**active_request_scope(),
            "expected_label_state_sha256": server.label_state_sha256()}


def active_config_write_scope():
    video = server.state.get("video")
    return {
        "expected_path": str(video.path) if video is not None else "",
        "expected_topic": server.state.get("topic"),
        "expected_file_activation_generation":
            server.current_file_activation_generation(),
        "expected_config_state_sha256": server.config_state_sha256(),
    }


def active_auto_clear_scope(frame):
    binding = server.auto_result_clear_binding(frame)
    return {**active_request_scope(),
            "expected_candidate_artifact_sha256":
                binding["candidate_artifact_sha256"],
            "expected_agent_run_sha256": binding["agent_run_sha256"]}


class ServerTests(unittest.TestCase):
    def setUp(self):
        # Product code must fail closed on low disk, but unrelated unit tests
        # must not inherit the developer machine's momentary free-space state.
        self._disk_usage_patcher = patch.object(
            server.shutil, "disk_usage",
            return_value=SimpleNamespace(
                total=20 * 1024**3, used=10 * 1024**3, free=10 * 1024**3))
        self._disk_usage_patcher.start()
        self.addCleanup(self._disk_usage_patcher.stop)

    def test_primary_model_prewarm_uses_main_route_without_state_writes(self):
        old_state = server.state
        old_status = server.run_status
        labels = ["floor", "slope", "person"]
        server.state = {"labels": labels, "config": {
            "llm_url": "http://primary/v1", "llm_model": "primary",
            "llm_fast_url": "http://fast/v1", "llm_fast_model": "fast",
        }, "truth": {"7": ["protected"]}, "candidates": {"8": ["protected"]}}
        server.run_status = {"status": "idle"}
        before = copy.deepcopy(server.state)
        discovery = {"visible_labels": ["floor"], "candidate_labels": ["floor"]}
        try:
            with patch.object(server, "discover_reference_labels",
                              return_value=(discovery, 0)) as discover:
                response = server.app.test_client().post("/api/model_prewarm", json={})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json["prewarmed"])
            self.assertTrue(response.json["candidate_only"])
            self.assertFalse(response.json["human_truth"])
            self.assertFalse(response.json["approved_for_training"])
            self.assertFalse(response.json["state_written"])
            config, image, passed_labels = discover.call_args.args
            self.assertEqual(config["llm_url"], "http://primary/v1")
            self.assertEqual(config["llm_busy_retries"], 0)
            self.assertEqual(config["llm_max_tokens"], 2048)
            self.assertNotIn("llm_fast_url", config)
            self.assertNotIn("llm_fast_model", config)
            self.assertEqual(passed_labels, labels)
            prewarm_image = Image.open(io.BytesIO(image))
            self.assertEqual(prewarm_image.size, (640, 512))
            self.assertTrue(any(low != high for low, high in prewarm_image.getextrema()))
            self.assertEqual(response.json["profile"], "synthetic_factory_present_v1")
            self.assertEqual(server.state, before)
        finally:
            server.state = old_state
            server.run_status = old_status

    def test_primary_model_prewarm_yields_to_active_workflow(self):
        old_status = server.run_status
        server.run_status = {"status": "running"}
        try:
            with patch.object(server, "discover_reference_labels") as discover:
                response = server.app.test_client().post("/api/model_prewarm", json={})
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json["code"], "active_workflow")
            discover.assert_not_called()
        finally:
            server.run_status = old_status

    def test_model_readiness_probes_sam_and_vlm_concurrently(self):
        old_state = server.state
        barrier = threading.Barrier(2, timeout=1)
        calls = []

        def probe(url, timeout=3):
            calls.append(url)
            barrier.wait()
            if url.endswith("/health"):
                return {"status": "ok", "model_loaded": True}
            return {"data": [{"id": "vlm-test"}]}

        server.state = {"config": {
            "sam3_url": "http://sam",
            "llm_url": "http://vlm/v1",
            "llm_model": "vlm-test",
        }}
        try:
            with patch.object(server, "_fetch_readiness_json", side_effect=probe):
                result = server.app.test_client().get("/api/model_readiness").json
            self.assertTrue(result["ready"])
            self.assertEqual(set(calls), {"http://sam/health", "http://vlm/v1/models"})
            self.assertGreaterEqual(result["elapsed_ms"], 0)
        finally:
            server.state = old_state

    def test_model_readiness_reports_optional_fast_route_without_gating_primary(self):
        old_state = server.state
        server.state = {"config": {
            "sam3_url": "http://sam", "llm_url": "http://vlm/v1",
            "llm_model": "primary", "llm_fast_url": "http://fast/v1",
            "llm_fast_model": "fast", "llm_fast_max_pixels": 327680,
            "llm_fast_max_long_edge": 640,
        }}
        def probe(url, timeout=3):
            if url == "http://sam/health":
                return {"status": "ok", "model_loaded": True}
            if url == "http://vlm/v1/models":
                return {"data": [{"id": "primary"}]}
            raise RuntimeError("fast offline")
        try:
            with patch.object(server, "_fetch_readiness_json", side_effect=probe):
                result = server.app.test_client().get("/api/model_readiness").json
            self.assertTrue(result["ready"])
            self.assertTrue(result["fast_vlm"]["configured"])
            self.assertTrue(result["fast_vlm"]["acceleration_only"])
            self.assertFalse(result["fast_vlm"]["ready"])
            self.assertEqual(result["fast_vlm"]["max_pixels"], 327680)
            self.assertEqual(result["fast_vlm"]["max_long_edge"], 640)
            self.assertEqual(result["fast_vlm"]["cooldown_seconds"], 60)
        finally:
            server.state = old_state

    def test_fast_route_config_validates_and_can_be_disabled(self):
        old_state = server.state
        server.state = {"config": {}}
        try:
            client = server.app.test_client()
            with patch.object(server, "save"):
                previous_sha = server.config_state_sha256()
                saved = client.post("/api/config", json={
                    "llm_fast_url": "http://127.0.0.1:38868/v1",
                    "llm_fast_model": "fast", "llm_fast_max_pixels": 327680,
                    "llm_fast_max_long_edge": 640,
                    "llm_fast_max_extra_pixels": 1310720,
                    "llm_fast_max_extra_long_edge": 1280,
                    "llm_fast_timeout_seconds": 30,
                    "llm_fast_cooldown_seconds": 60,
                    **active_config_write_scope()})
                self.assertEqual(saved.status_code, 200, saved.json)
                self.assertEqual(server.state["config"]["llm_fast_timeout_seconds"], 30)
                self.assertEqual(saved.json["previous_config_state_sha256"],
                                 previous_sha)
                rejected = client.post("/api/config", json={
                    "llm_fast_url": "http://user:secret@127.0.0.1/v1"})
                self.assertEqual(rejected.status_code, 400, rejected.json)
                malformed = client.post("/api/config", json=["not", "an", "object"])
                self.assertEqual(malformed.status_code, 400, malformed.json)
                disabled = client.post("/api/config", json={
                    "llm_fast_url": "", "llm_fast_model": "",
                    **active_config_write_scope()})
                self.assertEqual(disabled.status_code, 200, disabled.json)
                self.assertNotIn("llm_fast_url", server.state["config"])
                self.assertNotIn("llm_fast_model", server.state["config"])
        finally:
            server.state = old_state

    def test_config_write_rejects_stale_active_and_failed_transactions(self):
        old_state, old_status = server.state, server.run_status
        server.state = {"config": {"llm_model": "before"}}
        server.run_status = {"status": "idle"}
        client = server.app.test_client()
        try:
            stale = active_config_write_scope()
            stale["expected_config_state_sha256"] = "0" * 64
            response = client.post("/api/config", json={
                "llm_model": "stale", **stale})
            self.assertEqual(response.status_code, 409, response.json)
            self.assertEqual(server.state["config"], {"llm_model": "before"})

            server.run_status = {"status": "running"}
            response = client.post("/api/config", json={
                "llm_model": "active", **active_config_write_scope()})
            self.assertEqual(response.status_code, 503, response.json)
            self.assertEqual(server.state["config"], {"llm_model": "before"})

            server.run_status = {"status": "idle"}
            with patch.object(server, "save", side_effect=OSError("disk full")):
                response = client.post("/api/config", json={
                    "llm_model": "uncommitted", **active_config_write_scope()})
            self.assertEqual(response.status_code, 503, response.json)
            self.assertIn("已恢复原配置", response.json["error"])
            self.assertEqual(server.state["config"], {"llm_model": "before"})
        finally:
            server.state, server.run_status = old_state, old_status

    def test_config_readback_hides_both_keys_but_hash_tracks_them(self):
        old_state = server.state
        server.state = {
            "video": None, "topic": None, "labels": [], "truth": {},
            "reference_frames": {}, "config": {
                "llm_model": "model", "llm_key": "primary-private",
                "llm_fast_key": "fast-private"},
            "candidates": {}, "agent_runs": {},
        }
        try:
            before = server.config_state_sha256()
            client = server.app.test_client()
            for endpoint in ("/api/bootstrap", "/api/state"):
                response = client.get(endpoint)
                self.assertEqual(response.status_code, 200, response.json)
                self.assertNotIn("llm_key", response.json["config"])
                self.assertNotIn("llm_fast_key", response.json["config"])
                self.assertEqual(response.json["config_state_sha256"], before)
            server.state["config"]["llm_fast_key"] = "rotated"
            self.assertNotEqual(server.config_state_sha256(), before)
        finally:
            server.state = old_state

    def test_run_accepts_exact_retry_frames_without_reprocessing_successful_range(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            description = {"description": "a only", "prompts": [{"label": "a", "text": "a"}],
                           "reference_signature": server.reference_signature(0)}
            calls = []
            def successful(*args, **kwargs):
                calls.append(len(calls) + 1)
                return {"shapes": [shape(calls[-1])], "candidates_added": 1,
                        "applied": [], "review_queue": [],
                        "agent": {"stop_reason": "done", "rounds": 1}}
            try:
                client = server.app.test_client()
                server.save_run({
                    "task_id": "retry-source-task", "file": str(FakeVideo.path),
                    "topic": "camera0", "reference": 0, "status": "failed",
                    "total": 2, "targets": [2, 1],
                    "frames": [{"frame": 2, "failed": True},
                               {"frame": 1, "failed": True}],
                    "failed_frames": [2, 1]})
                with patch.object(server, "run", side_effect=successful):
                    response = client.post("/api/run", json={
                        "frames": [2, 1, 2], "description": description,
                        "expected_file": str(FakeVideo.path),
                        "expected_topic": "camera0",
                        "expected_result_task_id": "retry-source-task"})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual([item["frame"] for item in response.json["frames"]], [2, 1])
                self.assertEqual(response.json["requested_frames"], [2, 1])
                self.assertEqual(len(calls), 2)
                self.assertEqual(set(server.state["candidates"]), {"1", "2"})
                invalid_type = client.post("/api/run", json={
                    "frames": [True], "description": description})
                self.assertEqual(invalid_type.status_code, 400, invalid_type.json)
                invalid_range = client.post("/api/run", json={
                    "frames": [3], "description": description})
                self.assertEqual(invalid_range.status_code, 400, invalid_range.json)
                missing_identity = client.post("/api/run", json={
                    "frames": [1], "description": description})
                self.assertEqual(missing_identity.status_code, 400, missing_identity.json)
                self.assertIn("缺少来源任务编号", missing_identity.json["error"])
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_run_rejects_retry_after_a_newer_result_replaced_its_source_task(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {"1": [shape(8)]}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            description = {"description": "a only", "prompts": [{"label": "a", "text": "a"}],
                           "reference_signature": server.reference_signature(0)}
            server.save_run({
                "task_id": "newer-result-task", "file": str(FakeVideo.path),
                "topic": "camera0", "reference": 0, "status": "completed",
                "total": 1, "targets": [1],
                "frames": [{"frame": 1, "remaining": 1, "failed": False}],
                "failed_frames": []})
            before = copy.deepcopy(server.state["candidates"])
            try:
                with patch.object(server, "run") as inference:
                    response = server.app.test_client().post("/api/run", json={
                        "frames": [1], "description": description,
                        "expected_file": str(FakeVideo.path),
                        "expected_topic": "camera0",
                        "expected_result_task_id": "stale-failed-task"})
                self.assertEqual(response.status_code, 409, response.json)
                self.assertIn("未启动旧重试", response.json["error"])
                inference.assert_not_called()
                self.assertEqual(server.state["candidates"], before)
                self.assertEqual(server.load_run_summary()["task_id"], "newer-result-task")
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_run_rejects_preflight_scope_that_no_longer_matches_loaded_data(self):
        old_state, old_status = server.state, server.run_status
        server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
            "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
            "config": {}, "candidates": {}, "agent_runs": {}}
        server.run_status = {"status": "idle"}
        client = server.app.test_client()
        try:
            wrong_file = client.post("/api/run", json={"frame": 1, "count": 1,
                "expected_file": "D:/different-dataset.mcap", "expected_topic": "camera0"})
            self.assertEqual(wrong_file.status_code, 409, wrong_file.json)
            self.assertIn("本次自动标注未启动", wrong_file.json["error"])
            wrong_topic = client.post("/api/run", json={"frame": 1, "count": 1,
                "expected_file": str(FakeVideo.path), "expected_topic": "camera1"})
            self.assertEqual(wrong_topic.status_code, 409, wrong_topic.json)
            missing_pair = client.post("/api/run", json={"frame": 1, "count": 1,
                "expected_file": str(FakeVideo.path)})
            self.assertEqual(missing_pair.status_code, 400, missing_pair.json)
        finally:
            server.state, server.run_status = old_state, old_status

    def test_truth_comparison_run_keeps_truth_and_stores_separate_candidate(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            original_target_truth = [shape(7)]
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()], "1": copy.deepcopy(original_target_truth)},
                "reference_frames": {"camera0": [0]}, "config": {},
                "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            description = {"description": "a only", "prompts": [{"label": "a", "text": "a"}],
                           "reference_signature": server.reference_signature(0)}
            inferred = {"shapes": [shape(9)], "candidates_added": 1,
                        "applied": [], "review_queue": [],
                        "agent": {"stop_reason": "done", "rounds": 1}}
            try:
                with patch.object(server, "run", return_value=inferred):
                    response = server.app.test_client().post("/api/run", json={
                        "frame": 1, "count": 1, "compare_with_truth": True,
                        "description": description})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertTrue(response.json["compare_with_truth"])
                self.assertEqual(response.json["frames"][0]["frame"], 1)
                self.assertEqual(server.state["truth"]["1"], original_target_truth)
                self.assertEqual(server.state["candidates"]["1"][0]["id"], 9)
                self.assertFalse(server.state["agent_runs"]["1"].get("human_truth", False))
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_main_run_routes_solid_pallet_candidate_through_isolated_audit(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            original_truth = [shape(1, "pallet")]
            server.state = {
                "video": FakeVideo(), "topic": "camera0", "labels": ["pallet"],
                "truth": {"0": copy.deepcopy(original_truth)},
                "reference_frames": {"camera0": [0]}, "config": {},
                "candidates": {}, "agent_runs": {}, "candidate_review_status": {},
            }
            server.run_status = {"status": "idle"}
            description = {
                "description": "pallet only",
                "prompts": [{"label": "pallet", "text": "pallet"}],
                "reference_signature": server.reference_signature(0),
            }
            inferred = {
                "shapes": [shape(9, "pallet")], "candidates_added": 1,
                "applied": [], "review_queue": [],
                "agent": {"stop_reason": "done", "rounds": 1},
            }
            audit = {
                "schema_version": "labelagent.blind_candidate_class_audit.v1",
                "candidate_id": 9, "candidate_sha256": "f" * 64,
                "route": "human_review_required", "candidate_mutated": False,
                "human_truth": False, "approved_for_training": False,
                "decision": {"classification": "conflict"},
                "audit_sha256": "a" * 64,
            }
            queue = {
                "source": "candidate_class_audit", "candidate_id": 9,
                "shape_ids": [9], "route": "human_review_required",
                "candidate_sha256": "f" * 64,
                "candidate_artifact_sha256": "b" * 64,
                "audit_sha256": "a" * 64, "reason": "solid box",
            }
            try:
                with patch.object(server, "run", return_value=inferred), \
                     patch.object(server, "candidate_requires_class_audit",
                                  side_effect=lambda candidate: candidate.get("label") == "pallet"), \
                     patch.object(server, "audit_high_risk_candidates",
                                  return_value=([copy.deepcopy(audit)], [copy.deepcopy(queue)])) as focused:
                    response = server.app.test_client().post("/api/run", json={
                        "frame": 1, "count": 1, "description": description})
                self.assertEqual(response.status_code, 200, response.json)
                focused.assert_called_once()
                self.assertEqual(response.json["frames"][0]["review_queue"][0]["source"],
                                 "candidate_class_audit")
                saved = server.state["agent_runs"]["1"]
                self.assertEqual(saved["candidate_class_audit_status"], "complete")
                self.assertEqual(saved["candidate_class_audit_total"], 1)
                self.assertEqual(saved["candidate_class_audits"][0]["candidate_id"], 9)
                self.assertEqual(
                    server.state["candidate_review_status"]["1"]["f" * 64]["route"],
                    "human_review_required")
                self.assertEqual(server.state["truth"]["0"], original_truth)
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_reusable_candidate_class_audit_can_validate_an_explicit_subset(self):
        pallet, person = shape(9, "pallet"), shape(10, "person")
        candidates = [pallet, person]
        artifact_sha = server._sha256_json(candidates)
        input_sha = "1" * 64
        audit_version = {"schema_version": "test", "protocol_sha256": "2" * 64}
        fingerprint = server.candidate_fingerprint(pallet)
        decision = {
            "classification": "known_label", "observed_label": "pallet",
            "candidate_supported": True, "confidence": 0.97,
            "reason_code": "visible_known_label",
            "visible_cues": ["visible pallet deck"], "reason": "pallet structure visible",
        }
        audit = {
            "schema_version": "labelagent.blind_candidate_class_audit.v1",
            "candidate_id": 9, "candidate_sha256": fingerprint,
            "candidate_artifact_sha256": artifact_sha,
            "source_image_sha256": input_sha, "bound_current_label": "pallet",
            "audit_version": audit_version, "route": "candidate_preserved",
            "minimum_confidence": 0.90, "candidate_mutated": False,
            "human_truth": False, "approved_for_training": False,
            "approved_for_deployment": False, "decision": decision,
        }
        audit["audit_sha256"] = server._sha256_json(audit)
        run = {
            "candidate_class_audit_status": "complete",
            "candidate_class_audit_version": audit_version,
            "candidate_artifact_sha256": artifact_sha, "input_sha256": input_sha,
            "candidate_class_audits": [audit],
        }
        review_status = {fingerprint: {
            "candidate_fingerprint": fingerprint,
            "candidate_artifact_sha256": artifact_sha,
            "audit_sha256": audit["audit_sha256"], "route": "candidate_preserved",
        }}
        with patch.object(server, "candidate_requires_class_audit", return_value=True):
            self.assertFalse(server.reusable_candidate_class_audit(
                run, review_status, candidates, ["pallet", "person"], artifact_sha,
                input_sha, audit_version))
            self.assertTrue(server.reusable_candidate_class_audit(
                run, review_status, candidates, ["pallet", "person"], artifact_sha,
                input_sha, audit_version, candidate_ids={9}))
            self.assertFalse(server.reusable_candidate_class_audit(
                run, {}, candidates, ["pallet", "person"], artifact_sha,
                input_sha, audit_version, candidate_ids={9},
                require_review_bindings=True))

    def test_main_run_reuses_exact_pallet_audit_without_second_vlm_call(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {
                "video": FakeVideo(), "topic": "camera0", "labels": ["pallet"],
                "truth": {"0": [shape(1, "pallet")]},
                "reference_frames": {"camera0": [0]}, "config": {},
                "candidates": {}, "agent_runs": {}, "candidate_review_status": {},
            }
            server.run_status = {"status": "idle"}
            description = {
                "description": "pallet only",
                "prompts": [{"label": "pallet", "text": "pallet"}],
                "reference_signature": server.reference_signature(0),
            }
            inferred = {
                "shapes": [shape(9, "pallet")], "candidates_added": 1,
                "applied": [], "review_queue": [],
                "agent": {"stop_reason": "done", "rounds": 1},
            }

            def audit_once(config, frame_image, candidates, labels, definitions, chat, **kwargs):
                candidate = next(item for item in candidates if item["id"] == 9)
                audit = {
                    "schema_version": "labelagent.blind_candidate_class_audit.v1",
                    "candidate_id": 9,
                    "candidate_sha256": server.candidate_fingerprint(candidate),
                    "candidate_artifact_sha256": server._sha256_json(candidates),
                    "source_image_sha256": server.hashlib.sha256(frame_image).hexdigest(),
                    "bound_current_label": "pallet", "route": "candidate_preserved",
                    "minimum_confidence": 0.90, "candidate_mutated": False,
                    "human_truth": False, "approved_for_training": False,
                    "approved_for_deployment": False,
                    "decision": {
                        "classification": "known_label", "observed_label": "pallet",
                        "candidate_supported": True, "confidence": 0.97,
                        "reason_code": "visible_known_label",
                        "visible_cues": ["visible pallet deck"],
                        "reason": "pallet structure visible",
                    },
                }
                return [audit], []

            try:
                with patch.object(server, "run", side_effect=lambda *args, **kwargs: copy.deepcopy(inferred)), \
                     patch.object(server, "candidate_requires_class_audit",
                                  side_effect=lambda candidate: candidate.get("label") == "pallet"), \
                     patch.object(server, "audit_high_risk_candidates",
                                  side_effect=audit_once) as focused:
                    first = server.app.test_client().post("/api/run", json={
                        "frame": 1, "count": 1, "description": description})
                    second = server.app.test_client().post("/api/run", json={
                        "frame": 1, "count": 1, "description": description})
                self.assertEqual(first.status_code, 200, first.json)
                self.assertEqual(second.status_code, 200, second.json)
                self.assertEqual(focused.call_count, 1)
                self.assertFalse(first.json["frames"][0]["candidate_class_audit_cache_hit"])
                self.assertTrue(second.json["frames"][0]["candidate_class_audit_cache_hit"])
                saved = server.state["agent_runs"]["1"]
                self.assertTrue(saved["candidate_class_audit_cache_hit"])
                self.assertEqual(saved["candidate_class_audits"][0]["candidate_id"], 9)
                self.assertEqual(server.state["truth"], {"0": [shape(1, "pallet")]})
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_run_summary_is_bounded_scoped_and_recoverable_after_idle_restart(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0"}
            server.run_status = {"status": "idle"}
            output = {
                "task_id": "task-123", "status": "failed", "file": str(FakeVideo.path),
                "topic": "camera0", "reference": 0, "total": 2, "targets": [1, 2],
                "frames": [
                    {"frame": 1, "remaining": 3, "candidates_added": 2,
                     "failed": False, "agent": {"large": "x" * 10000}},
                    {"frame": 2, "remaining": 1, "candidates_added": 0,
                     "failed": True, "existing_candidates_preserved": True,
                     "review_queue": [{"reason": "large" * 1000}]},
                ],
                "failed_frames": [2], "message": "1/2 帧失败",
                "description": {"large": "y" * 10000}, "trace": [{"large": "z" * 10000}],
                "steps": [{"image": "huge" * 10000}], "current": None,
            }
            try:
                server.save_run(output)
                summary_path = server.WORK / "last_run_summary.json"
                self.assertTrue(summary_path.is_file())
                self.assertLess(summary_path.stat().st_size, 2048)
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                self.assertEqual(summary["task_id"], "task-123")
                self.assertEqual(summary["failed_frames"], [2])
                self.assertEqual(summary["pending_frames"], [])
                self.assertNotIn("trace", summary)
                self.assertNotIn("agent", summary["frames"][0])
                summary_path.unlink()
                migrated = server.load_run_summary()
                self.assertEqual(migrated["task_id"], "task-123")
                self.assertTrue(summary_path.is_file())
                client = server.app.test_client()
                recovered = client.get("/api/run/result?task_id=task-123")
                self.assertEqual(recovered.status_code, 200, recovered.json)
                status = client.get("/api/run/status").json
                self.assertTrue(status["result_available"])
                self.assertEqual(status["result_task_id"], "task-123")
                self.assertEqual(status["failed_frames"], [2])
                self.assertEqual(status["selected_file"], str(FakeVideo.path))
                self.assertEqual(status["selected_topic"], "camera0")
                stale = client.get("/api/run/result?task_id=other-task")
                self.assertEqual(stale.status_code, 409, stale.json)
                server.state["topic"] = "camera1"
                wrong_scope = client.get("/api/run/result?task_id=task-123")
                self.assertEqual(wrong_scope.status_code, 409, wrong_scope.json)
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_interrupted_run_summary_exposes_only_unfinished_frames_for_resume(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0"}
            server.run_status = {"status": "idle"}
            output = {
                "task_id": "interrupted-123", "status": "running",
                "file": str(FakeVideo.path), "topic": "camera0", "reference": 0,
                "total": 2, "targets": [1, 2],
                "frames": [{"frame": 1, "remaining": 2, "failed": False}],
                "current": {"frame": 2, "phase": "observing", "round": 1},
            }
            try:
                server.save_run(output)
                summary = server.load_run_summary()
                self.assertEqual(summary["pending_frames"], [2])
                self.assertEqual(summary["completed"], 1)
                client = server.app.test_client()
                status = client.get("/api/run/status").json
                self.assertTrue(status["result_available"])
                self.assertTrue(status["result_interrupted"])
                self.assertEqual(status["pending_frames"], [2])
                recovered = client.get(
                    "/api/run/result?task_id=interrupted-123").json
                self.assertEqual(recovered["pending_frames"], [2])
                self.assertEqual([item["frame"] for item in recovered["frames"]], [1])
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_run_and_debug_records_use_compact_atomic_json_without_semantic_loss(self):
        old_work = server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            output = {"status": "running", "file": "中文.mcap", "topic": "相机",
                      "targets": [1, 2], "frames": [{"frame": 1, "remaining": 3}],
                      "trace": [{"message": "斜坡", "values": list(range(100))}]}
            trace = [{"service": "GPT", "message": "人工复核", "round": 2}]
            try:
                server.save_run(output)
                server.save_debug(trace)
                run_bytes = (server.WORK / "last_run.json").read_bytes()
                debug_bytes = (server.WORK / "model_debug.json").read_bytes()
                self.assertNotIn(b"\n  ", run_bytes)
                self.assertNotIn(b"\n  ", debug_bytes)
                self.assertEqual(json.loads(run_bytes.decode("utf-8")), output)
                self.assertEqual(json.loads(debug_bytes.decode("utf-8")), {"trace": trace})
                self.assertFalse((server.WORK / "last_run.tmp").exists())
                self.assertFalse((server.WORK / "model_debug.tmp").exists())
            finally:
                server.WORK = old_work

    def test_timeline_endpoint_returns_exact_times_without_starting_playback(self):
        old_state, old_work = server.state, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0"}
            try:
                response = server.app.test_client().get("/api/timeline?topic=camera1")
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(response.json["times"], [0.0, 0.5, 1.0])
                self.assertEqual(response.json["topic"], "camera1")
                self.assertEqual(response.json["offset"], 0)
                self.assertFalse(response.json["playback_ready"])
            finally:
                server.state, server.WORK = old_state, old_work

    def test_fast_and_streaming_json_writes_are_equivalent(self):
        value = {"中文": [shape(), {"text": "斜坡", "value": 1.25}], "flag": True}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fast, streaming = root / "fast.json", root / "streaming.json"
            server._write_json(fast, value, fast=True)
            server._write_json(streaming, value, fast=False)
            self.assertEqual(fast.read_bytes(), streaming.read_bytes())
            self.assertEqual(json.loads(fast.read_text(encoding="utf-8")), value)
            self.assertTrue(server._fast_json_write_allowed(fast))
            oversized = root / "oversized.json"
            with oversized.open("wb") as handle:
                handle.truncate(server.FAST_JSON_WRITE_LIMIT + 1)
            self.assertFalse(server._fast_json_write_allowed(oversized))
            self.assertFalse(server._fast_json_write_allowed(root / "missing.json"))

    def test_mcap_catalog_index_reuses_metadata_and_invalidates_on_file_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording = root / "sample.mcap"
            recording.write_bytes(b"first-version")
            metadata = ({"camera": {
                "type": "image", "schema": "test.Image", "encoding": "protobuf",
                "count": 3, "offset": 0.0, "log_offset": 0.0,
            }}, 123, 2.5)
            with patch.object(mcap_video, "catalog", return_value=metadata) as build:
                first = mcap_video.Video(recording, root / "index")
                second = mcap_video.Video(recording, root / "index")
                self.assertEqual(build.call_count, 1)
                self.assertEqual(first.catalog, second.catalog)
                self.assertEqual(second.origin_ns, 123)
                self.assertEqual(second.duration, 2.5)
                recording.write_bytes(b"second-version-is-different")
                third = mcap_video.Video(recording, root / "index")
                self.assertEqual(build.call_count, 2)
                self.assertEqual(third.catalog, metadata[0])

    def test_run_claim_rejects_concurrent_run_and_reference_draft(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        old_task, old_stop = server.run_task_id, server.run_stop_event
        entered, release = threading.Event(), threading.Event()
        responses = {}
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            description = {"description": "a only", "prompts": [{"label": "a", "text": "a"}],
                           "reference_signature": server.reference_signature(0)}
            def blocked_run(*args, **kwargs):
                entered.set()
                self.assertTrue(release.wait(3))
                return {"shapes": [shape()], "candidates_added": 1, "applied": [],
                    "review_queue": [], "agent": {"stop_reason": "done", "rounds": 1}}
            def start_run():
                responses["run"] = server.app.test_client().post("/api/run", json={
                    "frame": 1, "count": 1, "description": description})
            worker = threading.Thread(target=start_run)
            try:
                with patch.object(server, "run", side_effect=blocked_run):
                    worker.start()
                    self.assertTrue(entered.wait(2))
                    responses["draft"] = server.app.test_client().post("/api/reference_draft", json={
                        "frame": 2, "labels": ["a"]})
                    self.assertEqual(responses["draft"].status_code, 503)
                    release.set()
                    worker.join(4)
                self.assertFalse(worker.is_alive())
                self.assertEqual(responses["run"].status_code, 200, responses["run"].json)
                self.assertEqual(server.run_status["status"], "completed")
            finally:
                release.set()
                if worker.ident is not None:
                    worker.join(4)
                server.state, server.run_status, server.WORK = old_state, old_status, old_work
                server.run_task_id, server.run_stop_event = old_task, old_stop

    def test_run_claim_rejects_concurrent_reference_draft_and_run(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        old_task, old_stop = server.run_task_id, server.run_stop_event
        entered, release = threading.Event(), threading.Event()
        responses = {}
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {}, "reference_frames": {}, "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            def blocked_describe(*args, **kwargs):
                entered.set()
                self.assertTrue(release.wait(3))
                return {"description": "a", "prompts": []}
            result = {"shapes": [shape()], "candidates_added": 1, "applied": [],
                      "review_queue": [], "agent": {"stop_reason": "done", "rounds": 1}}
            def start_draft():
                responses["draft"] = server.app.test_client().post("/api/reference_draft", json={
                    "frame": 1, "labels": ["a"]})
            worker = threading.Thread(target=start_draft)
            try:
                with patch.object(server, "describe", side_effect=blocked_describe), \
                     patch.object(server, "run", return_value=result), \
                     patch.object(server, "audit_high_risk_candidates", return_value=([], [])):
                    worker.start()
                    self.assertTrue(entered.wait(2))
                    responses["run"] = server.app.test_client().post("/api/run", json={"frame": 1})
                    self.assertEqual(responses["run"].status_code, 503)
                    for endpoint in ("/api/describe", "/api/segment_points",
                                     "/api/reference_candidates/audit"):
                        with self.subTest(endpoint=endpoint):
                            rejected = server.app.test_client().post(endpoint, json={"frame": 0})
                            self.assertEqual(rejected.status_code, 503, rejected.json)
                    release.set()
                    worker.join(4)
                self.assertFalse(worker.is_alive())
                self.assertEqual(responses["draft"].status_code, 200, responses["draft"].json)
                self.assertEqual(server.run_status["status"], "completed")
            finally:
                release.set()
                if worker.ident is not None:
                    worker.join(4)
                server.state, server.run_status, server.WORK = old_state, old_status, old_work
                server.run_task_id, server.run_stop_event = old_task, old_stop

    def test_run_claim_keeps_ownership_through_terminal_response_and_error_cleanup(self):
        old_status, old_task, old_stop = server.run_status, server.run_task_id, server.run_stop_event
        try:
            for terminal in ("completed", "failed", "stopped"):
                with self.subTest(terminal=terminal):
                    server.run_status = {"status": "idle"}
                    entered, release = threading.Event(), threading.Event()
                    errors, task_ids = [], []
                    @server.run_claimed
                    def old_task():
                        task_ids.append(server.run_task_id)
                        server.update_run_status(status=terminal, phase=terminal)
                        entered.set()
                        if not release.wait(3):
                            raise AssertionError("未释放旧任务收尾")
                        if terminal == "failed":
                            raise RuntimeError("旧任务异常")
                        server.update_run_status(error="旧任务收尾")
                    @server.run_claimed
                    def new_task():
                        task_ids.append(server.run_task_id)
                        self.assertFalse(server.run_stop_event.is_set())
                        server.update_run_status(status="completed", phase="completed", error="新任务")
                    def start_old():
                        try:
                            old_task()
                        except Exception as error:
                            errors.append(str(error))
                    worker = threading.Thread(target=start_old)
                    worker.start()
                    try:
                        self.assertTrue(entered.wait(2))
                        with self.assertRaisesRegex(RuntimeError, "已有推理任务"):
                            new_task()
                        self.assertEqual(len(task_ids), 1)
                    finally:
                        release.set()
                        worker.join(4)
                    self.assertFalse(worker.is_alive())
                    self.assertEqual(errors, ["旧任务异常"] if terminal == "failed" else [])
                    new_task()
                    self.assertNotEqual(task_ids[0], task_ids[1])
                    self.assertEqual(server.run_status["error"], "新任务")
                    self.assertEqual(server.run_status["status"], "completed")
        finally:
            server.run_status, server.run_task_id, server.run_stop_event = old_status, old_task, old_stop

    def test_run_claim_reserves_before_preparation_and_preserves_stop(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        old_task, old_stop = server.run_task_id, server.run_stop_event
        entered, release = threading.Event(), threading.Event()
        responses = []
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            description = {"description": "a", "prompts": [{"label": "a", "text": "a"}],
                           "reference_signature": server.reference_signature(0)}
            original = server.restrict_description
            def prepare(*args, **kwargs):
                entered.set()
                self.assertTrue(release.wait(3))
                return original(*args, **kwargs)
            def start():
                responses.append(server.app.test_client().post("/api/run", json={
                    "frame": 1, "count": 1, "description": description}))
            worker = threading.Thread(target=start)
            try:
                with patch.object(server, "restrict_description", side_effect=prepare), \
                     patch.object(server, "run") as inference:
                    worker.start()
                    self.assertTrue(entered.wait(2))
                    task_event = server.run_stop_event
                    self.assertTrue(server.app.test_client().post("/api/run/stop", json={
                        "expected_task_id": server.run_status["task_id"]}).json["stopped"])
                    conflict = server.app.test_client().post("/api/reference_draft", json={
                        "frame": 2, "labels": ["a"]})
                    self.assertEqual(conflict.status_code, 503)
                    self.assertIs(server.run_stop_event, task_event)
                    self.assertTrue(task_event.is_set())
                    release.set()
                    worker.join(4)
                    inference.assert_not_called()
                self.assertFalse(worker.is_alive())
                self.assertTrue(responses[0].json["stopped"])
                self.assertEqual(server.run_status["status"], "stopped")
            finally:
                release.set()
                if worker.ident is not None:
                    worker.join(4)
                server.state, server.run_status, server.WORK = old_state, old_status, old_work
                server.run_task_id, server.run_stop_event = old_task, old_stop

    def test_run_model_wait_does_not_block_state_and_cancel_prevents_commit(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        entered, release = threading.Event(), threading.Event()
        response_box = []
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            description = {"description": "a only", "prompts": [{"label": "a", "text": "a"}],
                           "reference_signature": server.reference_signature(0)}
            def blocked_run(*args, **kwargs):
                entered.set()
                self.assertTrue(release.wait(3))
                return {"shapes": [shape()], "candidates_added": 1, "applied": [],
                    "review_queue": [], "agent": {"stop_reason": "done", "rounds": 1}}
            def start_run():
                with server.app.test_client() as client:
                    response_box.append(client.post("/api/run", json={"frame": 1, "count": 1,
                                                                        "description": description}))
            worker = threading.Thread(target=start_run)
            try:
                with patch.object(server, "run", side_effect=blocked_run):
                    worker.start()
                    self.assertTrue(entered.wait(2))
                    self.assertEqual(server.app.test_client().get("/api/state").status_code, 200)
                    self.assertTrue(server.app.test_client().post("/api/run/stop", json={
                        "expected_task_id": server.run_status["task_id"]}).json["stopped"])
                    release.set()
                    worker.join(4)
                self.assertFalse(worker.is_alive())
                self.assertTrue(response_box[0].json["stopped"])
                self.assertNotIn("1", server.state["candidates"])
                self.assertNotIn("1", server.state["agent_runs"])
            finally:
                release.set()
                if worker.ident is not None:
                    worker.join(4)
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_run_rejects_stale_target_and_preserves_completed_frame(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        entered, release = threading.Event(), threading.Event()
        responses = []
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            description = {"description": "a only", "prompts": [{"label": "a", "text": "a"}],
                           "reference_signature": server.reference_signature(0)}
            calls = []
            def blocked_run(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    entered.set()
                    self.assertTrue(release.wait(3))
                return {"shapes": [shape()], "candidates_added": 1, "applied": [],
                    "review_queue": [], "agent": {"stop_reason": "done", "rounds": 1}}
            def start_run():
                with server.app.test_client() as client:
                    responses.append(client.post("/api/run", json={"frame": 1, "count": 3,
                                                                      "description": description}))
            worker = threading.Thread(target=start_run)
            try:
                with patch.object(server, "run", side_effect=blocked_run):
                    worker.start()
                    self.assertTrue(entered.wait(2))
                    with server.lock:
                        server.state["candidates"]["2"] = [shape(99)]
                    release.set()
                    worker.join(4)
                self.assertFalse(worker.is_alive())
                self.assertEqual(responses[0].status_code, 409, responses[0].json)
                self.assertIn("1", server.state["candidates"])
                self.assertEqual(server.state["candidates"]["2"], [shape(99)])
            finally:
                release.set()
                if worker.ident is not None:
                    worker.join(4)
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_run_rejects_concurrent_scope_config_truth_reference_labels_and_run_changes(self):
        mutations = {
            "scope": lambda: server.state.update(topic="camera1"),
            "video": lambda: server.state.update(video=FakeVideo()),
            "config": lambda: server.state["config"].update(llm_key="new-key"),
            "labels": lambda: server.state["labels"].append("b"),
            "auto_labels": lambda: server.state.update(auto_labels=["a"]),
            "truth": lambda: server.state["truth"].update({"1": [shape(77)]}),
            "reference_truth": lambda: server.state["truth"]["0"].append(shape(2)),
            "references": lambda: server.state["reference_frames"].update(camera0=[0, 2]),
            "candidate": lambda: server.state["candidates"].update({"1": [shape(88)]}),
            "run": lambda: server.state["agent_runs"].update({"1": {"manual": True}}),
        }
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        for name, mutate in mutations.items():
            with self.subTest(change=name), tempfile.TemporaryDirectory() as directory:
                entered, release = threading.Event(), threading.Event()
                responses = []
                server.WORK = Path(directory)
                server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                    "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                    "config": {"llm_key": "captured-key"}, "candidates": {}, "agent_runs": {}}
                server.run_status = {"status": "idle"}
                description = {"description": "a only", "prompts": [{"label": "a", "text": "a"}],
                               "reference_signature": server.reference_signature(0)}
                def blocked_run(config, *args, **kwargs):
                    entered.set()
                    self.assertTrue(release.wait(4))
                    callback = workflow._trace_callback.get()
                    callback({"id": "snapshot", "message": "captured-key"})
                    self.assertEqual(config["llm_key"], "captured-key")
                    return {"shapes": [shape()], "candidates_added": 1, "applied": [],
                        "review_queue": [], "agent": {"stop_reason": "done", "rounds": 1}}
                def start_run():
                    with server.app.test_client() as client:
                        responses.append(client.post("/api/run", json={"frame": 1, "count": 1,
                                                                          "description": description}))
                worker = threading.Thread(target=start_run)
                try:
                    with patch.object(server, "run", side_effect=blocked_run):
                        worker.start()
                        self.assertTrue(entered.wait(2))
                        with server.lock:
                            mutate()
                            expected_candidates = copy.deepcopy(server.state["candidates"])
                            expected_runs = copy.deepcopy(server.state["agent_runs"])
                            expected_truth = copy.deepcopy(server.state["truth"])
                        release.set()
                        worker.join(5)
                    self.assertFalse(worker.is_alive())
                    self.assertEqual(responses[0].status_code, 409, responses[0].json)
                    self.assertEqual(server.state["candidates"], expected_candidates)
                    self.assertEqual(server.state["agent_runs"], expected_runs)
                    self.assertEqual(server.state["truth"], expected_truth)
                    trace = json.loads((server.WORK / "model_debug.json").read_text())["trace"][0]
                    self.assertEqual(trace["topic"], "camera0")
                    self.assertEqual(trace["message"], "[REDACTED]")
                finally:
                    release.set()
                    if worker.ident is not None:
                        worker.join(5)
                    server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_state_read_waits_for_atomic_mutation_even_without_pending_reviews(self):
        old_state = server.state
        server.state = {"video": None, "topic": None, "labels": [], "truth": {},
                        "reference_frames": {}, "config": {}, "candidates": {},
                        "agent_runs": {}, "human_review_transactions": {"done": {"status": "committed"}}}
        acquired, release, finished = threading.Event(), threading.Event(), threading.Event()
        responses = []
        def hold_lock():
            with server.lock:
                acquired.set()
                release.wait(5)
        def read_state():
            try:
                with server.app.test_request_context("/api/state"):
                    responses.append(server.get_state().status_code)
            finally:
                finished.set()
        holder = threading.Thread(target=hold_lock)
        reader = threading.Thread(target=read_state)
        holder.start()
        try:
            self.assertTrue(acquired.wait(2))
            reader.start()
            self.assertFalse(finished.wait(0.1), "state read escaped the mutation lock")
            release.set()
            self.assertTrue(finished.wait(2), "state read did not resume after mutation")
            self.assertEqual(responses, [200])
        finally:
            release.set()
            holder.join(5)
            if reader.ident is not None:
                reader.join(5)
            server.state = old_state

    def test_state_summary_is_small_and_preserves_guide_decisions(self):
        old_state, old_work = server.state, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {
                "video": FakeVideo(), "topic": "camera0", "labels": ["person", "slope"],
                "truth": {"2": [shape()], "10": [shape()]},
                "reference_frames": {"camera0": [2]}, "config": {"llm_key": "secret"},
                "candidates": {"3": [shape(), shape()], "8": []}, "agent_runs": {},
                "human_review_transactions": {},
            }
            try:
                with patch.object(server, "current_truth_review_status", return_value={
                        "2": {"reusable_as_reference": True},
                        "10": {"reusable_as_reference": False}}):
                    response = server.app.test_client().get("/api/state/summary")
                self.assertEqual(response.status_code, 200)
                payload = response.json
                self.assertEqual(payload["schema_version"], "labelagent.state-summary.v1")
                self.assertEqual(payload["truth_frames"], ["2", "10"])
                self.assertEqual(payload["verified_truth_frames"], ["2"])
                self.assertEqual(payload["candidate_counts"], {"3": 2, "8": 0})
                self.assertEqual(payload["reference_frames"], {"camera0": [2]})
                self.assertNotIn("truth", payload)
                self.assertNotIn("candidates", payload)
                self.assertNotIn("agent_runs", payload)
                self.assertNotIn("config", payload)
                self.assertLess(len(response.data), 1024)
            finally:
                server.state, server.WORK = old_state, old_work

    def test_reference_preparation_indexes_frames_without_polygon_geometry(self):
        old_state, old_work = server.state, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            dense = [shape(index + 1, "pallet" if index % 2 else "floor")
                     for index in range(240)]
            server.state = {
                "video": FakeVideo(), "topic": "camera0", "labels": ["floor", "pallet"],
                "truth": {"2": [shape(label="floor")], "10": [shape(label="pallet")]},
                "reference_frames": {"camera0": [2]}, "config": {"llm_key": "secret"},
                "candidates": {"3": dense, "8": []}, "agent_runs": {"3": {"rounds": 4}},
                "human_review_transactions": {},
            }
            try:
                with patch.object(server, "current_truth_review_status", return_value={
                        "2": {"reusable_as_reference": True, "truth_sha256": "a" * 64,
                              "legacy_source_counts": {"manual": 1}},
                        "10": {"reusable_as_reference": False,
                               "reason_code": "missing_current_human_review_receipt",
                               "legacy_source_counts": {"auto": 1}}}):
                    response = server.app.test_client().get(
                        "/api/reference-preparation", query_string={
                            "file_activation_generation":
                                server.current_file_activation_generation()})
                self.assertEqual(response.status_code, 200, response.json)
                payload = response.json
                self.assertEqual(payload["schema_version"], "labelagent.reference-preparation.v1")
                self.assertTrue(payload["compact_geometry"])
                self.assertEqual(payload["file_activation_generation"],
                                 server.current_file_activation_generation())
                self.assertEqual(payload["frame_count"], 3)
                self.assertEqual(payload["auto_labels"], ["floor"])
                self.assertEqual(payload["truth_index"], {
                    "2": {"count": 1, "labels": ["floor"]},
                    "10": {"count": 1, "labels": ["pallet"]}})
                self.assertEqual(payload["candidate_index"]["3"], {
                    "count": 240, "labels": ["floor", "pallet"]})
                self.assertNotIn("8", payload["candidate_index"])
                self.assertEqual(payload["truth_review_status"]["2"]["truth_sha256"], "a" * 64)
                self.assertNotIn("truth", payload)
                self.assertNotIn("candidates", payload)
                self.assertNotIn("agent_runs", payload)
                self.assertNotIn("config", payload)
                self.assertNotIn("points", response.get_data(as_text=True))
                self.assertLess(len(response.data), 2048)
                stale = server.app.test_client().get(
                    "/api/reference-preparation", query_string={
                        "file_activation_generation":
                            server.current_file_activation_generation() - 1})
                self.assertEqual(stale.status_code, 409)
            finally:
                server.state, server.WORK = old_state, old_work

    def test_batch_review_index_preserves_browser_seed_without_geometry(self):
        old_state, old_work = server.state, server.WORK
        old_cache = server.batch_review_index_cache
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.batch_review_index_cache = None
            points = [[1.0, 2], [3.5, 4], [5, 6]]
            candidate = {"id": 1, "label": "斜坡", "score": 0.75,
                         "points": points, "holes": [],
                         "parts": [{"points": points, "holes": []}],
                         "dataset_split": "test24"}
            server.state = {
                "video": FakeVideo(), "topic": "camera0", "labels": ["斜坡"],
                "truth": {}, "reference_frames": {},
                "config": {"dataset_split": "test24", "llm_key": "secret"},
                "candidates": {"3": [candidate], "4": []},
                "agent_runs": {"3": {"review_queue": [{"label": "斜坡"}],
                                      "stop_reason": "needs_review", "split": "test24"}},
                "candidate_review_status": {
                    "camera0:3": {"frame": 3, "topic": "camera0",
                                  "remediation_required": True, "private": "hidden"}},
                "human_review_transactions": {},
            }
            try:
                client = server.app.test_client()
                with patch.object(server, "_candidate_browser_seed",
                                  wraps=server._candidate_browser_seed) as browser_seed:
                    response = client.get("/api/review/index")
                    warm_response = client.get("/api/review/index")
                    self.assertEqual(browser_seed.call_count, 1)
                    self.assertEqual(warm_response.data, response.data)
                    server.save()
                    self.assertIsNone(server.batch_review_index_cache)
                    refreshed_response = client.get("/api/review/index")
                    self.assertEqual(browser_seed.call_count, 2)
                    self.assertEqual(refreshed_response.data, response.data)
                self.assertEqual(response.status_code, 200, response.json)
                payload = response.json
                self.assertEqual(payload["schema_version"], "labelagent.batch-review-index.v1")
                self.assertTrue(payload["compact_geometry"])
                summary = payload["candidates"]["3"][0]
                self.assertEqual(summary["review_seed"], "877258d2")
                self.assertEqual(summary["dataset_split"], "test24")
                self.assertNotIn("points", summary)
                self.assertNotIn("parts", summary)
                self.assertNotIn("holes", summary)
                self.assertEqual(payload["agent_runs"]["3"]["review_queue"], [{}])
                self.assertEqual(payload["candidate_review_status"]["camera0:3"], {
                    "frame": 3, "topic": "camera0", "remediation_required": True})
                self.assertNotIn("llm_key", response.get_data(as_text=True))
                self.assertNotIn("private", response.get_data(as_text=True))
                self.assertLess(len(response.data), 2048)
            finally:
                server.state, server.WORK = old_state, old_work
                server.batch_review_index_cache = old_cache

    def test_reference_frame_rejects_stale_file_and_topic_scope(self):
        old_state, old_work = server.state, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {
                "video": FakeVideo(), "topic": "camera0", "labels": ["floor"],
                "truth": {"2": [shape(label="floor")]}, "reference_frames": {},
                "config": {}, "candidates": {}, "agent_runs": {},
                "human_review_transactions": {},
            }
            try:
                client = server.app.test_client()
                generation = server.current_file_activation_generation()
                stale_generation = client.post("/api/reference_frame", json={
                    "frame": 2, "expected_topic": "camera0",
                    "expected_path": str(FakeVideo.path),
                    "expected_file_activation_generation": generation - 1})
                self.assertEqual(stale_generation.status_code, 409)
                wrong_topic = client.post("/api/reference_frame", json={
                    "frame": 2, "expected_topic": "camera1",
                    "expected_path": str(FakeVideo.path),
                    "expected_file_activation_generation": generation})
                self.assertEqual(wrong_topic.status_code, 409)
                wrong_file = client.post("/api/reference_frame", json={
                    "frame": 2, "expected_topic": "camera0",
                    "expected_path": "C:/stale.mcap",
                    "expected_file_activation_generation": generation})
                self.assertEqual(wrong_file.status_code, 409)
                self.assertEqual(server.state["reference_frames"], {})
                with patch.object(server, "current_truth_review_status", return_value={
                        "2": {"reusable_as_reference": True}}):
                    valid = client.post("/api/reference_frame", json={
                        "frame": 2, "expected_topic": "camera0",
                        "expected_path": str(FakeVideo.path),
                        "expected_file_activation_generation": generation})
                self.assertEqual(valid.status_code, 200, valid.json)
                self.assertEqual(server.state["reference_frames"], {"camera0": [2]})
            finally:
                server.state, server.WORK = old_state, old_work

    def test_workbench_bootstrap_preserves_metadata_without_all_frame_geometry(self):
        old_state, old_work = server.state, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            dense = [shape(index + 1) for index in range(240)]
            server.state = {
                "video": FakeVideo(), "topic": "camera0", "labels": ["a", "slope"],
                "truth": {"0": [shape()], "2": [shape(2)]},
                "reference_frames": {}, "config": {"llm_model": "test", "llm_key": "secret"},
                "candidates": {"1": dense, "2": [shape(3)]},
                "agent_runs": {"1": {"rounds": 2}},
                "human_review_transactions": {},
            }
            try:
                client = server.app.test_client()
                full = client.get("/api/state")
                bootstrap = client.get("/api/bootstrap")
                self.assertEqual(bootstrap.status_code, 200, bootstrap.json)
                payload = bootstrap.json
                self.assertEqual(payload["schema_version"], "labelagent.workbench-bootstrap.v1")
                self.assertTrue(payload["compact_geometry"])
                self.assertTrue(payload["loaded"])
                self.assertEqual(payload["path"], str(FakeVideo.path))
                self.assertEqual(payload["topics"], FakeVideo.topics)
                self.assertEqual(payload["truth_frames"], ["0", "2"])
                self.assertEqual(payload["candidate_counts"], {"1": 240, "2": 1})
                self.assertEqual(payload["truth"], {})
                self.assertEqual(payload["candidates"], {})
                self.assertEqual(payload["agent_runs"], {})
                self.assertNotIn("llm_key", payload["config"])
                self.assertLess(len(bootstrap.data), len(full.data) * 0.05)
            finally:
                server.state, server.WORK = old_state, old_work

    def test_reference_draft_inference_allows_state_reads_and_guards_commit(self):
        old_state, old_status, old_work, old_stop = (server.state, server.run_status,
                                                     server.WORK, server.run_stop_event)
        cases = [(stage, "success") for stage in ("discover", "describe", "run", "audit")]
        cases += [("run", change) for change in ("video", "topic", "config", "labels",
                  "candidates", "agent_runs", "truth", "reference_frames", "stop")]
        try:
            for stage, change in cases:
                with self.subTest(stage=stage, change=change), tempfile.TemporaryDirectory() as directory:
                    server.WORK = Path(directory)
                    server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["person"],
                        "truth": {}, "reference_frames": {}, "config": {"llm_model": "test"},
                        "candidates": {}, "agent_runs": {}}
                    server.run_status = {"status": "idle"}
                    server.run_stop_event = threading.Event()
                    entered, release, read_finished = threading.Event(), threading.Event(), threading.Event()
                    responses, reads = [], []
                    discovery = {"visible_labels": ["person"], "uncertain_labels": [],
                                 "unknown_candidates": []}
                    description = {"description": "person", "prompts": []}
                    result = {"shapes": [shape(label="person")], "candidates_added": 1,
                              "applied": [], "review_queue": [],
                              "agent": {"stop_reason": "done", "rounds": 1, "audit": []}}
                    def block(name, value):
                        def invoke(*args, **kwargs):
                            if name == stage:
                                entered.set()
                                if not release.wait(5):
                                    raise RuntimeError("测试推理等待超时")
                            return value
                        return invoke
                    def start():
                        body = ({"frame": 1, "labels": ["person"]}
                                if stage == "describe"
                                else {"frame": 1, "auto_detect_visible": True})
                        responses.append(server.app.test_client().post(
                            "/api/reference_draft", json=body))
                    def read_state():
                        try:
                            reads.append(server.app.test_client().get("/api/state"))
                        finally:
                            read_finished.set()
                    worker, reader = threading.Thread(target=start), threading.Thread(target=read_state)
                    try:
                        with patch.object(server, "discover_reference_labels", side_effect=block("discover", (discovery, 0))), \
                             patch.object(server, "describe", side_effect=block("describe", description)), \
                             patch.object(server, "run", side_effect=block("run", result)), \
                             patch.object(server, "audit_high_risk_candidates", side_effect=block("audit", ([], []))):
                            worker.start()
                            self.assertTrue(entered.wait(3), "未进入模拟推理")
                            reader.start()
                            self.assertTrue(read_finished.wait(3), "推理期间状态读取被阻塞")
                            self.assertEqual(reads[0].status_code, 200)
                            with server.lock:
                                if change == "video":
                                    server.state["video"] = FakeVideo()
                                elif change == "topic":
                                    server.state["topic"] = "camera1"
                                elif change == "config":
                                    server.state["config"]["llm_model"] = "changed"
                                elif change == "labels":
                                    server.state["labels"].append("other")
                                elif change in ("candidates", "agent_runs", "truth"):
                                    server.state[change]["1"] = {"changed": True} if change == "agent_runs" else [{"changed": True}]
                                elif change == "reference_frames":
                                    server.state[change]["camera0"] = [1]
                            if change == "stop":
                                self.assertTrue(server.app.test_client().post("/api/run/stop", json={
                                    "expected_task_id": server.run_status["task_id"]}).json["stopped"])
                            before_candidates = copy.deepcopy(server.state["candidates"])
                            before_runs = copy.deepcopy(server.state["agent_runs"])
                            release.set()
                            worker.join(5)
                            self.assertFalse(worker.is_alive())
                        self.assertEqual(len(responses), 1)
                        if change == "success":
                            self.assertEqual(responses[0].status_code, 200, responses[0].json)
                            self.assertIn("1", server.state["candidates"])
                        else:
                            self.assertEqual(server.state["candidates"], before_candidates)
                            self.assertEqual(server.state["agent_runs"], before_runs)
                            if change == "stop":
                                self.assertTrue(responses[0].json["stopped"])
                                self.assertEqual(server.run_status["status"], "stopped")
                            else:
                                self.assertEqual(responses[0].status_code, 409, responses[0].json)
                                self.assertEqual(server.run_status["status"], "failed")
                    finally:
                        release.set()
                        if worker.ident is not None:
                            worker.join(5)
                        if reader.ident is not None:
                            reader.join(5)
        finally:
            server.state, server.run_status, server.WORK, server.run_stop_event = (
                old_state, old_status, old_work, old_stop)

    def test_state_read_recovers_pending_review_under_lock(self):
        old_state, old_work = server.state, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": None, "topic": None, "labels": [], "truth": {},
                            "reference_frames": {}, "config": {}, "candidates": {},
                            "agent_runs": {}, "human_review_transactions": {}}
            try:
                transaction = {"status": "state_saved_pending_audit"}
                server.state["human_review_transactions"]["pending"] = transaction
                audit_folder = server.WORK / "human_reviews"
                audit_folder.mkdir()
                (audit_folder / "record-pending.json").write_text("{}", encoding="utf-8")
                original_save = server.save
                def check_save():
                    self.assertTrue(server.lock._is_owned())
                    original_save()
                with patch.object(server, "save", side_effect=check_save) as saved:
                    response = server.app.test_client().get("/api/state")
                self.assertEqual(response.status_code, 200)
                saved.assert_called_once()
                self.assertEqual(transaction["status"], "committed")
                self.assertEqual(transaction["audit_file"], "record-pending.json")
                persisted = json.loads((server.WORK / "state.json").read_text(encoding="utf-8"))
                self.assertEqual(persisted["human_review_transactions"]["pending"], transaction)
                (audit_folder / "record-pending.json").unlink()
                (server.WORK / "state.json").unlink()
            finally:
                server.state, server.WORK = old_state, old_work

    def test_visibility_queue_preserves_conflict_even_with_candidate(self):
        flag = {"label": "other_robot", "reason_code": "vlm_cross_label_entity_conflict",
                "route": "human_review_required", "conflicting_labels": ["车辆", "other_robot"],
                "shared_anchor": [700, 450]}
        discovery = {"uncertain_labels": ["other_robot"], "consistency_flags": [flag]}
        queue = server.reference_visibility_review_queue(discovery, [shape(label="other_robot")])
        conflicts = [item for item in queue if item["reason_code"] == flag["reason_code"]]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(len(queue), 1)
        self.assertTrue(conflicts[0]["candidate_available"])
        self.assertEqual(conflicts[0]["evidence"], flag)
        conflicts[0]["evidence"]["shared_anchor"][0] = 0
        self.assertEqual(flag["shared_anchor"], [700, 450])

    def test_point_segment_has_bounded_timeout_and_elapsed_receipt(self):
        old_state = server.state
        server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["person"],
            "truth": {}, "reference_frames": {}, "config": {}, "candidates": {}, "agent_runs": {}}
        payload = {"points": [[1, 1], [2, 1], [2, 2]], "holes": [],
                   "parts": [{"points": [[1, 1], [2, 1], [2, 2]], "holes": []}]}
        try:
            with patch.object(server.urllib.request, "urlopen",
                              return_value=io.BytesIO(json.dumps(payload).encode())) as opened:
                response = server.app.test_client().post("/api/segment_points", json={
                    **active_request_scope(), "frame": 1,
                    "points": [{"x": 1, "y": 1, "label": 1}]})
            self.assertEqual(response.status_code, 200, response.json)
            self.assertGreaterEqual(response.json["inference_elapsed_seconds"], 0)
            self.assertEqual(response.json["path"], str(FakeVideo.path))
            self.assertEqual(response.json["topic"], "camera0")
            self.assertEqual(response.json["frame"], 1)
            self.assertEqual(response.json["file_activation_generation"],
                             server.current_file_activation_generation())
            self.assertEqual(opened.call_args.kwargs["timeout"], 180)
        finally:
            server.state = old_state

    def test_interactive_edits_reject_stale_activation_before_frame_read(self):
        old_state = server.state
        server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["person"],
            "truth": {}, "reference_frames": {}, "config": {}, "candidates": {},
            "agent_runs": {}}
        stale_scope = active_request_scope()
        stale_scope["file_activation_generation"] -= 1
        try:
            with patch.object(server, "frame_image", wraps=server.frame_image) as frame_reader, \
                 patch.object(server.urllib.request, "urlopen") as opened:
                point = server.app.test_client().post("/api/segment_points", json={
                    **stale_scope, "frame": 1,
                    "points": [{"x": 1, "y": 1, "label": 1}]})
                region = server.app.test_client().post("/api/edit_region", json={
                    **stale_scope, "frame": 1, "operation": "add",
                    "polygon": [[1, 1], [5, 1], [5, 5]], "shape": shape()})
            self.assertEqual(point.status_code, 409, point.json)
            self.assertEqual(region.status_code, 409, region.json)
            self.assertEqual(frame_reader.call_count, 0)
            opened.assert_not_called()
        finally:
            server.state = old_state

    def test_point_segment_rejects_completion_after_same_file_reload(self):
        old_state = server.state
        old_generation = server.file_activation_generation
        server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["person"],
            "truth": {}, "reference_frames": {}, "config": {}, "candidates": {},
            "agent_runs": {}}
        entered, release = threading.Event(), threading.Event()
        response_holder = []
        payload = {"points": [[1, 1], [2, 1], [2, 2]], "holes": [],
                   "parts": [{"points": [[1, 1], [2, 1], [2, 2]], "holes": []}]}

        class BlockingResponse(io.BytesIO):
            def __enter__(self):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("测试未释放 SAM 响应")
                return self

            def __exit__(self, *args):
                self.close()

        request_scope = active_request_scope()
        def post_point():
            with server.app.test_client() as client:
                response_holder.append(client.post("/api/segment_points", json={
                    **request_scope, "frame": 1,
                    "points": [{"x": 1, "y": 1, "label": 1}]}))
        worker = threading.Thread(target=post_point)
        try:
            with patch.object(server.urllib.request, "urlopen",
                              return_value=BlockingResponse(json.dumps(payload).encode())):
                worker.start()
                self.assertTrue(entered.wait(5))
                server.claim_file_activation()
                release.set()
                worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(response_holder), 1)
            self.assertEqual(response_holder[0].status_code, 409, response_holder[0].json)
            self.assertEqual(server.point_segment_flights, {})
        finally:
            release.set()
            if worker.ident is not None:
                worker.join(5)
            with server.file_activation_intent_lock:
                server.file_activation_generation = old_generation
            server.state = old_state

    def test_region_edit_returns_exact_activation_scope(self):
        old_state = server.state
        server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["person"],
            "truth": {}, "reference_frames": {}, "config": {}, "candidates": {},
            "agent_runs": {}}
        try:
            response = server.app.test_client().post("/api/edit_region", json={
                **active_request_scope(), "frame": 1, "operation": "add",
                "polygon": [[20, 20], [28, 20], [28, 28], [20, 28]],
                "shape": shape(label="person")})
            self.assertEqual(response.status_code, 200, response.json)
            self.assertEqual(response.json["path"], str(FakeVideo.path))
            self.assertEqual(response.json["topic"], "camera0")
            self.assertEqual(response.json["frame"], 1)
            self.assertEqual(response.json["file_activation_generation"],
                             server.current_file_activation_generation())
        finally:
            server.state = old_state

    def test_point_segment_rejects_duplicate_concurrent_request(self):
        old_state = server.state
        server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["person"],
            "truth": {}, "reference_frames": {}, "config": {}, "candidates": {}, "agent_runs": {}}
        self.assertTrue(server.point_segment_lock.acquire(blocking=False))
        try:
            response = server.app.test_client().post("/api/segment_points", json={
                **active_request_scope(), "frame": 1,
                "points": [{"x": 1, "y": 1, "label": 1}]})
            self.assertEqual(response.status_code, 503, response.json)
            self.assertIn("不同的点提示", response.json["error"])
        finally:
            server.point_segment_lock.release()
            server.state = old_state

    def test_point_segment_coalesces_only_identical_inflight_requests(self):
        old_state = server.state
        server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["person"],
            "truth": {}, "reference_frames": {}, "config": {}, "candidates": {},
            "agent_runs": {}}
        entered, release = threading.Event(), threading.Event()
        payload = {"points": [[1, 1], [2, 1], [2, 2]], "holes": [],
                   "parts": [{"points": [[1, 1], [2, 1], [2, 2]], "holes": []}]}

        class BlockingResponse(io.BytesIO):
            def __enter__(self):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("测试未释放 SAM 响应")
                return self

            def __exit__(self, *args):
                self.close()

        responses = []
        errors = []

        def post_point():
            try:
                with server.app.test_client() as client:
                    responses.append(client.post("/api/segment_points", json={
                        **active_request_scope(), "frame": 1,
                        "points": [{"x": 1, "y": 1, "label": 1}],
                    }))
            except BaseException as error:
                errors.append(error)

        try:
            with patch.object(server.urllib.request, "urlopen",
                              return_value=BlockingResponse(
                                  json.dumps(payload).encode())) as opened:
                first = threading.Thread(target=post_point)
                second = threading.Thread(target=post_point)
                first.start()
                self.assertTrue(entered.wait(5))
                second.start()
                time.sleep(0.05)
                release.set()
                first.join(5)
                second.join(5)
            self.assertFalse(errors)
            self.assertEqual(len(responses), 2)
            self.assertTrue(all(item.status_code == 200 for item in responses))
            opened.assert_called_once()
            roles = {item.json["request_coalescing"]["role"] for item in responses}
            self.assertEqual(roles, {"owner", "coalesced_waiter"})
            self.assertTrue(any(item.json["request_coalescing"]["coalesced"]
                                for item in responses))
            self.assertEqual(responses[0].json["parts"], responses[1].json["parts"])
            self.assertEqual(server.point_segment_flights, {})
        finally:
            release.set()
            server.state = old_state

    def test_point_segment_different_inflight_request_stays_busy(self):
        old_state = server.state
        server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["person"],
            "truth": {}, "reference_frames": {}, "config": {}, "candidates": {},
            "agent_runs": {}}
        entered, release = threading.Event(), threading.Event()
        payload = {"points": [[1, 1], [2, 1], [2, 2]], "holes": [],
                   "parts": [{"points": [[1, 1], [2, 1], [2, 2]], "holes": []}]}

        class BlockingResponse(io.BytesIO):
            def __enter__(self):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("测试未释放 SAM 响应")
                return self

            def __exit__(self, *args):
                self.close()

        first_response = []

        def post_first():
            with server.app.test_client() as client:
                first_response.append(client.post("/api/segment_points", json={
                    **active_request_scope(), "frame": 1,
                    "points": [{"x": 1, "y": 1, "label": 1}]}))

        try:
            with patch.object(server.urllib.request, "urlopen",
                              return_value=BlockingResponse(
                                  json.dumps(payload).encode())) as opened:
                first = threading.Thread(target=post_first)
                first.start()
                self.assertTrue(entered.wait(5))
                with server.app.test_client() as client:
                    different = client.post("/api/segment_points", json={
                        **active_request_scope(), "frame": 1,
                        "points": [{"x": 2, "y": 1, "label": 1}],
                    })
                self.assertEqual(different.status_code, 503, different.json)
                self.assertIn("已有推理任务", different.json["error"])
                release.set()
                first.join(5)
            self.assertEqual(len(first_response), 1)
            self.assertEqual(first_response[0].status_code, 200, first_response[0].json)
            opened.assert_called_once()
            self.assertEqual(server.point_segment_flights, {})
        finally:
            release.set()
            server.state = old_state

    def test_reference_draft_releases_lock_and_rejects_stale_commits(self):
        mutations = {
            "unchanged": None,
            "candidate": lambda: server.state["candidates"].update({"1": [shape(label="person")]}),
            "agent_run": lambda: server.state["agent_runs"].update({"1": {"manual": True}}),
            "topic": lambda: server.state.update(topic="camera1"),
            "video": lambda: server.state.update(video=FakeVideo()),
            "config": lambda: server.state["config"].update(llm_model="changed"),
            "credential": lambda: server.state["config"].update(llm_key="changed-key"),
            "labels": lambda: server.state["labels"].append("new-label"),
            "truth": lambda: server.state["truth"].update({"1": [shape(label="person")]}),
            "reference": lambda: server.state["reference_frames"].update(camera0=[1]),
            "audit_version": lambda: setattr(server, "_CANDIDATE_AUDIT_LOADED_CODE_SHA256", "changed"),
            "cancel": lambda: server.app.test_client().post("/api/run/stop", json={
                "expected_task_id": server.run_status["task_id"]}),
        }
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        old_stop = server.run_stop_event
        old_protocol = server._CANDIDATE_AUDIT_LOADED_CODE_SHA256
        try:
            for stage in ("describe", "audit"):
                for name, mutate in mutations.items():
                    with self.subTest(stage=stage, mutation=name), tempfile.TemporaryDirectory() as directory:
                        server.WORK = Path(directory)
                        server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["person"],
                            "truth": {}, "reference_frames": {}, "config": {"llm_model": "test-vlm"},
                            "candidates": {}, "agent_runs": {}}
                        server.run_status = {"status": "idle"}
                        server.run_stop_event = threading.Event()
                        server._CANDIDATE_AUDIT_LOADED_CODE_SHA256 = old_protocol
                        entered, release, finished = (threading.Event() for _ in range(3))
                        responses, errors = [], []
                        result = {"shapes": [shape(label="person")], "candidates_added": 1,
                            "applied": [], "review_queue": [], "agent": {"stop_reason": "done"}}
                        def block_model(*args, **kwargs):
                            entered.set()
                            if not release.wait(5):
                                raise RuntimeError("测试未释放模型")
                            return ({"description": "people", "prompts": []}
                                    if stage == "describe" else ([], []))
                        def draft():
                            try:
                                responses.append(server.app.test_client().post("/api/reference_draft",
                                    json={"frame": 1, "labels": ["person"]}))
                            except Exception as error:
                                errors.append(error)
                            finally:
                                finished.set()
                        worker = threading.Thread(target=draft)
                        with patch.object(server, "describe", side_effect=(block_model if stage == "describe"
                                else lambda *args, **kwargs: {"description": "people", "prompts": []})), \
                             patch.object(server, "run", return_value=result), \
                             patch.object(server, "audit_high_risk_candidates", side_effect=(block_model
                                if stage == "audit" else lambda *args, **kwargs: ([], []))):
                            worker.start()
                            try:
                                self.assertTrue(entered.wait(2), "模型未进入阻塞点")
                                state_done, state_responses = threading.Event(), []
                                def read_state():
                                    try:
                                        state_responses.append(server.app.test_client().get("/api/state"))
                                    finally:
                                        state_done.set()
                                reader = threading.Thread(target=read_state)
                                reader.start()
                                self.assertTrue(state_done.wait(2), "模型运行时状态接口被阻塞")
                                reader.join(2)
                                self.assertEqual(state_responses[0].status_code, 200)
                                self.assertEqual(state_responses[0].json["candidates"], {})
                                self.assertEqual(state_responses[0].json["agent_runs"], {})
                                duplicate = server.app.test_client().post("/api/reference_draft",
                                    json={"frame": 1, "labels": ["person"]})
                                self.assertEqual(duplicate.status_code, 503)
                                if mutate:
                                    with server.lock:
                                        mutate()
                                expected_candidates = copy.deepcopy(server.state["candidates"])
                                expected_runs = copy.deepcopy(server.state["agent_runs"])
                            finally:
                                release.set()
                                worker.join(5)
                            self.assertTrue(finished.is_set(), "草稿线程未退出")
                            self.assertEqual(errors, [])
                            response = responses[0]
                            if name == "unchanged":
                                self.assertEqual(response.status_code, 200, response.json)
                                self.assertEqual(server.run_status["status"], "completed")
                            elif name == "cancel":
                                self.assertEqual(response.status_code, 200, response.json)
                                self.assertTrue(response.json["stopped"])
                                self.assertEqual(server.run_status["status"], "stopped")
                            else:
                                self.assertEqual(response.status_code, 409, response.json)
                                self.assertTrue(response.json["stale_review"])
                            if name != "unchanged":
                                self.assertEqual(server.state["candidates"], expected_candidates)
                                self.assertEqual(server.state["agent_runs"], expected_runs)
                                self.assertFalse((server.WORK / "state.json").exists())
                                self.assertEqual(server.state.get("candidate_revisions", {}), {})
        finally:
            server.state, server.run_status, server.WORK = old_state, old_status, old_work
            server.run_stop_event = old_stop
            server._CANDIDATE_AUDIT_LOADED_CODE_SHA256 = old_protocol

    def test_reference_draft_records_candidate_only_provenance(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["person"],
                "truth": {}, "reference_frames": {}, "config": {"llm_model": "test-vlm"},
                "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            generated = shape(label="person")
            result = {"shapes": [generated], "candidates_added": 1, "applied": [],
                "review_queue": [], "agent": {"converged": False, "rounds": 1,
                "max_rounds": 100, "stop_reason": "no_progress", "audit": []}}
            try:
                with patch.object(server, "describe", return_value={"description": "people",
                        "prompts": [{"label": "person", "text": "person"}]}), \
                     patch.object(server, "run", return_value=result):
                    response = server.app.test_client().post("/api/reference_draft", json={
                        "frame": 1, "labels": ["person"]})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertTrue(response.json["candidate_only"])
                self.assertFalse(response.json["human_truth"])
                self.assertFalse(response.json["approved_for_training"])
                self.assertEqual(len(response.json["input_sha256"]), 64)
                self.assertEqual(len(response.json["candidate_event_id"]), 64)
                self.assertEqual(len(response.json["candidate_request_id"]), 64)
                self.assertEqual(len(response.json["candidate_output_sha256"]), 64)
                self.assertEqual(len(response.json["candidate_artifact_sha256"]), 64)
                candidate = server.state["candidates"]["1"][0]
                self.assertTrue(candidate["candidate_only"])
                self.assertFalse(candidate["human_truth"])
                self.assertFalse(candidate["approved_for_training"])
                self.assertEqual(candidate["provenance"]["pipeline"], "sam_vlm_agent")
                self.assertEqual(candidate["provenance"]["candidate_request_id"],
                                 response.json["candidate_request_id"])
                self.assertEqual(candidate["provenance"]["candidate_output_sha256"],
                                 response.json["candidate_output_sha256"])
                self.assertEqual(candidate["provenance"]["sam_prompts"], [
                    {"label": "person", "text": "person"}])
                self.assertEqual(server.state["truth"], {})
                self.assertEqual(server.state["reference_frames"], {})
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_reference_draft_blocks_low_storage_before_frame_or_model_work(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["person"],
                "truth": {}, "reference_frames": {}, "config": {"llm_model": "test-vlm"},
                "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            try:
                with patch.object(server.shutil, "disk_usage", return_value=SimpleNamespace(
                        total=1000, used=1000, free=0)), \
                     patch.object(server, "frame_image") as frame_image, \
                     patch.object(server, "run") as run_model:
                    response = server.app.test_client().post("/api/reference_draft", json={
                        "frame": 1, "labels": ["person"]})
                self.assertEqual(response.status_code, 507, response.json)
                self.assertEqual(response.json["code"], "storage_low")
                self.assertFalse(response.json["storage"]["ready"])
                self.assertTrue(response.json["candidate_only"])
                self.assertFalse(response.json["human_truth"])
                self.assertFalse(response.json["approved_for_training"])
                frame_image.assert_not_called()
                run_model.assert_not_called()
                self.assertEqual(server.state["candidates"], {})
                self.assertEqual(server.state["truth"], {})
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_reference_draft_rerun_archives_prior_candidates_and_audit_append_only(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            prior_candidate = shape(label="person", left=2, right=12)
            prior_candidates = [prior_candidate]
            prior_sha = server._sha256_json(prior_candidates)
            prior_run = {"candidate_event_id": "a" * 64,
                         "candidate_artifact_sha256": prior_sha,
                         "stop_reason": "done", "rounds": 1, "audit": [{"round": 1}]}
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["person"],
                "truth": {}, "reference_frames": {}, "config": {"llm_model": "test-vlm"},
                "candidates": {"1": copy.deepcopy(prior_candidates)},
                "agent_runs": {"1": copy.deepcopy(prior_run)}, "candidate_revisions": {}}
            server.run_status = {"status": "idle"}
            generated = shape(label="person", left=4, right=18)
            result = {"shapes": [generated], "candidates_added": 1, "applied": [],
                "review_queue": [], "agent": {"converged": False, "rounds": 1,
                "max_rounds": 100, "stop_reason": "no_progress", "audit": []}}
            try:
                with patch.object(server, "describe", return_value={"description": "people",
                        "prompts": [{"label": "person", "text": "person"}]}), \
                     patch.object(server, "run", return_value=result):
                    response = server.app.test_client().post("/api/reference_draft", json={
                        "frame": 1, "labels": ["person"]})
                self.assertEqual(response.status_code, 200, response.json)
                revisions = server.state["candidate_revisions"]["1"]
                self.assertEqual(len(revisions), 1)
                revision = revisions[0]
                self.assertEqual(response.json["prior_candidate_revision_id"], revision["revision_id"])
                self.assertEqual(response.json["candidate_revision_count"], 1)
                self.assertEqual(revision["prior_candidates"], prior_candidates)
                self.assertEqual(revision["prior_agent_run"], prior_run)
                self.assertEqual(revision["prior_candidate_artifact_sha256"], prior_sha)
                self.assertTrue(revision["prior_artifact_integrity_match"])
                self.assertTrue(revision["candidate_only"])
                self.assertFalse(revision["human_truth"])
                self.assertFalse(revision["approved_for_training"])
                self.assertNotEqual(response.json["candidate_event_id"],
                                    prior_run["candidate_event_id"])
                self.assertEqual(server.state["truth"], {})
                state_response = server.app.test_client().get("/api/state")
                summary = state_response.json["candidate_revision_summary"]["1"][0]
                self.assertEqual(summary["revision_id"], revision["revision_id"])
                self.assertNotIn("prior_candidates", summary)
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_reference_candidate_event_id_changes_with_output_not_stable_request(self):
        request_id = "a" * 64
        first = server.reference_candidate_event_id(request_id, "b" * 64)
        replay = server.reference_candidate_event_id(request_id, "b" * 64)
        changed = server.reference_candidate_event_id(request_id, "c" * 64)
        self.assertEqual(first, replay)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(first), 64)

    def test_reference_discovery_prefetch_warms_exact_request_without_state_writes(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        server.clear_reference_discovery_cache()
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0",
                "labels": ["person", "slope"], "truth": {}, "reference_frames": {},
                "config": {"llm_model": "test-vlm", "llm_busy_retries": 5},
                "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            discovery = {"visible_labels": ["person"], "uncertain_labels": ["slope"],
                         "unknown_candidates": []}
            try:
                warmed = {"prefetched": True, "prompts": {
                    "person": {"results": []}, "slope": {"results": []}},
                    "diagnostics": {"response_cache": {"hit": False,
                        "coalesced": False, "entries": 1}}}
                roi_warmed = {"prefetched": True, "results": [],
                              "diagnostics": {"response_cache": {"hit": False}}}
                with patch.object(server, "discover_reference_labels",
                                  return_value=(discovery, 0)) as discover_mock, \
                     patch.object(server, "sam_text_multi_prefetch",
                                  return_value=warmed) as sam_prefetch_mock, \
                     patch.object(server, "sam_text_prefetch",
                                  return_value=roi_warmed) as roi_prefetch_mock, \
                     patch.object(server, "save") as save_mock:
                    response = server.app.test_client().post(
                        "/api/reference_discovery/prefetch", json={
                            "frame": 1, "expected_path": str(FakeVideo.path),
                            "expected_topic": "camera0"})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(response.json["schema_version"],
                                 "labelagent.reference-discovery-prefetch.v1")
                self.assertTrue(response.json["prefetched"])
                self.assertTrue(response.json["candidate_only"])
                self.assertFalse(response.json["human_truth"])
                self.assertFalse(response.json["approved_for_training"])
                self.assertEqual(response.json["visible_labels"], ["person"])
                self.assertEqual(response.json["uncertain_labels"], ["slope"])
                self.assertTrue(response.json["sam_prefetch"]["prefetched"])
                self.assertTrue(response.json["sam_prefetch"]["parallel_with_discovery"])
                self.assertTrue(response.json["registry_sam_prefetch"]["prefetched"])
                self.assertTrue(response.json["registry_sam_prefetch"]
                                ["parallel_with_discovery"])
                self.assertEqual(response.json["sam_prefetch"]["selected_labels"],
                                 ["person", "slope"])
                self.assertEqual(response.json["sam_prefetch"]["cache_hit"], False)
                self.assertEqual(
                    response.json["sam_prefetch"]["slope_roi_prefetch"]["completed"], 4)
                self.assertEqual(len(response.json["input_sha256"]), 64)
                self.assertEqual(len(response.json["discovery_sha256"]), 64)
                self.assertTrue(response.json["server_discovery_cache_stored"])
                request_config, _, request_labels = discover_mock.call_args.args[:3]
                self.assertEqual(request_config["llm_busy_retries"], 0)
                self.assertEqual(request_labels, ["person", "slope"])
                self.assertEqual(server.state["config"]["llm_busy_retries"], 5)
                sam_args = sam_prefetch_mock.call_args.args
                self.assertEqual(sam_args[1], {"person": "person",
                                               "slope": "traversable surface"})
                self.assertEqual(sam_args[2]["llm_busy_retries"], 0)
                self.assertEqual(roi_prefetch_mock.call_count, 4)
                save_mock.assert_not_called()
                self.assertEqual(server.state["truth"], {})
                self.assertEqual(server.state["candidates"], {})
                stale = server.app.test_client().post(
                    "/api/reference_discovery/prefetch", json={
                        "frame": 1, "expected_path": "C:/stale.mcap",
                        "expected_topic": "camera0"})
                self.assertEqual(stale.status_code, 200)
                self.assertFalse(stale.json["prefetched"])
                self.assertEqual(stale.json["code"], "scope_changed")
                self.assertTrue(stale.json["stale_request_discarded"])
                self.assertFalse(stale.json["human_truth"])
                self.assertFalse(stale.json["approved_for_training"])
            finally:
                server.clear_reference_discovery_cache()
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_reference_discovery_prefetch_starts_registry_sam_while_vlm_is_running(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        server.clear_reference_discovery_cache()
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0",
                "labels": ["person", "slope"], "truth": {}, "reference_frames": {},
                "config": {"sam_multi_prompt": True}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            sam_started, allow_sam_finish = threading.Event(), threading.Event()
            warmed = {"prefetched": True, "prompts": {
                "person": {"results": []}, "slope": {"results": []}},
                "diagnostics": {"response_cache": {"hit": False,
                    "coalesced": False, "entries": 1}}}

            def blocked_sam(*_args, **_kwargs):
                sam_started.set()
                self.assertTrue(allow_sam_finish.wait(2))
                return warmed

            def concurrent_discovery(*_args, **_kwargs):
                self.assertTrue(sam_started.wait(1))
                allow_sam_finish.set()
                return ({"visible_labels": ["person"], "uncertain_labels": ["slope"],
                         "unknown_candidates": []}, 0)

            try:
                with patch.object(server, "sam_text_multi_prefetch",
                                  side_effect=blocked_sam) as sam_mock, \
                     patch.object(server, "discover_reference_labels",
                                  side_effect=concurrent_discovery):
                    response = server.app.test_client().post(
                        "/api/reference_discovery/prefetch", json={
                            "frame": 1, "expected_path": str(FakeVideo.path),
                            "expected_topic": "camera0"})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(sam_mock.call_count, 1)
                self.assertTrue(response.json["registry_sam_prefetch"]["prefetched"])
                self.assertTrue(response.json["sam_prefetch"]["parallel_with_discovery"])
                self.assertEqual(server.state["truth"], {})
                self.assertEqual(server.state["candidates"], {})
            finally:
                server.clear_reference_discovery_cache()
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_reference_discovery_prefetch_does_not_wait_for_slow_parallel_sam(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        server.clear_reference_discovery_cache()
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0",
                "labels": ["person", "slope"], "truth": {}, "reference_frames": {},
                "config": {"sam_multi_prompt": True}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            sam_started, allow_sam_finish, sam_finished = (
                threading.Event(), threading.Event(), threading.Event())
            warmed = {"prefetched": True, "prompts": {
                "person": {"results": []}, "slope": {"results": []}},
                "diagnostics": {"response_cache": {"hit": False,
                    "coalesced": False, "entries": 1}}}

            def blocked_sam(*_args, **_kwargs):
                sam_started.set()
                self.assertTrue(allow_sam_finish.wait(2))
                sam_finished.set()
                return warmed

            try:
                with patch.object(server, "sam_text_multi_prefetch",
                                  side_effect=blocked_sam) as sam_mock, \
                     patch.object(server, "discover_reference_labels",
                                  return_value=({"visible_labels": ["person"],
                                      "uncertain_labels": ["slope"],
                                      "unknown_candidates": []}, 0)):
                    started = time.perf_counter()
                    response = server.app.test_client().post(
                        "/api/reference_discovery/prefetch", json={
                            "frame": 1, "expected_path": str(FakeVideo.path),
                            "expected_topic": "camera0"})
                    elapsed = time.perf_counter() - started
                self.assertTrue(sam_started.is_set())
                self.assertLess(elapsed, 0.75)
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(sam_mock.call_count, 1)
                self.assertEqual(response.json["registry_sam_prefetch"]["code"],
                                 "warming")
                self.assertTrue(response.json["registry_sam_prefetch"]
                                ["deferred_completion"])
                self.assertEqual(response.json["sam_prefetch"]["code"],
                                 "registry_warming")
                self.assertEqual(server.state["truth"], {})
                self.assertEqual(server.state["candidates"], {})
            finally:
                allow_sam_finish.set()
                self.assertTrue(sam_finished.wait(1))
                server.clear_reference_discovery_cache()
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_reference_draft_reuses_exact_prefetched_discovery_without_second_vlm_call(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        server.clear_reference_discovery_cache()
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0",
                "labels": ["person"], "truth": {}, "reference_frames": {},
                "config": {"llm_model": "test-vlm", "llm_busy_retries": 5},
                "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            discovery = {"assessments": [{"label": "person", "evidence": "present",
                "confidence": .95, "reason": "visible"}],
                "visible_labels": ["person"], "uncertain_labels": [],
                "candidate_labels": ["person"], "consistency_flags": [],
                "unknown_candidates": [], "registry_conflicts": [],
                "open_set_scan_complete": True, "registry_mutated": False}
            result = {"shapes": [shape(label="person")], "candidates_added": 1,
                "applied": [], "review_queue": [],
                "agent": {"stop_reason": "done", "rounds": 0, "audit": []}}
            try:
                with patch.object(server, "discover_reference_labels",
                                  return_value=(discovery, 0)) as discover_mock, \
                     patch.object(server, "generate_only", return_value=result) as generate_mock:
                    prefetched = server.app.test_client().post(
                        "/api/reference_discovery/prefetch", json={
                            "frame": 1, "expected_path": str(FakeVideo.path),
                            "expected_topic": "camera0"})
                    response = server.app.test_client().post(
                        "/api/reference_draft", json={"frame": 1,
                            "auto_detect_visible": True,
                            "defer_agent_review": True})
                self.assertEqual(prefetched.status_code, 200, prefetched.json)
                self.assertEqual(response.status_code, 200, response.json)
                discover_mock.assert_called_once()
                generate_mock.assert_called_once()
                self.assertTrue(response.json["discovery_prefetch_cache_hit"])
                self.assertTrue(response.json["agent"]["discovery_prefetch_cache_hit"])
                reused = response.json["visible_label_discovery"]
                self.assertEqual(reused["assessments"], discovery["assessments"])
                self.assertEqual(reused["visible_labels"], discovery["visible_labels"])
                self.assertEqual(reused["uncertain_labels"], discovery["uncertain_labels"])
                self.assertEqual(reused["recall_probe_labels"], [])
                self.assertEqual(server.state["truth"], {})
                self.assertFalse(response.json["human_truth"])
                self.assertFalse(response.json["approved_for_training"])
            finally:
                server.clear_reference_discovery_cache()
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_reference_discovery_cache_requires_exact_binding_and_expires(self):
        server.clear_reference_discovery_cache()
        video = FakeVideo()
        base = server.reference_discovery_cache_key(
            video, "camera0", 1, "a" * 64, ["person"], "b" * 64,
            activation_generation=7)
        changed = server.reference_discovery_cache_key(
            video, "camera0", 1, "a" * 64, ["person", "slope"], "b" * 64,
            activation_generation=7)
        discovery = {"visible_labels": ["person"]}
        server.store_reference_discovery_cache(base, discovery, 0, now=100.0)
        self.assertEqual(server.get_reference_discovery_cache(base, now=101.0),
                         (discovery, 0))
        self.assertIsNone(server.get_reference_discovery_cache(changed, now=101.0))
        self.assertIsNone(server.get_reference_discovery_cache(
            base, now=100.0 + server.REFERENCE_DISCOVERY_CACHE_TTL_SECONDS + 1))
        server.clear_reference_discovery_cache()

    def test_reference_discovery_prefetch_does_not_queue_behind_busy_vlm(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["person"],
                "truth": {}, "reference_frames": {}, "config": {},
                "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            try:
                with patch.object(server, "discover_reference_labels", side_effect=ValueError(
                        "大模型请求失败（HTTP 503）：Local VLM is busy")), \
                     patch.object(server, "sam_text_multi_prefetch") as sam_prefetch_mock:
                    response = server.app.test_client().post(
                        "/api/reference_discovery/prefetch", json={
                            "frame": 1, "expected_path": str(FakeVideo.path),
                            "expected_topic": "camera0"})
                self.assertEqual(response.status_code, 202, response.json)
                self.assertFalse(response.json["prefetched"])
                self.assertEqual(response.json["code"], "vlm_busy")
                sam_prefetch_mock.assert_not_called()
                self.assertEqual(server.state["truth"], {})
                self.assertEqual(server.state["candidates"], {})
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_reference_discovery_prefetch_busy_vlm_does_not_wait_for_parallel_sam(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0",
                "labels": ["person", "slope"], "truth": {}, "reference_frames": {},
                "config": {"sam_multi_prompt": True}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            sam_started, allow_sam_finish, sam_finished = (
                threading.Event(), threading.Event(), threading.Event())

            def blocked_sam(*_args, **_kwargs):
                sam_started.set()
                self.assertTrue(allow_sam_finish.wait(2))
                sam_finished.set()
                return {"prefetched": True, "prompts": {}, "diagnostics": {}}

            try:
                with patch.object(server, "discover_reference_labels", side_effect=ValueError(
                        "大模型请求失败（HTTP 503）：Local VLM is busy")), \
                     patch.object(server, "sam_text_multi_prefetch",
                                  side_effect=blocked_sam) as sam_prefetch_mock:
                    started = time.perf_counter()
                    response = server.app.test_client().post(
                        "/api/reference_discovery/prefetch", json={
                            "frame": 1, "expected_path": str(FakeVideo.path),
                            "expected_topic": "camera0"})
                    elapsed = time.perf_counter() - started
                self.assertTrue(sam_started.is_set())
                self.assertLess(elapsed, 0.75)
                self.assertEqual(response.status_code, 202, response.json)
                self.assertFalse(response.json["prefetched"])
                self.assertEqual(response.json["code"], "vlm_busy")
                self.assertEqual(sam_prefetch_mock.call_count, 1)
                self.assertEqual(server.state["truth"], {})
                self.assertEqual(server.state["candidates"], {})
            finally:
                allow_sam_finish.set()
                self.assertTrue(sam_finished.wait(1))
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_reference_discovery_prefetch_scope_change_during_model_is_clean_noop(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0",
                "labels": ["person"], "truth": {}, "reference_frames": {},
                "config": {"llm_model": "test-vlm"},
                "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            try:
                def switch_scope(*_args, **_kwargs):
                    server.state["topic"] = "camera1"
                    return ({"visible_labels": ["person"],
                             "uncertain_labels": [], "unknown_candidates": []}, 0)

                with patch.object(server, "discover_reference_labels",
                                  side_effect=switch_scope), \
                     patch.object(server, "sam_text_multi_prefetch") as sam_prefetch_mock, \
                     patch.object(server, "save") as save_mock:
                    response = server.app.test_client().post(
                        "/api/reference_discovery/prefetch", json={
                            "frame": 1, "expected_path": str(FakeVideo.path),
                            "expected_topic": "camera0"})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertFalse(response.json["prefetched"])
                self.assertEqual(response.json["code"], "scope_changed")
                self.assertTrue(response.json["stale_request_discarded"])
                self.assertFalse(response.json["human_truth"])
                self.assertFalse(response.json["approved_for_training"])
                sam_prefetch_mock.assert_not_called()
                save_mock.assert_not_called()
                self.assertEqual(server.state["truth"], {})
                self.assertEqual(server.state["candidates"], {})
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_reference_draft_auto_discovers_visible_labels_before_sam(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0",
                "labels": ["person", "pallet", "slope"], "truth": {}, "reference_frames": {},
                "config": {"llm_model": "test-vlm"}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            discovery = {"assessments": [
                    {"label": "person", "evidence": "present", "confidence": .95, "reason": "visible"},
                    {"label": "pallet", "evidence": "absent", "confidence": .91, "reason": "absent"},
                    {"label": "slope", "evidence": "uncertain", "confidence": .51, "reason": "occluded"}],
                "visible_labels": ["person"], "uncertain_labels": ["slope"],
                "unknown_candidates": [{"candidate_name": "rack", "evidence": "visible rack",
                    "confidence": .7, "route": "human_review_required",
                    "ontology_mutation_allowed": False}], "registry_mutated": False}
            generated = shape(label="person")
            result = {"shapes": [generated], "candidates_added": 1, "applied": [],
                "review_queue": [], "agent": {"stop_reason": "done", "rounds": 1, "audit": []}}
            try:
                with patch.object(server, "discover_visible_labels", return_value=discovery) as discover_mock, \
                     patch.object(server, "describe", return_value={"description": "person",
                        "prompts": [{"label": "person", "text": "person"}]}) as describe_mock, \
                     patch.object(server, "run", return_value=result) as run_mock:
                    response = server.app.test_client().post("/api/reference_draft", json={
                        "frame": 1, "auto_detect_visible": True})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertTrue(response.json["auto_detect_visible"])
                self.assertEqual(response.json["labels"], ["person", "slope"])
                self.assertTrue(response.json["human_review_required"])
                self.assertEqual(response.json["visible_label_discovery"]["uncertain_labels"], ["slope"])
                self.assertFalse(response.json["visible_label_discovery"]["unknown_candidates"][0]
                                 ["ontology_mutation_allowed"])
                discover_mock.assert_called_once()
                self.assertEqual(discover_mock.call_args.args[2], ["person", "pallet", "slope"])
                describe_mock.assert_not_called()
                self.assertEqual(run_mock.call_args.args[3], ["person", "slope"])
                self.assertEqual(run_mock.call_args.args[2]["prompt_source"],
                                 "deterministic_canonical_fallback")
                self.assertEqual(response.json["description_source"],
                                 "deterministic_after_visible_label_discovery")
                queue = server.state["agent_runs"]["1"]["review_queue"]
                self.assertIn({
                    "label": "slope", "reason_code": "vlm_visibility_uncertain",
                    "route": "human_review_required", "source": "visible_label_discovery",
                    "candidate_available": False}, queue)
                self.assertTrue(any(item["reason_code"] == "vlm_unknown_candidate" for item in queue))
                class_review = next(item for item in queue
                                    if item.get("source") == "candidate_class_audit")
                self.assertEqual(class_review["shape_ids"], [generated["id"]])
                self.assertEqual(class_review["candidate_artifact_sha256"],
                                 response.json["candidate_artifact_sha256"])
                self.assertEqual(server.state["truth"], {})
                self.assertEqual(server.state["reference_frames"], {})
                self.assertTrue(server.state["candidates"]["1"][0]["candidate_only"])
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_reference_draft_auto_detection_without_visible_known_label_fails_closed(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["person"],
                "truth": {}, "reference_frames": {}, "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            discovery = {"assessments": [{"label": "person", "evidence": "absent",
                "confidence": .9, "reason": "absent"}], "visible_labels": [], "uncertain_labels": [],
                "unknown_candidates": [], "registry_mutated": False}
            try:
                with patch.object(server, "discover_visible_labels", return_value=discovery), \
                     patch.object(server, "describe") as describe_mock, patch.object(server, "run") as run_mock:
                    response = server.app.test_client().post("/api/reference_draft", json={
                        "frame": 1, "auto_detect_visible": True})
                self.assertEqual(response.status_code, 400, response.json)
                self.assertIn("未发现可生成候选的已知类别", response.json["error"])
                describe_mock.assert_not_called(); run_mock.assert_not_called()
                self.assertEqual(server.state["candidates"], {})
                self.assertEqual(server.state["truth"], {})
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_reference_draft_coverage_probes_missing_selected_classes(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0",
                "labels": ["person", "pallet", "slope"], "truth": {},
                "reference_frames": {}, "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            discovery = {"assessments": [], "visible_labels": ["person"],
                "uncertain_labels": [], "candidate_labels": ["person"],
                "unknown_candidates": [], "registry_mutated": False}
            result = {"shapes": [shape(label="person")], "candidates_added": 1,
                "applied": [], "review_queue": [],
                "agent": {"stop_reason": "done", "rounds": 0, "audit": []}}
            try:
                with patch.object(server, "discover_visible_labels", return_value=discovery), \
                     patch.object(server, "generate_only", return_value=result) as generate_mock:
                    response = server.app.test_client().post("/api/reference_draft", json={
                        "frame": 1, "auto_detect_visible": True,
                        "probe_missing_selected": True, "defer_agent_review": True,
                        "defer_class_audit": True})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(response.json["labels"], ["person", "pallet", "slope"])
                self.assertTrue(response.json["probe_missing_selected"])
                self.assertEqual(response.json["visible_label_discovery"]["recall_probe_labels"],
                                 ["pallet", "slope"])
                self.assertEqual(generate_mock.call_args.args[3], ["person", "pallet", "slope"])
                provenance = response.json["candidates"][0]["provenance"]
                self.assertTrue(provenance["probe_missing_selected"])
                self.assertFalse(response.json["human_truth"])
                self.assertEqual(server.state["truth"], {})
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_reference_draft_all_selected_uses_deterministic_prompts_without_vlm_description(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            labels = ["person", "pallet", "slope"]
            server.state = {"video": FakeVideo(), "topic": "camera0",
                "labels": labels, "truth": {}, "reference_frames": {}, "config": {},
                "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            generated = shape(label="person")
            result = {"shapes": [generated], "candidates_added": 1,
                "applied": [], "review_queue": [],
                "agent": {"stop_reason": "done", "rounds": 0, "audit": []}}
            try:
                with patch.object(server, "describe") as describe_mock, \
                     patch.object(server, "generate_only", return_value=result) as generate_mock:
                    response = server.app.test_client().post("/api/reference_draft", json={
                        "frame": 1, "labels": labels,
                        "deterministic_description": True,
                        "defer_agent_review": True, "defer_class_audit": True})
                self.assertEqual(response.status_code, 200, response.json)
                describe_mock.assert_not_called()
                self.assertEqual(generate_mock.call_args.args[2],
                                 workflow.canonical_unreferenced_description(labels))
                self.assertEqual(response.json["description_source"],
                                 "deterministic_all_selected")
                self.assertTrue(response.json["deterministic_description"])
                self.assertTrue(response.json["agent"]["deterministic_description"])
                provenance = response.json["candidates"][0]["provenance"]
                self.assertTrue(provenance["deterministic_description"])
                self.assertFalse(response.json["human_truth"])
                self.assertEqual(server.state["truth"], {})
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_reference_draft_rejects_deterministic_prompts_for_partial_label_list(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0",
                "labels": ["person", "pallet"], "truth": {}, "reference_frames": {},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            try:
                with patch.object(server, "describe") as describe_mock, \
                     patch.object(server, "generate_only") as generate_mock:
                    response = server.app.test_client().post("/api/reference_draft", json={
                        "frame": 1, "labels": ["person"],
                        "deterministic_description": True,
                        "defer_agent_review": True})
                self.assertEqual(response.status_code, 400, response.json)
                self.assertIn("完整的当前标签表", response.json["error"])
                describe_mock.assert_not_called()
                generate_mock.assert_not_called()
                self.assertEqual(server.state["candidates"], {})
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_stop_endpoint_terminates_active_run_without_saving_partial_frame(self):
        old_state, old_status, old_work, old_stop = server.state, server.run_status, server.WORK, server.run_stop_event
        entered = threading.Event()
        responses = []
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            server.run_stop_event = threading.Event()
            description = {"description": "a only", "prompts": [{"label": "a", "text": "square"}],
                           "reference_signature": server.reference_signature(0)}
            def fake_run(*args, **kwargs):
                self.assertEqual(kwargs["max_rounds"], 5)
                entered.set()
                server.run_stop_event.wait(3)
                workflow.check_cancelled()
                raise AssertionError("run was not cancelled")
            def start():
                with server.app.test_client() as client:
                    responses.append(client.post("/api/run", json={"frame": 1, "count": 1,
                        "max_rounds": 5, "description": description}))
            worker = threading.Thread(target=start)
            try:
                with patch.object(server, "run", side_effect=fake_run):
                    worker.start()
                    self.assertTrue(entered.wait(2))
                    client = server.app.test_client()
                    task_id = server.run_status["task_id"]
                    stopped = client.post("/api/run/stop", json={"expected_task_id": task_id})
                    self.assertEqual(stopped.json, {"status": "stopping", "stopped": True,
                                                    "task_id": task_id})
                    worker.join(3)
                    self.assertFalse(worker.is_alive())
                self.assertEqual(responses[0].status_code, 200)
                self.assertTrue(responses[0].json["stopped"])
                self.assertEqual(client.get("/api/run/status").json["status"], "stopped")
                self.assertFalse(server.state["candidates"])
                self.assertEqual(server.state["truth"], {"0": [shape()]})
                self.assertFalse(client.post("/api/run/stop", json={
                    "expected_task_id": task_id}).json["stopped"])
            finally:
                server.run_stop_event.set()
                worker.join(3)
                server.state, server.run_status, server.WORK, server.run_stop_event = old_state, old_status, old_work, old_stop

    def test_stop_endpoint_rejects_missing_or_stale_task_identity_without_stopping_active_run(self):
        old_status, old_task, old_stop = server.run_status, server.run_task_id, server.run_stop_event
        try:
            server.run_status = {"status": "running", "phase": "generating",
                                 "task_id": "new-active-task"}
            server.run_task_id = "new-active-task"
            server.run_stop_event = threading.Event()
            client = server.app.test_client()

            missing = client.post("/api/run/stop", json={})
            self.assertEqual(missing.status_code, 400)
            self.assertIn("缺少任务编号", missing.json["error"])
            self.assertFalse(server.run_stop_event.is_set())
            self.assertEqual(server.run_status["status"], "running")

            stale = client.post("/api/run/stop", json={
                "expected_task_id": "old-stale-task"})
            self.assertEqual(stale.status_code, 409)
            self.assertTrue(stale.json["stale_review"])
            self.assertIn("未停止当前新任务", stale.json["error"])
            self.assertFalse(server.run_stop_event.is_set())
            self.assertEqual(server.run_status["status"], "running")
            self.assertEqual(server.run_status["task_id"], "new-active-task")
        finally:
            server.run_status, server.run_task_id, server.run_stop_event = old_status, old_task, old_stop

    def test_old_description_requires_explicit_reference_preparation_then_is_reused(self):
        previous_state, previous_status, previous_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            try:
                with patch.object(server, "DESCRIPTION_INPUT_VERSION", 2):
                    old_signature = server.reference_signature(0)
                stale = {"description": "old rule", "prompts": [{"label": "a", "text": "target"}],
                         "reference_signature": old_signature}
                (server.WORK / "description.json").write_text(json.dumps(stale))
                client = server.app.test_client()
                self.assertIsNone(client.get("/api/state").json["description"])
                result = {"shapes": [], "candidates_added": 0, "applied": [], "review_queue": [],
                          "agent": {"converged": True, "rounds": 1, "stop_reason": "done"}}
                with patch.object(server, "describe", return_value={"description": "new rule",
                        "prompts": [{"label": "a", "text": "target"}]}) as describe, \
                     patch.object(server, "run", return_value=result) as run:
                    response = client.post("/api/run", json={"frame": 1, "count": 1, "description": stale})
                    self.assertEqual(response.status_code, 400, response.json)
                    describe.assert_not_called()
                    run.assert_not_called()
                    prepared = client.post("/api/describe", json={"frame": 0})
                    self.assertEqual(prepared.status_code, 200, prepared.json)
                    current = prepared.json
                    self.assertNotEqual(current["reference_signature"], old_signature)
                    self.assertEqual(current["description"], "new rule")
                    self.assertEqual(client.get("/api/state").json["description"], current)
                    self.assertTrue(server.description_cache_path(
                        current["reference_signature"]).exists())
                    for _ in range(2):
                        reused = client.post("/api/run", json={"frame": 1, "count": 1})
                        self.assertEqual(reused.status_code, 200, reused.json)
                    self.assertEqual(describe.call_count, 1)
                    self.assertEqual(run.call_count, 2)
                    for call in run.call_args_list:
                        self.assertEqual(call.kwargs["reference"]["annotations"], [shape()])
            finally:
                server.state, server.run_status, server.WORK = previous_state, previous_status, previous_work

    def test_descriptions_are_cached_per_topic_and_restored_on_camera_switch(self):
        previous_state, previous_work = server.state, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {
                "video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]},
                "truths_by_topic": {"camera1": {"0": [shape(left=3)]}},
                "reference_frames": {"camera0": [0], "camera1": [0]},
                "config": {}, "candidates": {}, "candidates_by_topic": {},
                "agent_runs": {}, "agent_runs_by_topic": {},
                "candidate_revisions": {}, "candidate_revisions_by_topic": {},
            }
            try:
                camera0 = {"description": "camera zero", "prompts": [
                    {"label": "a", "text": "zero"}],
                    "reference_signature": server.reference_signature(0)}
                server.save_reference_description(camera0)
                client = server.app.test_client()

                switched = client.post("/api/topic", json={"topic": "camera1"})
                self.assertEqual(switched.status_code, 200, switched.json)
                self.assertFalse(switched.json["description_cache_hit"])
                camera1 = {"description": "camera one", "prompts": [
                    {"label": "a", "text": "one"}],
                    "reference_signature": server.reference_signature(0)}
                server.save_reference_description(camera1)
                self.assertTrue(client.get("/api/run/readiness").json["ready"])

                restored = client.post("/api/topic", json={"topic": "camera0"})
                self.assertEqual(restored.status_code, 200, restored.json)
                self.assertTrue(restored.json["description_cache_hit"])
                self.assertEqual(restored.json["description"], camera0)
                self.assertEqual(client.get("/api/state").json["description"], camera0)
                self.assertTrue(client.get("/api/run/readiness").json["ready"])
                readiness = client.get("/api/state/summary").json["topic_readiness"]
                self.assertTrue(readiness["camera0"]["ready"])
                self.assertTrue(readiness["camera1"]["ready"])
                self.assertEqual(readiness["camera0"]["reference_frame"], 0)
                self.assertFalse(readiness["camera0"]["human_truth_verified"])

                cache_files = list((server.WORK / "descriptions").glob("*.json"))
                self.assertEqual(len(cache_files), 2)
                self.assertEqual({json.loads(path.read_text())["description"]
                                  for path in cache_files}, {"camera zero", "camera one"})
            finally:
                server.state, server.WORK = previous_state, previous_work

    def test_compact_topic_switch_loads_only_selected_frame_and_guards_lazy_reads(self):
        previous_state, previous_work = server.state, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            dense = [shape(index + 1) for index in range(240)]
            server.state = {
                "video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()], "2": [shape(2)]},
                "truths_by_topic": {"camera1": {}}, "reference_frames": {},
                "config": {}, "candidates": {"1": dense, "2": [shape(3)]},
                "candidates_by_topic": {}, "agent_runs": {"1": {"rounds": 2}},
                "agent_runs_by_topic": {}, "candidate_revisions": {},
                "candidate_revisions_by_topic": {}, "human_review_transactions": {},
            }
            try:
                client = server.app.test_client()
                full = client.post("/api/topic", json={"topic": "camera0"})
                compact = client.post("/api/topic", json={
                    "topic": "camera0", "frame": 0, "compact": True,
                    "path": str(FakeVideo.path),
                    "file_activation_generation": server.current_file_activation_generation()})
                self.assertEqual(compact.status_code, 200, compact.json)
                self.assertTrue(compact.json["compact_geometry"])
                self.assertEqual(compact.json["frame"], 0)
                self.assertEqual(compact.json["truth"], {"0": [shape()]})
                self.assertEqual(compact.json["candidates"], {})
                self.assertEqual(compact.json["truth_frames"], ["0", "2"])
                self.assertEqual(compact.json["candidate_counts"], {"1": 240, "2": 1})
                self.assertEqual(compact.json["labels"], ["a"])
                self.assertEqual(compact.json["auto_labels"], ["a"])
                self.assertEqual(compact.json["reference_only_labels"], [])
                self.assertLess(len(compact.data), len(full.data) * 0.05)

                stale_topic = server.state["topic"]
                stale = client.post("/api/topic", json={
                    "topic": "camera1", "frame": 0, "compact": True,
                    "path": "/stale/other-file.mcap",
                    "file_activation_generation": server.current_file_activation_generation()})
                self.assertEqual(stale.status_code, 409, stale.json)
                self.assertIn("另一个窗口切换", stale.json["error"])
                self.assertEqual(server.state["topic"], stale_topic)

                lazy = client.get("/api/frame_state/1", query_string={
                    "topic": "camera0", "path": str(FakeVideo.path),
                    "file_activation_generation": server.current_file_activation_generation()})
                self.assertEqual(lazy.status_code, 200, lazy.json)
                self.assertEqual(len(lazy.json["candidates"]["1"]), 240)
                self.assertEqual(lazy.json["agent_runs"], {"1": {"rounds": 2}})
                self.assertEqual(lazy.json["reference_frames"], {})
                binding = lazy.json["candidate_review_bindings"]["1"]
                self.assertEqual(len(binding["candidates"]), 240)
                audit_version = lazy.json["candidate_class_audit_version"]
                self.assertEqual(audit_version["schema_version"],
                                 "labelagent.candidate_class_audit_version.v1")
                self.assertEqual(len(audit_version["config_sha256"]), 64)
                self.assertEqual(len(audit_version["protocol_sha256"]), 64)
                self.assertEqual(len(audit_version["labels_sha256"]), 64)

                stale = client.get("/api/frame_state/1", query_string={
                    "topic": "camera1", "path": str(FakeVideo.path),
                    "file_activation_generation": server.current_file_activation_generation()})
                self.assertEqual(stale.status_code, 409)
                wrong_file = client.get("/api/frame_state/1", query_string={
                    "topic": "camera0", "path": "/tmp/other.mcap",
                    "file_activation_generation": server.current_file_activation_generation()})
                self.assertEqual(wrong_file.status_code, 409)
                old_window = client.get("/api/frame_state/1", query_string={
                    "topic": "camera0", "path": str(FakeVideo.path),
                    "file_activation_generation": server.current_file_activation_generation() - 1})
                self.assertEqual(old_window.status_code, 409)
            finally:
                server.state, server.WORK = previous_state, previous_work

    def test_describe_releases_state_lock_and_rejects_stale_completion(self):
        old_state, old_status, old_work, old_stop = (
            server.state, server.run_status, server.WORK, server.run_stop_event)
        entered, release = threading.Event(), threading.Event()
        responses = {}
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            server.run_stop_event = threading.Event()
            def blocked_describe(*args, **kwargs):
                entered.set()
                self.assertTrue(release.wait(3))
                return {"description": "a only",
                        "prompts": [{"label": "a", "text": "a"}]}
            def request_description():
                responses["describe"] = server.app.test_client().post(
                    "/api/describe", json={"frame": 0})
            def read_state():
                responses["state"] = server.app.test_client().get("/api/state")
            worker = threading.Thread(target=request_description)
            reader = threading.Thread(target=read_state)
            try:
                with patch.object(server, "describe", side_effect=blocked_describe):
                    worker.start()
                    self.assertTrue(entered.wait(2))
                    reader.start()
                    reader.join(1)
                    self.assertFalse(reader.is_alive(), "描述模型调用不应阻塞状态读取")
                    self.assertEqual(responses["state"].status_code, 200)
                    with server.lock:
                        server.state["config"]["llm_prompt_profile"] = "compact"
                    release.set()
                    worker.join(3)
                self.assertFalse(worker.is_alive())
                self.assertEqual(responses["describe"].status_code, 409)
                self.assertFalse((server.WORK / "description.json").exists())
                self.assertEqual(server.run_status["status"], "failed")
            finally:
                release.set()
                worker.join(3)
                reader.join(3)
                server.state, server.run_status, server.WORK, server.run_stop_event = (
                    old_state, old_status, old_work, old_stop)

    def test_reference_outline_does_not_tint_raw_floor_or_reuse_old_description(self):
        old_state = server.state
        server.state = {"video": FakeVideo(), "topic": "camera0",
                        "truth": {"0": [shape(label="地面")]}}
        try:
            original = Image.open(io.BytesIO(server.frame_image(0))).convert("RGB")
            outline = Image.open(io.BytesIO(server.overlay(0))).convert("RGB")
            center = outline.getpixel((25, 25))
            self.assertTrue(all(abs(a-b) < 45 for a, b in zip(center, original.getpixel((25, 25)))))
            self.assertGreater(outline.getpixel((2, 10))[0], outline.getpixel((2, 10))[1])
            old_signature = server.reference_signature(0)
            original_truth = server.state["truth"]["0"]
            server.state["truth"]["0"] = [shape(label="地面", left=3)]
            self.assertNotEqual(old_signature, server.reference_signature(0))
            self.assertEqual(original_truth[0]["label"], "地面")
        finally:
            server.state = old_state

    def test_pallet_cleanup_refresh_is_bound_reversible_and_idempotent(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            mask = np.zeros((40, 40), dtype=np.uint8)
            mask[1:39, 1:39] = 1
            mask[12:17, 12:17] = 0
            mask[0, 20] = 1
            pallet = _shape_from_mask(mask, "pallet", 7)
            candidate_sha = server._sha256_json([pallet])
            prior_run = {
                "candidate_artifact_sha256": candidate_sha,
                # A legacy cleanup receipt lacks the boundary-simplification
                # field and must not block a one-time upgrade to this version.
                "candidate_geometry_adjustments": [{
                    "shape_id": 7,
                    "reason_code": "pallet_texture_geometry_cleanup",
                }],
                "candidate_class_audits": [{"candidate_id": 7}],
                "candidate_class_audit_status": "complete",
                "review_queue": [{"source": "candidate_class_audit", "candidate_id": 7}],
            }
            original_truth = {"1": [shape(99, "pallet")]}
            server.state = {
                "video": FakeVideo(), "topic": "camera0", "labels": ["pallet"],
                "truth": copy.deepcopy(original_truth), "reference_frames": {"camera0": [1]},
                "config": {}, "candidates": {"1": [copy.deepcopy(pallet)]},
                "agent_runs": {"1": copy.deepcopy(prior_run)},
                "candidate_review_status": {"1": {"old": {"route": "candidate_preserved"}}},
                "candidate_revisions": {},
            }
            server.run_status = {"status": "completed"}
            client = server.app.test_client()
            try:
                stale = active_auto_clear_scope(1)
                stale["expected_candidate_artifact_sha256"] = "0" * 64
                self.assertEqual(client.post(
                    "/api/candidates/pallet_cleanup/1", json=stale).status_code, 409)

                response = client.post(
                    "/api/candidates/pallet_cleanup/1", json=active_auto_clear_scope(1))
                self.assertEqual(response.status_code, 200, response.json)
                self.assertTrue(response.json["changed"])
                self.assertEqual(response.json["adjusted_candidate_count"], 1)
                self.assertFalse(response.json["human_truth"])
                self.assertFalse(response.json["approved_for_training"])
                self.assertFalse(response.json["approved_for_deployment"])
                self.assertTrue(response.json["truth_preserved"])
                refreshed = server.state["candidates"]["1"][0]
                refreshed_mask = _rasterize(refreshed, 40, 40)
                self.assertEqual(refreshed_mask[14, 14], 1)
                self.assertGreater(response.json["changed_pixels"], 0)
                run = server.state["agent_runs"]["1"]
                self.assertEqual(run["candidate_class_audits"], [])
                self.assertEqual(run["candidate_class_audit_status"], "complete")
                self.assertFalse(run["approved_for_training"])
                self.assertEqual(server.state["candidate_review_status"]["1"], {})
                self.assertEqual(server.state["truth"], original_truth)
                self.assertEqual(
                    server.state["candidate_revisions"]["1"][-1]["reason"],
                    "pallet_geometry_cleanup_refresh")

                before_second = copy.deepcopy(server.state["candidates"]["1"])
                second = client.post(
                    "/api/candidates/pallet_cleanup/1", json=active_auto_clear_scope(1))
                self.assertEqual(second.status_code, 200, second.json)
                self.assertFalse(second.json["changed"])
                self.assertEqual(server.state["candidates"]["1"], before_second)
                self.assertEqual(len(server.state["candidate_revisions"]["1"]), 1)
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_clear_auto_result_removes_only_selected_frame_and_preserves_truth(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            original_truth = {"0": [shape()], "2": [shape(2)]}
            server.state = {"video": FakeVideo(), "topic": "camera0",
                "truth": original_truth.copy(), "candidates": {"1": [shape()], "2": [shape(2)]},
                "agent_runs": {"1": {"rounds": 1}, "2": {"rounds": 1}},
                "reference_frames": {"camera0": [0]}, "config": {}}
            server.run_status = {"status": "completed", "topic": "camera0",
                "file": str(FakeVideo.path), "completed": 2, "total": 2,
                "events": [{"frame": 1}, {"frame": 2}]}
            def call(frame, topic="camera0", purpose="自动标注"):
                return {"id": str(frame) + topic + purpose, "frame": frame, "topic": topic,
                    "file": str(FakeVideo.path), "purpose": purpose}
            kept_image = server.save_inspection_image(image_bytes())
            marked = io.BytesIO()
            Image.new("RGB", (40, 40), "black").save(marked, "JPEG")
            removed_image = server.save_inspection_image(marked.getvalue())
            shared_description = {"description": "floor", "prompts": [{"label": "a", "text": "floor"}],
                                  "reference_signature": server.reference_signature(0)}
            server.save_description_trace([call(0, purpose="生成真值描述")], shared_description)
            server.save_run({"file": str(FakeVideo.path), "topic": "camera0",
                "description": shared_description,
                "trace": [call(1) | {"input": {"image": {"url": kept_image}}},
                          call(2) | {"input": {"image": {"url": removed_image}}},
                          call(2, purpose="生成真值描述"), call(2, topic="camera1")],
                "steps": [{"frame": 1, "image": kept_image}, {"frame": 2, "image": removed_image}],
                "frames": [{"frame": 1}, {"frame": 2}]})
            server.save_debug([call(1), call(2), call(2, topic="camera1")])
            try:
                client = server.app.test_client()
                wrong_topic = active_auto_clear_scope(1)
                wrong_topic["topic"] = "camera1"
                self.assertEqual(client.delete(
                    "/api/auto_result/1", json=wrong_topic).status_code, 409)
                stale_activation = active_auto_clear_scope(1)
                stale_activation["file_activation_generation"] -= 1
                self.assertEqual(client.delete(
                    "/api/auto_result/1", json=stale_activation).status_code, 409)
                self.assertIn("1", server.state["candidates"])
                server.run_status["status"] = "running"
                self.assertEqual(client.delete(
                    "/api/auto_result/1",
                    json=active_auto_clear_scope(1)).status_code, 503)
                server.run_status["status"] = "completed"
                response = client.delete(
                    "/api/auto_result/2", json=active_auto_clear_scope(2))
                self.assertEqual(response.status_code, 200, response.json)
                self.assertTrue(response.json["truth_preserved"])
                self.assertEqual(response.json["path"], str(FakeVideo.path))
                self.assertEqual(response.json["topic"], "camera0")
                self.assertEqual(response.json["file_activation_generation"],
                                 server.current_file_activation_generation())
                self.assertEqual(server.state["truth"], original_truth)
                self.assertEqual(list(server.state["candidates"]), ["1"])
                self.assertEqual(list(server.state["agent_runs"]), ["1"])
                self.assertEqual(list(json.loads(server.state_path().read_text())["candidates"]), ["1"])
                output = json.loads((server.WORK / "last_run.json").read_text())
                self.assertEqual([item["frame"] for item in output["steps"]], [1])
                self.assertEqual([item["frame"] for item in output["frames"]], [1])
                self.assertEqual(len(output["trace"]), 3)
                self.assertEqual(len(json.loads((server.WORK / "model_debug.json").read_text())["trace"]), 2)
                inspection = client.get("/api/run/inspection?topic=camera0&frame=2").json
                self.assertFalse(inspection["steps"])
                self.assertFalse(inspection["trace"])
                self.assertIsNone(inspection["description"])
                kept = client.get("/api/run/inspection?topic=camera0&frame=1").json
                self.assertTrue(kept["trace"])
                self.assertFalse(any(item.get("purpose") == "生成真值描述" for item in kept["trace"]))
                self.assertEqual(kept["description"], shared_description)
                reference = client.get("/api/run/inspection?topic=camera0&frame=0").json
                self.assertEqual(reference["description"], shared_description)
                self.assertTrue(any(item["purpose"] == "生成真值描述"
                                    for item in reference["trace"]))
                self.assertEqual(server.run_status["events"], [{"frame": 1}])
                self.assertEqual(server.run_status["completed"], 1)
                self.assertTrue((server.WORK / "inspection" / kept_image.rsplit("/", 1)[-1]).exists())
                self.assertFalse((server.WORK / "inspection" / removed_image.rsplit("/", 1)[-1]).exists())
                stale_result = active_auto_clear_scope(1)
                server.state["candidates"]["1"] = [shape(3)]
                rejected = client.delete("/api/auto_result/1", json=stale_result)
                self.assertEqual(rejected.status_code, 409, rejected.json)
                self.assertEqual(server.state["candidates"]["1"], [shape(3)])
                server.save_run({"trace": [call(1), call(1, topic="camera1")],
                    "frames": [{"frame": 1}], "steps": [{"frame": 1}]})
                self.assertEqual(client.delete(
                    "/api/auto_result/1",
                    json=active_auto_clear_scope(1)).status_code, 200)
                self.assertEqual(server.state["candidates"], {})
                self.assertEqual(server.state["truth"], original_truth)
                legacy = json.loads((server.WORK / "last_run.json").read_text())
                self.assertEqual(legacy["frames"], [])
                self.assertEqual(legacy["steps"], [])
                self.assertEqual([item["topic"] for item in legacy["trace"]], ["camera1"])
                self.assertEqual(server.run_status["status"], "idle")
                self.assertEqual(server.run_status["events"], [])
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_inspection_preserves_segmentation_protocol_retry_and_edited_result(self):
        previous_state, previous_status, previous_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            decisions = [{"status": "continue", "operations": [
                {"operation": "erase_region", "shape_ids": [1], "reason": "edge"}]},
                {"status": "continue", "reason": "trim edge", "operations": [
                {"operation": "erase_region", "shape_ids": [1], "reason": "edge",
                 "polygon": [[12, 0], [39, 0], [39, 39], [12, 39]]}]},
                {"status": "done", "reason": "matches reference", "operations": []}]
            try:
                with patch.object(workflow, "_sam_request", return_value={"results": [sam_result()]}), \
                     patch.object(workflow, "_chat", side_effect=decisions):
                    client = server.app.test_client()
                    response = client.post("/api/run", json={"frame": 2, "count": 1,
                        "description": {"description": "square only", "prompts": [{"label": "a", "text": "square"}],
                                        "reference_signature": server.reference_signature(0)}})
                self.assertEqual(response.status_code, 200, response.json)
                record = client.get("/api/run/inspection?topic=camera0&frame=2").json
                self.assertEqual([step["phase"] for step in record["steps"]], ["generated", "edited", "finished"])
                self.assertNotEqual(record["steps"][0]["image"], record["steps"][-1]["image"])
                self.assertEqual([call["service"] for call in record["trace"]], ["SAM 3", "GPT", "GPT", "GPT"])
                reviews = record["trace"][1:]
                self.assertEqual([call["round"] for call in reviews], [1, 1, 2])
                self.assertEqual([call["review_attempt"] for call in reviews], [1, 2, 1])
                self.assertEqual([call["status"] for call in reviews], ["rejected", "succeeded", "succeeded"])
                self.assertIn("polygon", reviews[0]["error"])
                self.assertIn("polygon", reviews[1]["retry_feedback"])
                self.assertEqual(reviews[0]["output"]["parsed"], decisions[0])
                self.assertEqual(reviews[0]["input"]["images"], reviews[1]["input"]["images"])
                self.assertEqual(server.state["agent_runs"]["2"]["rounds"], 2)
                for call in reviews:
                    images = call["input"]["images"]
                    self.assertEqual(len(images), 5)
                    self.assertEqual(images[3]["url"], server.save_inspection_image(server.frame_image(0)))
                    self.assertEqual(images[4]["url"], server.save_inspection_image(server.overlay(0)))
                    self.assertIn(json.dumps([shape()], ensure_ascii=False), call["input"]["instruction"])
                self.assertNotEqual(reviews[1]["input"]["images"][1]["url"], reviews[2]["input"]["images"][1]["url"])
                self.assertEqual(record["steps"][-1]["image"], reviews[2]["input"]["images"][1]["url"])
                self.assertEqual(server.state["truth"], {"0": [shape()]})
            finally:
                server.state, server.run_status, server.WORK = previous_state, previous_status, previous_work

    def test_inspection_exposes_model_images_and_filters_frame_and_topic(self):
        old_state, old_work = server.state, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "config": {}}
            image = image_bytes()
            try:
                with patch.object(workflow, "_sam_request", return_value={"results": [sam_result()] }):
                    events = []
                    with workflow.trace_calls(events.append, server.save_inspection_image):
                        workflow.sam_text(image, "floor", {})
                self.assertEqual(events[-1]["status"], "succeeded")
                url = events[-1]["output"]["segmentation"]["url"]
                with server.app.test_client().get(url) as image_response:
                    self.assertEqual(image_response.status_code, 200)
                self.assertIn("/api/inspection/image/", events[-1]["input"]["image"]["url"])
                event = events[-1] | {"frame": 2, "topic": "camera0", "file": str(FakeVideo.path), "purpose": "自动标注"}
                other = event | {"frame": 1, "id": "another"}
                server.save_run({"file": str(FakeVideo.path), "topic": "camera0",
                    "reference": 0, "description": {"description": "floor"},
                    "trace": [event, other], "steps": [{"frame": 2, "phase": "finished", "image": url}]})
                client = server.app.test_client()
                result = client.get("/api/run/inspection?topic=camera0&frame=2").json
                self.assertEqual(result["frames"], [1, 2])
                self.assertEqual([item["frame"] for item in result["trace"]], [2])
                self.assertEqual(result["steps"][0]["phase"], "finished")
                self.assertFalse(client.get("/api/run/inspection?topic=camera1&frame=2").json["trace"])
            finally:
                server.state, server.WORK = old_state, old_work

    def test_inspection_prefers_newer_live_trace_for_persisted_candidate_frame(self):
        old_state, old_work = server.state, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "config": {},
                            "candidates": {"2": [shape()]},
                            "agent_runs": {"2": {"agent_review_status": "complete"}},
                            "reference_frames": {"camera0": []}}
            saved = {"id": "saved", "service": "SAM 3", "status": "succeeded",
                     "frame": 2, "topic": "camera0", "file": str(FakeVideo.path),
                     "purpose": "自动标注", "started_at": 10}
            live_sam = saved | {"id": "live-sam", "started_at": 20,
                                "purpose": "参考帧机器草稿"}
            live_gpt = saved | {"id": "live-gpt", "service": "GPT",
                                "started_at": 21, "purpose": "参考帧机器草稿",
                                "input": {"instruction": "review"},
                                "output": {"parsed": {"status": "done"}}}
            try:
                server.save_run({"file": str(FakeVideo.path), "topic": "camera0",
                                 "trace": [saved], "steps": [], "frames": [{"frame": 2}]})
                server.save_debug([live_sam, live_gpt])
                result = server.app.test_client().get(
                    "/api/run/inspection?topic=camera0&frame=2").json
                self.assertEqual([item["id"] for item in result["trace"]],
                                 ["live-sam", "live-gpt"])
            finally:
                server.state, server.WORK = old_state, old_work

    def test_inspection_keeps_per_frame_trace_after_later_run_overwrites_debug_files(self):
        old_state, old_work = server.state, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            persisted = {"id": "persisted-gpt", "service": "GPT", "status": "succeeded",
                         "frame": 2, "topic": "camera0", "file": str(FakeVideo.path),
                         "purpose": "自动标注", "started_at": 20,
                         "input": {"instruction": "review frame two"},
                         "output": {"parsed": {"status": "done"}}}
            later = persisted | {"id": "later-gpt", "frame": 3, "started_at": 30}
            server.state = {"video": FakeVideo(), "topic": "camera0", "config": {},
                            "candidates": {"2": [shape()]},
                            "agent_runs": {"2": {"agent_review_status": "complete",
                                                   "inspection_trace": [persisted]}},
                            "reference_frames": {"camera0": []}}
            try:
                server.save_run({"file": str(FakeVideo.path), "topic": "camera0",
                                 "trace": [later], "steps": [], "frames": [{"frame": 3}]})
                server.save_debug([later])
                result = server.app.test_client().get(
                    "/api/run/inspection?topic=camera0&frame=2").json
                self.assertEqual([item["id"] for item in result["trace"]],
                                 ["persisted-gpt"])
            finally:
                server.state, server.WORK = old_state, old_work

    def test_missing_or_stale_description_rejected_without_model_calls_or_record_changes(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            record = server.WORK / "last_run.json"
            record.write_text('{"previous":true}')
            try:
                with patch.object(server, "describe") as describe, patch.object(server, "run") as run:
                    client = server.app.test_client()
                    missing = client.post("/api/run", json={"frame": 1, "count": 1})
                    self.assertEqual(missing.status_code, 400)
                    stale = {"description": "old", "prompts": [{"label": "a", "text": "square"}],
                             "reference_signature": "stale"}
                    (server.WORK / "description.json").write_text(json.dumps(stale))
                    response = client.post("/api/run", json={"frame": 1, "count": 1})
                    self.assertEqual(response.status_code, 400)
                    self.assertIn("生成真值描述", response.json["error"])
                    describe.assert_not_called()
                    run.assert_not_called()
                    self.assertEqual(record.read_text(), '{"previous":true}')
                    self.assertEqual(server.run_status, {"status": "idle"})
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_debug_persists_calls_without_overwriting_failed_candidate_audit(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {"llm_url": "http://example.test", "llm_model": "test",
                           "llm_key": "secret-token"}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            try:
                responses = [io.BytesIO(b'{"results":[]}'), io.BytesIO(json.dumps({
                    "choices": [{"message": {"content": json.dumps({"status": "needs_review",
                        "reason": "secret-token no masks", "operations": []})}}]}).encode())]
                with patch.object(workflow.urllib.request, "urlopen", side_effect=responses):
                    client = server.app.test_client()
                    response = client.post("/api/run", json={"frame": 1, "count": 1,
                        "description": {"description": "a only", "prompts": [{"label": "a", "text": "square"}],
                                        "reference_signature": server.reference_signature(0)}})
                self.assertEqual(response.status_code, 200, response.json)
                debug = client.get("/api/run/debug").json
                self.assertEqual(len(debug["trace"]), 2)
                self.assertEqual(debug["trace"][0]["frame"], 1)
                self.assertEqual(debug["trace"][0]["output"]["returned_count"], 0)
                self.assertNotIn("secret-token", json.dumps(debug))
                self.assertIn("[REDACTED]", json.dumps(debug))
                self.assertEqual(json.loads((server.WORK / "model_debug.json").read_text()), debug)
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_reference_only_range_rejected_without_model_calls_or_overwriting_audit(self):
        previous = server.state
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(server, "WORK", Path(directory)), \
             patch.object(server, "describe") as describe, \
             patch.object(server, "run") as run:
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]}}
            path = Path(directory) / "last_run.json"
            path.write_text('{"previous": true}')
            try:
                response = server.app.test_client().post("/api/run", json={"frame": 0, "count": 1})
                self.assertEqual(response.status_code, 400)
                self.assertIn("没有待处理帧", response.json["error"])
                describe.assert_not_called()
                run.assert_not_called()
                self.assertEqual(json.loads(path.read_text()), {"previous": True})
            finally:
                server.state = previous

    def test_auto_labels_can_run_unreferenced_classes_as_a_separate_batch(self):
        previous = server.state
        with tempfile.TemporaryDirectory() as directory, patch.object(server, "WORK", Path(directory)):
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a", "b"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            try:
                before = server.reference_signature(0)
                client = server.app.test_client()
                response = client.post("/api/auto_labels", json={
                    **active_label_write_scope(), "labels": ["a", "b"]})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(response.json["auto_labels"], ["a", "b"])
                self.assertEqual(response.json["path"], str(FakeVideo.path))
                self.assertEqual(response.json["topic"], "camera0")
                self.assertTrue(server._valid_sha256(
                    response.json["label_state_sha256"]))
                self.assertNotEqual(before, server.reference_signature(0))
                state = client.get("/api/state").json
                self.assertEqual(state["auto_labels"], ["a", "b"])
                self.assertEqual(state["reference_only_labels"], ["a"])
                stale = {**active_request_scope(), "labels": ["b"],
                         "expected_label_state_sha256":
                             server.label_state_sha256({
                                 "labels": ["a", "b"], "auto_labels": ["a"]})}
                self.assertEqual(client.post(
                    "/api/auto_labels", json=stale).status_code, 409)
                separate = client.post("/api/auto_labels", json={
                    **active_label_write_scope(), "labels": ["b"]})
                self.assertEqual(separate.status_code, 200, separate.json)
                self.assertEqual(separate.json["auto_labels"], ["b"])
            finally:
                server.state = previous

    def test_label_writes_reject_old_file_activation_and_old_label_state(self):
        previous = server.state
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(server, "WORK", Path(directory)):
            server.state = {"video": FakeVideo(), "topic": "camera0",
                "labels": ["a", "b"], "auto_labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            try:
                client = server.app.test_client()
                stale_generation = {**active_label_write_scope(),
                    "labels": ["a", "b", "c"]}
                stale_generation["file_activation_generation"] -= 1
                response = client.post("/api/labels", json=stale_generation)
                self.assertEqual(response.status_code, 409, response.json)
                self.assertEqual(server.state["labels"], ["a", "b"])

                old_state = active_label_write_scope()
                changed = client.post("/api/auto_labels", json={
                    **old_state, "labels": ["b"]})
                self.assertEqual(changed.status_code, 200, changed.json)
                stale_write = client.post("/api/labels", json={
                    **old_state, "labels": ["a", "b", "c"]})
                self.assertEqual(stale_write.status_code, 409, stale_write.json)
                self.assertEqual(server.state["labels"], ["a", "b"])

                current = client.post("/api/labels", json={
                    **active_label_write_scope(), "labels": ["a", "b", "c"]})
                self.assertEqual(current.status_code, 200, current.json)
                self.assertEqual(current.json["labels"], ["a", "b", "c"])
                self.assertEqual(current.json["auto_labels"], ["b"])
                self.assertEqual(current.json["file_activation_generation"],
                                 server.current_file_activation_generation())
            finally:
                server.state = previous

    def test_separate_class_batch_preserves_other_class_candidates(self):
        existing = [{"label": "pallet", "points": [[0, 0], [2, 0], [2, 2]]},
                    {"label": "floor", "points": [[0, 0], [3, 0], [3, 3]]}]
        generated = [{"label": "floor", "points": [[1, 1], [4, 1], [4, 4]]},
                     {"label": "ego", "points": [[2, 2], [5, 2], [5, 5]]}]
        merged = server.merge_inference_candidates(existing, generated, ["floor", "ego"])
        self.assertEqual([shape["label"] for shape in merged], ["pallet", "floor", "ego"])
        self.assertEqual(merged[1:], generated)

    def test_describe_can_generate_prompts_for_unreferenced_selected_classes(self):
        response = {"description": "Reference class a; class b has no reference contour.",
                    "prompts": [{"label": "a", "text": "class a"},
                                {"label": "b", "text": "class b"}]}
        with patch.object(workflow, "chat", return_value=response) as chat:
            result = workflow.describe({}, b"image", [shape()], ["a", "b"],
                                       include_unreferenced=True)
        self.assertEqual([item["label"] for item in result["prompts"]], ["a", "b"])
        self.assertIn('"b"', chat.call_args.args[1])
        self.assertIn("没有参考轮廓", chat.call_args.args[1])

    def test_describe_can_use_only_unreferenced_selected_classes(self):
        response = {"description": "Class b uses a generic object rule.",
                    "prompts": [{"label": "b", "text": "class b"}]}
        with patch.object(workflow, "chat", return_value=response) as chat:
            result = workflow.describe({}, b"image", [], ["b"], include_unreferenced=True)
        self.assertEqual(result["prompts"], [{"label": "b", "text": "class b"}])
        self.assertIn('"b"', chat.call_args.args[1])

    def test_one_frame_starts_at_current_frame_and_uses_reference(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            observed = []
            def fake_run(config, image, description, labels, max_rounds, on_progress, reference):
                self.assertEqual(max_rounds, 2)
                observed.append(description)
                return {"shapes": [shape()], "candidates_added": 1, "applied": [],
                    "review_queue": [], "agent": {"converged": True, "rounds": 1,
                    "max_rounds": max_rounds, "stop_reason": "done", "audit": []}}
            (server.WORK / "description.json").write_text(json.dumps({"description": "a only",
                "prompts": [{"label": "a", "text": "a"}],
                "reference_signature": server.reference_signature(0)}))
            try:
                with patch.object(server, "run", side_effect=fake_run), \
                     patch.object(server, "describe", return_value={"description": "a only",
                         "prompts": [{"label": "a", "text": "a"}]}) as describe:
                    response = server.app.test_client().post("/api/run", json={
                        "frame": 1, "count": 1, "max_rounds": 2})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual([item["frame"] for item in response.json["frames"]], [1])
                self.assertEqual(observed, [{"description": "a only",
                    "prompts": [{"label": "a", "text": "a"}],
                    "reference_signature": server.reference_signature(0)}])
                describe.assert_not_called()
                self.assertIn("1", server.state["candidates"])
                self.assertEqual(server.state["truth"], {"0": [shape()]})
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_main_run_can_publish_reference_bound_candidates_before_agent_review(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            description = {"description": "a only", "prompts": [{"label": "a", "text": "a"}],
                           "reference_signature": server.reference_signature(0)}
            observed = []
            def fake_generate(config, image, supplied, labels, max_rounds,
                              on_progress, reference):
                observed.append((supplied, labels, reference))
                return {"shapes": [shape()], "candidates_added": 1, "applied": [],
                    "review_queue": [], "agent": {"converged": False, "rounds": 0,
                    "max_rounds": max_rounds, "stop_reason": "review_deferred", "audit": [],
                    "candidate_filters": [], "sam_generation": {"mode": "multi_prompt"}}}
            try:
                with patch.object(server, "generate_only", side_effect=fake_generate) as generate, \
                     patch.object(server, "run") as run:
                    response = server.app.test_client().post("/api/run", json={
                        "frame": 1, "count": 1, "max_rounds": 2,
                        "description": description, "defer_agent_review": True})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertTrue(response.json["agent_review_deferred"])
                self.assertEqual(response.json["frames"][0]["agent_review_status"], "pending")
                self.assertTrue(response.json["frames"][0]["agent_review_deferred"])
                self.assertRegex(response.json["frames"][0]["candidate_artifact_sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(response.json["frames"][0]["input_sha256"], r"^[0-9a-f]{64}$")
                generate.assert_called_once()
                run.assert_not_called()
                self.assertEqual(len(observed), 1)
                self.assertIsNotNone(observed[0][2])
                persisted = server.state["agent_runs"]["1"]
                self.assertEqual(persisted["agent_review_status"], "pending")
                self.assertEqual(persisted["review_reference_frame"], 0)
                self.assertEqual(persisted["review_reference_signature"],
                                 server.reference_signature(0))
                self.assertEqual(persisted["review_description_sha256"],
                                 server._sha256_json(description))
                self.assertEqual(server.state["truth"], {"0": [shape()]})
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_run_defaults_to_six_review_rounds_and_rejects_out_of_range(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            observed = []
            def fake_run(config, image, description, labels, max_rounds, on_progress, reference):
                observed.append(max_rounds)
                return {"shapes": [shape()], "candidates_added": 1, "applied": [],
                    "review_queue": [], "agent": {"converged": True, "rounds": 1,
                    "max_rounds": max_rounds, "stop_reason": "done", "audit": []}}
            description = {"description": "a only", "prompts": [{"label": "a", "text": "a"}],
                           "reference_signature": server.reference_signature(0)}
            try:
                with patch.object(server, "run", side_effect=fake_run):
                    client = server.app.test_client()
                    response = client.post("/api/run", json={"frame": 1, "count": 1,
                        "description": description})
                    invalid = client.post("/api/run", json={"frame": 2, "count": 1,
                        "max_rounds": 101, "description": description})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(observed, [6])
                self.assertEqual(response.json["frames"][0]["agent"]["max_rounds"], 6)
                self.assertEqual(invalid.status_code, 400, invalid.json)
                self.assertIn("复查轮数须为", invalid.json["error"])
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_run_exposes_shared_vlm_wait_as_a_live_phase(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            observed = {}
            def fake_run(*args, **kwargs):
                workflow._emit_trace({"id": "busy", "service": "GPT", "status": "waiting",
                    "busy_retry_count": 2, "retry_in_seconds": 10,
                    "input": {}, "started_at": 1})
                observed.update(server.app.test_client().get("/api/run/status").json)
                return {"shapes": [shape()], "candidates_added": 1, "applied": [],
                    "review_queue": [], "agent": {"converged": True, "rounds": 1,
                    "max_rounds": 6, "stop_reason": "done", "audit": []}}
            description = {"description": "a only", "prompts": [{"label": "a", "text": "a"}],
                           "reference_signature": server.reference_signature(0)}
            try:
                with patch.object(server, "run", side_effect=fake_run):
                    response = server.app.test_client().post("/api/run", json={
                        "frame": 1, "count": 1, "description": description})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(observed["phase"], "vlm_waiting")
                self.assertEqual(observed["busy_retry_count"], 2)
                self.assertEqual(observed["retry_in_seconds"], 10)
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_run_retries_one_transient_failure_then_saves_success(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            attempts = []
            failed = {"shapes": [], "candidates_added": 0, "applied": [],
                "review_queue": [{"reason": "Agent 调用失败",
                                  "error": "SAM 3 请求失败：[WinError 10061] 由于目标计算机积极拒绝，无法连接。"}],
                "agent": {"converged": False, "rounds": 0, "max_rounds": 100,
                          "stop_reason": "error", "audit": []}}
            succeeded = {"shapes": [shape()], "candidates_added": 1, "applied": [],
                "review_queue": [], "agent": {"converged": True, "rounds": 1,
                "max_rounds": 100, "stop_reason": "done", "audit": []}}
            def fake_run(*args, **kwargs):
                attempts.append(True)
                return copy.deepcopy(failed if len(attempts) == 1 else succeeded)
            try:
                with patch.object(server, "run", side_effect=fake_run):
                    response = server.app.test_client().post("/api/run", json={
                        "frame": 1, "count": 1,
                        "description": {"description": "a only",
                            "prompts": [{"label": "a", "text": "a"}],
                            "reference_signature": server.reference_signature(0)}})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(len(attempts), 2)
                self.assertFalse(response.json["frames"][0]["failed"])
                self.assertEqual(response.json["frames"][0]["agent"]["transient_retry_count"], 1)
                self.assertIn("1", server.state["candidates"])
                self.assertTrue(any(event["phase"] == "retrying"
                                    for event in server.run_status["events"]))
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_run_does_not_retry_deterministic_failure_and_preserves_candidates(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            existing = [shape()]
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {"1": copy.deepcopy(existing)},
                "agent_runs": {"1": {"stop_reason": "done"}}}
            server.run_status = {"status": "idle"}
            failed = {"shapes": [], "candidates_added": 0, "applied": [],
                "review_queue": [{"reason": "Agent 调用失败",
                                  "error": "GPT 未评估 SAM 3 点提示结果"}],
                "agent": {"converged": False, "rounds": 1, "max_rounds": 100,
                          "stop_reason": "error", "audit": []}}
            try:
                with patch.object(server, "run", return_value=copy.deepcopy(failed)) as run_mock:
                    response = server.app.test_client().post("/api/run", json={
                        "frame": 1, "count": 1,
                        "description": {"description": "a only",
                            "prompts": [{"label": "a", "text": "a"}],
                            "reference_signature": server.reference_signature(0)}})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(run_mock.call_count, 1)
                self.assertTrue(response.json["frames"][0]["failed"])
                self.assertEqual(response.json["frames"][0]["agent"]["transient_retry_count"], 0)
                self.assertEqual(server.state["candidates"]["1"], existing)
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_run_persists_audit_and_topic_switch(self):
        old_state = server.state
        old_status = server.run_status
        old_work = server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "video_path": str(FakeVideo.path),
                "topic": "camera0", "labels": ["a"], "truth": {"0": [shape()]},
                "reference_frames": {"camera0": [0]}, "config": {},
                "candidates": {}, "candidates_by_topic": {}, "agent_runs": {},
                "agent_runs_by_topic": {}}
            server.run_status = {"status": "idle"}
            audit = [{"round": 1, "decision": {"status": "done", "operations": []},
                      "tool_results": []}]

            def fake_run(config, image, description, labels, max_rounds, on_progress, reference):
                on_progress({"phase": "observing", "round": 1, "audit": audit,
                             "review_queue": [], "shapes": [shape()]})
                stored = json.loads((server.WORK / "last_run.json").read_text())
                self.assertEqual(stored["current"]["audit"], audit)
                status = server.app.test_client().get("/api/run/status").json
                self.assertEqual(status["status"], "running")
                self.assertEqual(status["frame"], 1)
                self.assertEqual(status["shape_count"], 1)
                self.assertEqual(status["events"][-1]["phase"], "observing")
                return {"shapes": [shape()], "candidates_added": 1, "applied": [],
                        "review_queue": [], "agent": {"converged": True,
                        "rounds": 1, "max_rounds": max_rounds,
                        "stop_reason": "done", "audit": audit}}

            try:
                with patch.object(server, "run", side_effect=fake_run):
                    client = server.app.test_client()
                    response = client.post("/api/run", json={"frame": 0,
                        "count": 2, "max_rounds": 2,
                        "description": {"description": "a only", "prompts": [{"label": "a", "text": "square"}],
                                        "reference_signature": server.reference_signature(0)}})
                    self.assertEqual(response.status_code, 200, response.json)
                    self.assertEqual(client.get("/api/run/status").json["status"], "completed")
                    self.assertEqual(response.json["frames"][0]["frame"], 1)
                    self.assertEqual(json.loads(server.state_path().read_text())
                        ["agent_runs"]["1"]["audit"], audit)
                    client.post("/api/topic", json={"topic": "camera1"})
                    self.assertEqual(client.get("/api/state").json["agent_runs"], {})
                    client.post("/api/topic", json={"topic": "camera0"})
                    self.assertEqual(client.get("/api/state").json["agent_runs"]["1"]["audit"], audit)
            finally:
                server.state = old_state
                server.run_status = old_status
                server.WORK = old_work

    def test_run_status_exposes_live_timing_and_sequence_contract(self):
        old_status = server.run_status
        try:
            server.run_status = {"status": "idle"}
            with patch.object(server.time, "time", side_effect=[100.0, 103.0]):
                server.update_run_status(status="running", phase="discovering",
                                         started_at=90.0, total=1, completed=0)
                status = server.app.test_client().get("/api/run/status").json
            self.assertEqual(status["progress_contract_version"], 2)
            self.assertEqual(status["status_sequence"], 1)
            self.assertEqual(status["elapsed_seconds"], 13.0)
            self.assertEqual(status["phase_elapsed_seconds"], 3.0)
            self.assertEqual(status["phase_started_at"], 100.0)
            with patch.object(server.time, "time", return_value=104.0):
                server.update_run_status(shape_count=2)
            self.assertEqual(server.run_status["status_sequence"], 2)
            self.assertEqual(server.run_status["phase_started_at"], 100.0)
            server.run_status.update(status="completed", updated_at=105.0)
            with patch.object(server.time, "time", return_value=999.0):
                completed = server.app.test_client().get("/api/run/status").json
            self.assertEqual(completed["elapsed_seconds"], 15.0)
            self.assertEqual(completed["phase_elapsed_seconds"], 5.0)
            self.assertEqual(completed["observed_at"], 999.0)
        finally:
            server.run_status = old_status

    def test_state_json_is_streamed_and_compact(self):
        old_state, old_work, old_dirty = server.state, server.WORK, server.file_sessions_dirty
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": None, "labels": ["地面"],
                            "truth": {"1": [shape()]}, "config": {}}
            try:
                server.save()
                raw = server.state_path().read_text(encoding="utf-8")
                self.assertNotIn("\n", raw)
                self.assertEqual(server._read_json(server.state_path())["labels"], ["地面"])
            finally:
                server.state, server.WORK = old_state, old_work
                server.file_sessions_dirty = old_dirty

    def test_file_sessions_migrate_once_and_are_not_rewritten_for_current_state(self):
        old_state, old_work, old_dirty = server.state, server.WORK, server.file_sessions_dirty
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            active = str((Path(directory) / "active.mcap").resolve())
            inactive = str((Path(directory) / "inactive.mcap").resolve())
            legacy = {
                "video_path": active, "topic": "camera0", "labels": ["a"],
                "truth": {"1": [shape()]}, "reference_frames": {}, "config": {},
                "candidates": {}, "agent_runs": {},
                "file_sessions": {
                    active: {"topic": "camera0", "truth": {"old": [shape(7)]}},
                    inactive: {"topic": "camera1", "truth": {"2": [shape(2)]}},
                },
            }
            server._write_json(server.state_path(), legacy)
            server.state = {"video": None, "topic": None, "labels": [], "truth": {},
                            "reference_frames": {}, "config": {}, "candidates": {},
                            "agent_runs": {}}
            try:
                server.load()
                self.assertNotIn(active, server.state["file_sessions"])
                self.assertIn(inactive, server.state["file_sessions"])
                self.assertTrue(server.file_sessions_dirty)
                server.save()
                current = server._read_json(server.state_path())
                sessions = server._read_json(server.file_sessions_path())
                self.assertNotIn("file_sessions", current)
                self.assertEqual(sessions[inactive]["truth"]["2"][0]["id"], 2)
                session_hash = server.file_sessions_path().read_bytes()
                server.state["labels"].append("b")
                server.save()
                self.assertEqual(server.file_sessions_path().read_bytes(), session_hash)
                self.assertEqual(server._read_json(server.state_path())["labels"], ["a", "b"])
            finally:
                server.state, server.WORK = old_state, old_work
                server.file_sessions_dirty = old_dirty

    def test_run_readiness_rejects_stale_description_then_accepts_current_reference(self):
        old_state, old_work = server.state, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "video_path": str(FakeVideo.path),
                            "topic": "camera0", "labels": ["a"], "auto_labels": ["a"],
                            "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                            "config": {}, "candidates": {}, "agent_runs": {}}
            try:
                client = server.app.test_client()
                missing = client.get("/api/run/readiness").json
                self.assertFalse(missing["ready"])
                self.assertEqual(missing["code"], "description_missing")
                (server.WORK / "description.json").write_text(json.dumps({
                    "description": "old", "prompts": [{"label": "a", "text": "a"}],
                    "reference_signature": "stale"}), encoding="utf-8")
                stale = client.get("/api/run/readiness").json
                self.assertFalse(stale["ready"])
                self.assertEqual(stale["code"], "description_stale")
                summary_stale = client.get("/api/state/summary").json["topic_readiness"]["camera0"]
                self.assertFalse(summary_stale["description_ready"])
                self.assertEqual(summary_stale["code"], "description_stale")
                (server.WORK / "description.json").write_text(json.dumps({
                    "description": "current", "prompts": [{"label": "a", "text": "a"}],
                    "reference_signature": server.reference_signature(0)}), encoding="utf-8")
                ready = client.get("/api/run/readiness").json
                self.assertTrue(ready["ready"])
                self.assertEqual(ready["reference_frame"], 0)
                self.assertEqual(ready["labels"], ["a"])
                summary_ready = client.get("/api/state/summary").json["topic_readiness"]["camera0"]
                self.assertTrue(summary_ready["description_ready"])
                self.assertEqual(summary_ready["code"], "ready")
            finally:
                server.state, server.WORK = old_state, old_work

    def test_run_data_readiness_decodes_one_requested_frame_and_reports_failures(self):
        old_state = server.state
        video = FakeVideo()
        calls = []
        original_frame = video.frame

        def tracked_frame(topic, index):
            calls.append((topic, index))
            return original_frame(topic, index)

        video.frame = tracked_frame
        server.state = {"video": video, "topic": "camera0"}
        try:
            client = server.app.test_client()
            ready = client.post("/api/run/data-readiness", json={
                "frames": [2, 1]}).json
            self.assertTrue(ready["ready"])
            self.assertEqual(ready["frame"], 2)
            self.assertEqual((ready["width"], ready["height"]), (40, 40))
            self.assertEqual(ready["scope"], "first_requested_frame")
            self.assertEqual(calls, [("camera0", 2)])

            video.frame = lambda topic, index: (_ for _ in ()).throw(
                ValueError("unsupported encoding"))
            failed = client.post("/api/run/data-readiness", json={
                "frame": 1, "count": 3}).json
            self.assertFalse(failed["ready"])
            self.assertEqual(failed["code"], "frame_decode_failed")
            self.assertIn("unsupported encoding", failed["message"])

            invalid = client.post("/api/run/data-readiness", json={
                "frame": 3, "count": 1}).json
            self.assertFalse(invalid["ready"])
            self.assertEqual(invalid["code"], "frame_out_of_range")
        finally:
            server.state = old_state

    def test_run_data_readiness_blocks_low_storage_before_frame_decode(self):
        old_state = server.state
        video = FakeVideo()
        calls = []
        original_frame = video.frame

        def tracked_frame(topic, index):
            calls.append((topic, index))
            return original_frame(topic, index)

        video.frame = tracked_frame
        server.state = {"video": video, "topic": "camera0"}
        try:
            client = server.app.test_client()
            with patch.object(server.shutil, "disk_usage", return_value=SimpleNamespace(
                    total=4 * 1024**3, used=3_700 * 1024**2, free=300 * 1024**2)):
                blocked = client.post("/api/run/data-readiness", json={
                    "frame": 1, "count": 10}).json
                self.assertFalse(blocked["ready"])
                self.assertEqual(blocked["code"], "storage_low")
                self.assertIn("尚未启动推理", blocked["message"])
                self.assertEqual(blocked["storage"]["max_safe_frames"], 5)
                self.assertIn("最多可安全运行 5 帧", blocked["message"])
                self.assertEqual(calls, [])

                allowed = client.post("/api/run/data-readiness", json={
                    "frame": 1, "count": 1}).json
                self.assertTrue(allowed["ready"])
                self.assertTrue(allowed["storage"]["warning"])
                self.assertEqual(allowed["storage"]["required_mib"], 264.0)
                self.assertEqual(allowed["storage"]["max_safe_frames"], 5)
                self.assertEqual(calls, [("camera0", 1)])
        finally:
            server.state = old_state

    def test_file_activation_probes_first_frame_before_mutating_current_dataset(self):
        old_state, old_work, old_dirty = server.state, server.WORK, server.file_sessions_dirty
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            current = FakeVideo()
            current.path = Path(directory) / "current.mcap"
            server.state = {
                "video": current, "video_path": str(current.path), "topic": "camera0",
                "labels": ["a"], "truth": {"0": [shape()]}, "truths_by_topic": {},
                "reference_frames": {"camera0": [0]}, "config": {}, "candidates": {},
                "candidates_by_topic": {}, "agent_runs": {}, "agent_runs_by_topic": {},
                "candidate_revisions": {}, "candidate_revisions_by_topic": {},
                "file_sessions": {},
            }
            bad = FakeVideo()
            bad.path = Path(directory) / "bad.mcap"
            bad.frame = lambda topic, index: (_ for _ in ()).throw(
                ValueError("damaged payload"))
            try:
                with patch.object(server, "Video", return_value=bad):
                    with self.assertRaisesRegex(ValueError, "damaged payload"):
                        server.activate_file({"path": str(bad.path)})
                self.assertIs(server.state["video"], current)
                self.assertEqual(server.state["truth"], {"0": [shape()]})
                self.assertEqual(server.state["file_sessions"], {})
                self.assertFalse(server.state_path().exists())

                good = FakeVideo()
                good.path = Path(directory) / "good.mcap"
                with patch.object(server, "Video", return_value=good):
                    loaded = server.activate_file({"path": str(good.path)})
                self.assertTrue(loaded["data_compatibility"]["ready"])
                self.assertEqual(loaded["data_compatibility"]["frame"], 0)
                self.assertEqual(loaded["data_compatibility"]["width"], 40)
                self.assertIn("data_probe", loaded["load_timings_ms"])
                self.assertIs(server.state["video"], good)
            finally:
                server.state, server.WORK = old_state, old_work
                server.file_sessions_dirty = old_dirty

    def test_file_removal_requires_current_workspace_library_and_entry_bindings(self):
        old_state, old_work, old_dirty = server.state, server.WORK, server.file_sessions_dirty
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            current_path = Path(directory) / "current.mcap"
            other_path = Path(directory) / "other.mcap"
            current_path.write_bytes(b"current-original")
            other_path.write_bytes(b"other-original")
            video = FakeVideo()
            video.path = current_path
            server.state = {
                "video": video, "video_path": str(current_path), "topic": "camera0",
                "labels": ["a"], "auto_labels": ["a"],
                "truth": {"0": [shape()]}, "truths_by_topic": {},
                "reference_frames": {"camera0": [0]}, "config": {},
                "candidates": {"1": [shape()]}, "candidates_by_topic": {},
                "agent_runs": {"1": {"rounds": 1}}, "agent_runs_by_topic": {},
                "candidate_revisions": {}, "candidate_revisions_by_topic": {},
                "file_sessions": {},
            }
            files = [
                {"id": "current", "path": str(current_path),
                 "name": current_path.name, "size": current_path.stat().st_size},
                {"id": "other", "path": str(other_path),
                 "name": other_path.name, "size": other_path.stat().st_size},
            ]
            server.write_library(files)
            try:
                client = server.app.test_client()

                def listing():
                    response = client.get("/api/files")
                    self.assertEqual(response.status_code, 200, response.json)
                    return response.json

                def removal_scope(data, item):
                    return {
                        "expected_path": str(current_path),
                        "expected_topic": "camera0",
                        "expected_file_activation_generation":
                            server.current_file_activation_generation(),
                        "expected_library_state_sha256": data["library_state_sha256"],
                        "expected_file_entry_sha256": item["entry_sha256"],
                        "expected_loaded": item["loaded"],
                    }

                first = listing()
                current = next(item for item in first["files"] if item["id"] == "current")
                other = next(item for item in first["files"] if item["id"] == "other")
                self.assertTrue(server._valid_sha256(first["library_state_sha256"]))
                self.assertTrue(current["loaded"])
                self.assertFalse(other["loaded"])

                stale_generation = removal_scope(first, other)
                stale_generation["expected_file_activation_generation"] -= 1
                response = client.delete("/api/files/other", json=stale_generation)
                self.assertEqual(response.status_code, 409, response.json)
                self.assertTrue(other_path.exists())

                wrong_library = removal_scope(first, other)
                wrong_library["expected_library_state_sha256"] = "0" * 64
                response = client.delete("/api/files/other", json=wrong_library)
                self.assertEqual(response.status_code, 409, response.json)
                self.assertEqual(len(server.read_library()), 2)

                wrong_entry = removal_scope(first, other)
                wrong_entry["expected_file_entry_sha256"] = "1" * 64
                response = client.delete("/api/files/other", json=wrong_entry)
                self.assertEqual(response.status_code, 409, response.json)

                wrong_loaded = removal_scope(first, current)
                wrong_loaded["expected_loaded"] = False
                response = client.delete("/api/files/current", json=wrong_loaded)
                self.assertEqual(response.status_code, 409, response.json)
                self.assertIs(server.state["video"], video)

                removed_other = client.delete(
                    "/api/files/other", json=removal_scope(first, other))
                self.assertEqual(removed_other.status_code, 200, removed_other.json)
                self.assertFalse(removed_other.json["unloaded"])
                self.assertFalse(removed_other.json["disk_file_deleted"])
                self.assertEqual(removed_other.json["removed_file_entry_sha256"],
                                 other["entry_sha256"])
                self.assertEqual(removed_other.json["removed_library_state_sha256"],
                                 first["library_state_sha256"])
                self.assertTrue(other_path.exists())
                self.assertIs(server.state["video"], video)

                second = listing()
                current = second["files"][0]
                removed_current = client.delete(
                    "/api/files/current", json=removal_scope(second, current))
                self.assertEqual(removed_current.status_code, 200, removed_current.json)
                self.assertTrue(removed_current.json["unloaded"])
                self.assertEqual(removed_current.json["path"], "")
                self.assertIsNone(removed_current.json["topic"])
                self.assertFalse(removed_current.json["disk_file_deleted"])
                self.assertEqual(removed_current.json["files"], [])
                self.assertTrue(server._valid_sha256(
                    removed_current.json["library_state_sha256"]))
                self.assertIsNone(server.state["video"])
                self.assertTrue(current_path.exists())
                self.assertIn(str(current_path), server.state["file_sessions"])
            finally:
                server.state, server.WORK = old_state, old_work
                server.file_sessions_dirty = old_dirty

    def test_newer_file_activation_supersedes_older_probe_before_state_mutation(self):
        old_state, old_work, old_dirty = server.state, server.WORK, server.file_sessions_dirty
        with server.file_activation_intent_lock:
            old_generation = server.file_activation_generation
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            current = FakeVideo()
            current.path = Path(directory) / "current.mcap"
            first_video, second_video = FakeVideo(), FakeVideo()
            first_video.path = Path(directory) / "first.mcap"
            second_video.path = Path(directory) / "second.mcap"
            entered, release, second_claimed = (threading.Event() for _ in range(3))
            original_first_frame = first_video.frame

            def delayed_first_frame(topic, index):
                entered.set()
                self.assertTrue(release.wait(2), "newer activation did not arrive")
                return original_first_frame(topic, index)

            first_video.frame = delayed_first_frame
            server.state = {
                "video": current, "video_path": str(current.path), "topic": "camera0",
                "labels": ["a"], "truth": {"0": [shape()]}, "truths_by_topic": {},
                "reference_frames": {"camera0": [0]}, "config": {}, "candidates": {},
                "candidates_by_topic": {}, "agent_runs": {}, "agent_runs_by_topic": {},
                "candidate_revisions": {}, "candidate_revisions_by_topic": {},
                "file_sessions": {},
            }
            files = [
                {"id": "first", "path": str(first_video.path), "name": "first.mcap", "size": 1},
                {"id": "second", "path": str(second_video.path), "name": "second.mcap", "size": 1},
            ]
            server.write_library(files)
            responses = {}
            original_claim = server.claim_file_activation
            claims = []

            def tracked_claim():
                generation = original_claim()
                claims.append(generation)
                if len(claims) == 2:
                    second_claimed.set()
                return generation

            def open_video(path):
                return first_video if Path(path) == first_video.path else second_video

            def activate(name):
                response = server.app.test_client().post(f"/api/files/{name}/load", json={})
                responses[name] = (response.status_code, response.json)

            try:
                with patch.object(server, "Video", side_effect=open_video), \
                        patch.object(server, "claim_file_activation", side_effect=tracked_claim):
                    first = threading.Thread(target=activate, args=("first",))
                    second = threading.Thread(target=activate, args=("second",))
                    first.start()
                    self.assertTrue(entered.wait(2), "older activation did not reach probe")
                    second.start()
                    self.assertTrue(second_claimed.wait(2), "newer activation was not registered")
                    release.set()
                    first.join(2)
                    second.join(2)
                    self.assertFalse(first.is_alive())
                    self.assertFalse(second.is_alive())

                self.assertEqual(responses["first"][0], 409, responses["first"])
                self.assertTrue(responses["first"][1]["stale_review"])
                self.assertIn("最后一次选择", responses["first"][1]["error"])
                self.assertEqual(responses["second"][0], 200, responses["second"])
                self.assertIs(server.state["video"], second_video)
                self.assertEqual(server.state["topic"], "camera0")
                self.assertEqual(
                    responses["second"][1]["file_activation_generation"], claims[1])
                summary = server.app.test_client().get("/api/state/summary").json
                self.assertEqual(summary["file_activation_generation"], claims[1])
                self.assertEqual(summary["path"], str(second_video.path))
            finally:
                release.set()
                server.state, server.WORK = old_state, old_work
                server.file_sessions_dirty = old_dirty
                with server.file_activation_intent_lock:
                    server.file_activation_generation = old_generation


if __name__ == "__main__":
    unittest.main()
