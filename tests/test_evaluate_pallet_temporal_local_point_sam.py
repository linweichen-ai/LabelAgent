import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from tools.evaluate_pallet_lidar_conflict_points import prompt_groups
from tools.evaluate_pallet_temporal_local_point_sam import (
    _baseline_delta, _bound_json, disjoint_prompt_negatives,
    exclusive_conflict_negative_pixels,
    normalized_batch_items, recompute_temporal_positive_pixels,
    resolve_manifest_root, validate_pixels, validate_recomputed_opportunity,
    validate_reports, validate_loopback_sam_url,
)
from tools.evaluate_pallet_box3d_roi_sam import digest
import tools.evaluate_pallet_temporal_local_point_opportunities as temporal_local


def pallet(obj_id, center=(0, 0, 5), scale=(4, 4, 4)):
    return {
        "obj_id": obj_id, "obj_type": "Pallet", "ignore": False,
        "obj_psr": {
            "position": dict(zip(("x", "y", "z"), center)),
            "rotation": {"roll": 0, "pitch": 0, "yaw": 0},
            "scale": dict(zip(("x", "y", "z"), scale)),
        },
    }


def overlap():
    return {
        "schema_version": "labelagent.pallet_box3d_overlap_pairs.v1",
        "status": "offline_instance_diagnostic_not_deployed",
        "human_pixel_truth": False, "approved_for_training": False,
        "production_deployed": False, "workbench_writes": 0,
        "input_sha256": {"frozen": "hash"},
        "rows": [{
            "sequence": "010", "frame_id": "105", "camera": 1,
            "depth_split_metrics": [
                {"obj_id": str(index), "handled_candidate": False}
                for index in range(7)],
        }],
    }


def opportunity():
    rows = [{
        "sequence": "010", "frame_id": "105", "primary_camera": 1,
        "target_obj_id": str(index),
        "evidence_status":
            "label_conditioned_temporal_local_point_prompt_opportunity",
    } for index in range(7)]
    return {
        "schema_version": "labelagent.pallet_temporal_local_point_opportunities.v1",
        "status": "offline_temporal_local_point_prompt_opportunities_not_deployed",
        "candidate_only": True, "human_pixel_truth": False,
        "approved_for_training": False, "production_deployed": False,
        "human_truth_writes": 0, "training_data_writes": 0,
        "workbench_writes": 0, "oss_writes": 0,
        "input_sha256": {"frozen": "hash"},
        "tool_sha256": digest(Path(temporal_local.__file__)),
        "overlap_report_sha256": "overlap-hash",
        "opportunity_count": 7, "rows": rows,
    }


def baseline():
    return {
        "schema_version": "labelagent.pallet_lidar_conflict_points.v1",
        "status": "offline_candidate_diagnostic_not_deployed",
        "human_pixel_truth": False, "approved_for_training": False,
        "production_deployed": False, "workbench_writes": 0,
        "training_data_writes": 0, "oss_writes": 0,
        "input_sha256": {"source": "hash"},
        "rows": [],
    }


def frames():
    result = []
    for index in range(11):
        result.append({
            "frame_id": str(100 + index),
            "objects": [pallet("target")],
            "points": np.asarray([
                [-1.5 + index * .04, 0, 5],
                [0 + index * .04, 0, 5],
                [1.5 + index * .04, 0, 5]], dtype=float),
        })
    return result


class TemporalLocalPointSamTest(unittest.TestCase):
    def test_sam_url_must_remain_on_loopback(self):
        for value in ("http://127.0.0.1:28766", "http://localhost:28766",
                      "http://[::1]:28766"):
            validate_loopback_sam_url(value)
        for value in ("http://172.16.88.68:28766", "file:///tmp/model",
                      "http://user:pass@127.0.0.1:28766"):
            with self.assertRaisesRegex(ValueError, "loopback"):
                validate_loopback_sam_url(value)

    def test_only_exactly_seven_opportunity_rows_are_selected(self):
        selected = validate_reports(opportunity(), overlap(), baseline())
        self.assertEqual(len(selected), 7)
        changed = opportunity(); changed["rows"][0]["target_obj_id"] = "unknown"
        with self.assertRaisesRegex(ValueError, "not unresolved"):
            validate_reports(changed, overlap(), baseline())
        changed = opportunity(); changed["rows"].pop()
        with self.assertRaisesRegex(ValueError, "exactly seven"):
            validate_reports(changed, overlap(), baseline())

    def test_temporal_pixels_are_recomputed_from_bound_frames(self):
        def projected(points, camera, cell, slack, near=.2):
            indices = np.arange(len(points), dtype=np.int64)
            pixels = np.column_stack((indices + 10, indices + 20))
            return indices, pixels

        with patch(
                "tools.evaluate_pallet_temporal_local_point_sam.project_visible",
                side_effect=projected), \
             patch(
                "tools.evaluate_pallet_temporal_local_point_opportunities.project_visible",
                side_effect=projected):
            pixels, summary = recompute_temporal_positive_pixels(
                frames(), "105", {}, "target")
        self.assertGreaterEqual(len(pixels), 3)
        self.assertEqual(summary["evidence_status"],
                         "label_conditioned_temporal_local_point_prompt_opportunity")
        self.assertGreaterEqual(summary["noncenter_contributing_frame_count"], 2)

    def test_opportunity_summary_mismatch_fails_closed(self):
        reported = {
            "evidence_status":
                "label_conditioned_temporal_local_point_prompt_opportunity",
            "noncenter_shared_unique_pixel_count": 3,
            "noncenter_contributing_frame_count": 2,
            "shared_across_grids_unique_voxel_index_count": 3,
            "aggregate_unique_voxel_count": 4,
        }
        validate_recomputed_opportunity(dict(reported), reported)
        changed = dict(reported); changed["aggregate_unique_voxel_count"] = 5
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_recomputed_opportunity(changed, reported)

    def test_conflict_negatives_exclude_points_ambiguous_between_boxes(self):
        target = pallet("target", center=(-1, 0, 5), scale=(2, 2, 2))
        first = pallet("first", center=(1, 0, 5), scale=(2, 2, 2))
        second = pallet("second", center=(2, 0, 5), scale=(2, 2, 2))
        points = np.array([[-1, 0, 5], [.5, 0, 5], [1.5, 0, 5], [2.5, 0, 5]])
        pixels = np.array([[10, 10], [20, 20], [30, 30], [40, 40]])
        result = exclusive_conflict_negative_pixels(
            points, np.arange(4), pixels, [target, first, second],
            {"first", "second"})
        self.assertEqual(result.tolist(), [[20, 20], [40, 40]])

    def test_frozen_strategy_has_original_positive_only_and_sparse_branches(self):
        groups = prompt_groups(
            [[10, 10], [20, 20], [30, 30]],
            [[40, 40], [50, 50]], (0, 0, 100, 100), scale=2,
            positive_only_fallback=True, sparse_negative_fallback=True,
            pairwise_negative_fallback=False)
        names = [name for name, _ in groups]
        self.assertEqual(names[:4], [
            "one_positive_all_negative", "three_positive_all_negative",
            "three_positive_two_negative", "three_positive_only"])
        self.assertEqual(names[4:], [
            "three_positive_one_negative_0",
            "three_positive_one_negative_1"])

    def test_contradictory_pixel_is_not_sent_as_both_prompt_labels(self):
        pool, overlap = disjoint_prompt_negatives(
            np.array([[10, 10], [20, 20]]),
            np.array([[20, 20], [30, 30], [40, 40]]))
        self.assertEqual(overlap, 1)
        self.assertEqual(pool.tolist(), [[30, 30], [40, 40]])

    def test_batch_response_errors_and_missing_items_are_fail_closed_attempts(self):
        groups = [("a", []), ("b", [])]
        normalized = normalized_batch_items(
            groups, {"items": [{"index": 0, "error": "backend"}]})
        self.assertEqual(normalized[0]["error"], "backend")
        self.assertEqual(normalized[1]["error"], "missing_response_item")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalized_batch_items(groups, {"items": [
                {"index": 0}, {"index": 0}]})
        with self.assertRaisesRegex(ValueError, "invalid prompt group index"):
            normalized_batch_items(groups, {"items": [{"index": 2}]})
        with self.assertRaisesRegex(ValueError, "no items"):
            normalized_batch_items(groups, {"error": "busy"})
        for invalid in (False, 0.9, "0"):
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                normalized_batch_items(groups, {"items": [{"index": invalid}]})

    def test_empty_or_out_of_bounds_points_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_pixels(np.empty((0, 2), dtype=int), 100, 100, "positive")
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_pixels(np.array([[100, 0]], dtype=int), 100, 100, "positive")
        validate_pixels(np.array([[99, 99]], dtype=int), 100, 100, "positive")

    def test_json_parse_time_drift_is_rejected(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.json"
            path.write_text(json.dumps({"value": 1}), encoding="utf-8")
            with patch(
                    "tools.evaluate_pallet_temporal_local_point_sam.digest",
                    side_effect=["before", "after"]), \
                 self.assertRaisesRegex(RuntimeError, "changed while being parsed"):
                _bound_json(path)

    def test_relative_manifest_root_resolves_once_and_ambiguity_fails_closed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); manifest = root / "frozen" / "manifest.json"
            manifest.parent.mkdir(); target = manifest.parent / "images"; target.mkdir()
            self.assertEqual(resolve_manifest_root(manifest, "images"),
                             target.resolve())
            with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
                resolve_manifest_root(manifest, "absent")

    def test_round677_difference_is_explicitly_same_source(self):
        self.assertEqual(_baseline_delta(True, False),
                         "newly_recovered_vs_round677_same_source")
        self.assertEqual(_baseline_delta(False, False),
                         "not_recovered_in_either_same_source_run")
        self.assertEqual(_baseline_delta(False, True),
                         "regressed_vs_round677_same_source")


if __name__ == "__main__":
    unittest.main()
