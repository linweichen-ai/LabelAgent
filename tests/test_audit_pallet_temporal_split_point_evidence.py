import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tools.audit_pallet_temporal_split_point_evidence import (
    audit_attempt, discover_support_reports, index_rows, read_mask,
    recompute_round700_temporal_positive_pixels, validate_report_batch,
)
from tools.evaluate_pallet_box3d_roi_sam import digest


def attempt(raw_negative=.5):
    return {
        "positive_coverage_all_temporal_points": 1.0,
        "negative_clear_all_center_conflict_points": raw_negative,
        "metrics": {"center_distance_pixels": 1,
                    "projected_box_coverage": .5,
                    "proposal_inside_projected_box": .8},
        "split_metrics": [{"candidate_pass": True}],
        "separations": [{"separated_at_0_15": True}],
    }


class TemporalSplitPointEvidenceAuditTest(unittest.TestCase):
    def test_final_split_can_rescue_raw_negative_clear_failure(self):
        mask = np.zeros((12, 12), bool); mask[2:7, 2:7] = True
        positives = np.array([[3, 3], [4, 4], [5, 5]])
        negatives = np.array([[8, 8], [9, 9]])
        result = audit_attempt(attempt(.5), mask, positives, negatives)
        self.assertEqual(result["final_split_positive_coverage"], 1)
        self.assertEqual(result["final_split_negative_clear"], 1)
        self.assertTrue(result["post_split_evidence_candidate"])
        self.assertTrue(result["rescues_raw_negative_clear_failure"])

    def test_final_split_positive_loss_still_fails(self):
        mask = np.zeros((12, 12), bool); mask[2, 2] = True
        positives = np.array([[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
        negatives = np.array([[8, 8]])
        result = audit_attempt(attempt(.5), mask, positives, negatives)
        self.assertEqual(result["final_split_positive_coverage"], .2)
        self.assertFalse(result["final_split_point_gate_pass"])
        self.assertFalse(result["post_split_evidence_candidate"])

    def test_final_mask_hash_drift_fails_closed(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "mask.png"
            cv2.imwrite(str(path), np.ones((8, 8), np.uint8) * 255)
            expected = digest(path)
            with patch("tools.audit_pallet_temporal_split_point_evidence.digest",
                       side_effect=[expected, "f" * 64]), \
                 self.assertRaisesRegex(RuntimeError, "changed while being decoded"):
                read_mask(path, expected, {})

    def test_duplicate_source_identity_fails_closed(self):
        row = {"sequence": "000", "frame_id": "1", "camera": 1,
               "target_obj_id": "2"}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            index_rows([row] * 7)

    def test_support_report_hash_and_schema_are_bound(self):
        schemas = {
            "opportunity": "labelagent.pallet_temporal_local_point_opportunities.v1",
            "overlap": "labelagent.pallet_box3d_overlap_pairs.v1",
            "baseline": "labelagent.pallet_lidar_conflict_points.v1",
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary); source_inputs = {}; inputs = {}
            for index, schema in enumerate(schemas.values()):
                path = root / str(index) / "results.json"; path.parent.mkdir()
                path.write_text(json.dumps({"schema_version": schema}), encoding="utf-8")
                source_inputs[str(path.resolve())] = digest(path)
            found = discover_support_reports(source_inputs, inputs)
            self.assertEqual(set(found), set(schemas))
            first = next(iter(source_inputs)); source_inputs[first] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash differs"):
                discover_support_reports(source_inputs, {})

    def test_bound_support_batch_identity_mismatch_fails_closed(self):
        source_rows = {("000", "1", 1, str(index)): {}
                       for index in range(7)}
        selected = [{"sequence": "000", "frame_id": "1", "camera": 1,
                     "target_obj_id": str(index)} for index in range(7)]
        support = {
            "opportunity": {"overlap_report_sha256": "same"},
            "overlap": {"_audit_bound_sha256": "same"}, "baseline": {}}
        with patch(
                "tools.audit_pallet_temporal_split_point_evidence.temporal_sam.validate_reports",
                return_value=selected):
            validate_report_batch(source_rows, support)
            changed = dict(source_rows); changed.pop(("000", "1", 1, "6"))
            changed[("999", "1", 1, "6")] = {}
            with self.assertRaisesRegex(ValueError, "support batch"):
                validate_report_batch(changed, support)

    def test_cross_batch_overlap_receipt_fails_closed(self):
        support = {
            "opportunity": {"overlap_report_sha256": "first"},
            "overlap": {"_audit_bound_sha256": "second"}, "baseline": {}}
        with self.assertRaisesRegex(ValueError, "cross-batch"):
            validate_report_batch({}, support)

    def test_historical_opportunity_tool_receipt_must_match(self):
        support = {
            "opportunity": {"overlap_report_sha256": "same",
                            "tool_sha256": "old"},
            "overlap": {"_audit_bound_sha256": "same"}, "baseline": {}}
        with self.assertRaisesRegex(ValueError, "tool receipt"):
            validate_report_batch({}, support, "different")

    def test_round700_recompute_keeps_historical_target_only_visibility(self):
        frames = []
        for index in range(11):
            frames.append({
                "frame_id": str(index),
                "objects": [{"obj_id": "target", "obj_type": "Pallet",
                             "ignore": False, "obj_attr": {},
                             "obj_psr": {"position": {"x": 0, "y": 0, "z": 5},
                                         "rotation": {"roll": 0, "pitch": 0, "yaw": 0},
                                         "scale": {"x": 4, "y": 4, "z": 4}}}],
                "points": np.asarray([[-1.0, 0, 5], [0, 0, 5], [1.0, 0, 5]],
                                     dtype=float),
            })

        def projected(points, camera, cell, slack, near=.2):
            indices = np.arange(len(points), dtype=np.int64)
            pixels = np.column_stack((indices + 10, indices + 20))
            return indices, pixels

        with patch(
                "tools.audit_pallet_temporal_split_point_evidence.project_visible",
                side_effect=projected):
            pixels, summary = recompute_round700_temporal_positive_pixels(
                frames, "5", {}, "target")
        self.assertEqual(pixels.shape[1], 2)
        self.assertIn(summary["evidence_status"], {
            "insufficient",
            "label_conditioned_temporal_local_point_prompt_opportunity"})


if __name__ == "__main__":
    unittest.main()
