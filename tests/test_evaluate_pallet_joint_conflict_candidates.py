import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tools.evaluate_pallet_joint_conflict_candidates import (
    bind_file, conflict_components, evaluate_combinations,
    index_overlap_rows, index_temporal_rows, intrinsic_candidate_pass,
    read_mask, temporal_options,
)


def projected(obj_id, mask, depth):
    return {"obj_id": str(obj_id), "mask": np.asarray(mask, dtype=bool),
            "camera_depth": float(depth)}


def option(name, mask, source="round700_temporal_raw"):
    return {"source": source, "policy": name, "option_id": name,
            "path": name, "sha256": name, "mask": np.asarray(mask, dtype=bool)}


class JointConflictCandidateTest(unittest.TestCase):
    def test_projected_conflicts_form_transitive_components(self):
        a = np.zeros((20, 30), bool); a[2:12, 2:12] = True
        b = np.zeros_like(a); b[2:12, 8:18] = True
        c = np.zeros_like(a); c[2:12, 14:24] = True
        d = np.zeros_like(a); d[15:19, 25:29] = True
        groups = conflict_components([
            projected("a", a, 1), projected("b", b, 2),
            projected("c", c, 3), projected("d", d, 4)])
        self.assertEqual([[row["obj_id"] for row in group] for group in groups],
                         [["a", "b", "c"], ["d"]])

    def test_intrinsic_filter_requires_localization_and_full_point_gates(self):
        passing = {
            "metrics": {"center_distance_pixels": 1,
                        "projected_box_coverage": .5,
                        "proposal_inside_projected_box": .8},
            "positive_coverage_all_temporal_points": .8,
            "negative_clear_all_center_conflict_points": .8,
        }
        self.assertTrue(intrinsic_candidate_pass(passing))
        for key in ("positive_coverage_all_temporal_points",
                    "negative_clear_all_center_conflict_points"):
            changed = dict(passing); changed[key] = .79
            self.assertFalse(intrinsic_candidate_pass(changed))
        changed = {**passing, "metrics": {**passing["metrics"],
                                           "center_distance_pixels": 25}}
        self.assertFalse(intrinsic_candidate_pass(changed))

    def test_joint_search_can_succeed_only_with_temporal_choice(self):
        shape = (20, 20)
        left = np.zeros(shape, bool); left[2:12, 2:10] = True
        right = np.zeros(shape, bool); right[2:12, 8:16] = True
        component = [projected("a", left, 1), projected("b", right, 2)]
        incumbent_a = np.zeros(shape, bool); incumbent_a[2:12, 2:12] = True
        incumbent_b = np.zeros(shape, bool); incumbent_b[2:12, 2:12] = True
        temporal_a = np.zeros(shape, bool); temporal_a[2:12, 2:8] = True
        good_b = np.zeros(shape, bool); good_b[2:12, 10:16] = True
        result = evaluate_combinations(component, {
            "a": [option("inc-a", incumbent_a, "round676_incumbent_raw"),
                  option("temp-a", temporal_a)],
            "b": [option("inc-b", incumbent_b, "round676_incumbent_raw"),
                  option("temp-b", good_b)],
        })
        self.assertFalse(result["incumbent_combination_pass"])
        self.assertTrue(result["joint_temporal_recovered"])
        self.assertEqual(result["selected"]["temporal_choice_count"], 2)

    def test_joint_search_failure_is_explicit(self):
        mask = np.zeros((20, 20), bool); mask[2:12, 2:12] = True
        component = [projected("a", mask, 1), projected("b", mask, 2)]
        result = evaluate_combinations(component, {
            "a": [option("a", mask, "round676_incumbent_raw")],
            "b": [option("b", mask, "round676_incumbent_raw")],
        })
        self.assertEqual(result["passing_combination_count"], 0)
        self.assertFalse(result["joint_temporal_recovered"])
        self.assertIsNone(result["selected"])
        self.assertIsNotNone(result["best_failed_combination"])
        self.assertTrue(result["member_failure_histogram"])

    def test_joint_search_preserves_integer_label_ids(self):
        left = np.zeros((20, 40), bool); left[2:12, 2:10] = True
        right = np.zeros((20, 40), bool); right[2:12, 20:28] = True
        component = [
            {**projected("10", left, 1), "obj_id": 10},
            {**projected("11", right, 2), "obj_id": 11},
        ]
        result = evaluate_combinations(component, {
            "10": [option("a", left, "round676_incumbent_raw")],
            "11": [option("b", right, "round676_incumbent_raw")],
        })
        self.assertEqual(result["evaluated_combination_count"], 1)
        self.assertTrue(result["incumbent_combination_pass"])

    def test_tie_break_is_deterministic(self):
        a = np.zeros((20, 40), bool); a[2:12, 2:10] = True
        b = np.zeros((20, 40), bool); b[2:12, 20:28] = True
        component = [projected("a", a, 1), projected("b", b, 2)]
        choices = {
            "a": [option("z", a), option("a", a)],
            "b": [option("b", b, "round676_incumbent_raw")],
        }
        first = evaluate_combinations(component, choices)
        second = evaluate_combinations(component, choices)
        self.assertEqual(first["selected"]["choice_ids"], ["a", "b"])
        self.assertEqual(first["selected"], second["selected"])

    def test_missing_mask_and_hash_drift_fail_closed(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "mask.png"; inputs = {}
            with self.assertRaisesRegex(ValueError, "missing"):
                bind_file(path, inputs)
            cv2.imwrite(str(path), np.ones((8, 8), np.uint8) * 255)
            with self.assertRaisesRegex(ValueError, "hash differs"):
                read_mask(path, "0" * 64, inputs)
            from tools.evaluate_pallet_box3d_roi_sam import digest
            expected = digest(path)
            with patch("tools.evaluate_pallet_joint_conflict_candidates.digest",
                       side_effect=[expected, "f" * 64]), \
                 self.assertRaisesRegex(RuntimeError, "changed while being decoded"):
                read_mask(path, expected, {})

    def test_valid_attempt_mask_is_bound_before_intrinsic_rejection(self):
        attempt = {
            "policy": "weak", "raw_mask_path": "missing.png",
            "raw_mask_sha256": "0" * 64,
            "metrics": {"center_distance_pixels": 30,
                        "projected_box_coverage": .5,
                        "proposal_inside_projected_box": .8},
            "positive_coverage_all_temporal_points": 1,
            "negative_clear_all_center_conflict_points": 1,
        }
        with self.assertRaisesRegex(ValueError, "missing"):
            temporal_options({"attempts": [attempt]}, {}, (8, 8))

    def test_duplicate_report_identities_fail_closed(self):
        temporal = {"sequence": "000", "frame_id": "1", "camera": 1,
                    "target_obj_id": "2"}
        overlap = {"sequence": "000", "frame_id": "1", "camera": 1}
        with self.assertRaisesRegex(ValueError, "duplicate temporal"):
            index_temporal_rows([temporal, temporal])
        with self.assertRaisesRegex(ValueError, "duplicate overlap"):
            index_overlap_rows([overlap, overlap])


if __name__ == "__main__":
    unittest.main()
