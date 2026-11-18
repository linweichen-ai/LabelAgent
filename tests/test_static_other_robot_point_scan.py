import unittest

from tools.evaluate_reference_point_scan import (
    CLASS_PROMPT, classification_disagreement_review_ids,
    fixed_point_grid, foreground_ego_pair_ids, mark_foreground_ego_pairs)
from tools.evaluate_static_other_robot_point_scan import (
    safe_sample_dir, summarize_frame, verify_manifest)


def manifest():
    return {
        "schema_version": "labelagent.static-other-robot-point-scan.v1",
        "human_truth": False,
        "candidate_only": True,
        "production_deployed": False,
        "approved_for_training": False,
        "samples": [{
            "sample_id": "sample-1", "image_path": "/tmp/image.png",
            "image_sha256": "0" * 64,
            "assistant_review_candidate": {
                "review_kind": "assistant_visual_candidate",
                "human_truth": False,
                "expected_evidence": "present", "reason": "separate robot",
            },
        }],
    }


class StaticOtherRobotPointScanTests(unittest.TestCase):
    def test_fixed_point_grid_rejects_empty_scan_region(self):
        with self.assertRaisesRegex(ValueError, "no scan points"):
            fixed_point_grid(64, 64, columns=24, rows=12,
                             top_fraction=0.01)

    def test_manifest_keeps_assistant_review_outside_truth(self):
        value = manifest()
        self.assertEqual(verify_manifest(value), value["samples"])
        value["human_truth"] = True
        with self.assertRaises(ValueError):
            verify_manifest(value)

    def test_manifest_rejects_duplicate_sample_ids(self):
        value = manifest()
        value["samples"].append(dict(value["samples"][0]))
        with self.assertRaises(ValueError):
            verify_manifest(value)

    def test_manifest_rejects_non_mapping_sample_and_invalid_hash(self):
        value = manifest()
        value["samples"] = ["not-a-sample"]
        with self.assertRaises(ValueError):
            verify_manifest(value)
        value = manifest()
        value["samples"][0]["image_sha256"] = "not-a-sha256"
        with self.assertRaises(ValueError):
            verify_manifest(value)

    def test_output_directory_identifier_is_restricted(self):
        self.assertEqual(safe_sample_dir("camera0-t100"), "camera0-t100")
        with self.assertRaises(ValueError):
            safe_sample_dir("../outside")

    def test_frame_summary_reports_candidate_agreement_not_accuracy(self):
        sample = manifest()["samples"][0]
        report = {
            "selected_ids": ["C01"], "uncertain_review_ids": [],
            "point_count": 184, "scan_seconds": 1.0,
            "point_grid": {"columns": 24, "rows": 12,
                           "top_fraction": 2 / 3},
            "known_instance_seconds": 0.5, "candidate_filter_seconds": 0.1,
            "frame_elapsed_seconds": 2.0, "raw_candidates": 10,
            "geometric_candidates": 5, "nms_candidates": 4,
            "known_exclusion_passed": 3, "classified_count": 3,
            "classification_batch_seconds": {},
            "classification_strategy": "separate_dual_pass_batches",
            "surface_veto_batch_seconds": [], "rows": [],
            "foreground_ego_pair_veto_ids": [],
            "preclassification_foreground_ego_pair_veto": [],
            "classification_disagreement_review_ids": [],
        }
        row = summarize_frame(sample, report)
        self.assertTrue(row["assistant_candidate_agreement"])
        self.assertEqual(row["triage_status"], "auto_candidate_present")
        self.assertTrue(row["assistant_candidate_review_covered"])
        self.assertFalse(row["human_truth"])
        self.assertFalse(row["approved_for_training"])

    def test_review_only_frame_is_not_promoted_to_present(self):
        sample = manifest()["samples"][0]
        report = {
            "selected_ids": [], "uncertain_review_ids": ["C02"],
            "point_count": 184, "scan_seconds": 1.0,
            "point_grid": {"columns": 24, "rows": 12,
                           "top_fraction": 2 / 3},
            "known_instance_seconds": 0.5, "candidate_filter_seconds": 0.1,
            "frame_elapsed_seconds": 2.0, "raw_candidates": 10,
            "geometric_candidates": 5, "nms_candidates": 4,
            "known_exclusion_passed": 3, "classified_count": 3,
            "classification_batch_seconds": {},
            "classification_strategy": "separate_dual_pass_batches",
            "surface_veto_batch_seconds": [], "rows": [],
            "foreground_ego_pair_veto_ids": [],
            "preclassification_foreground_ego_pair_veto": [],
            "classification_disagreement_review_ids": ["C02"],
        }
        row = summarize_frame(sample, report)
        self.assertEqual(row["effective_evidence"], "absent")
        self.assertEqual(row["triage_status"], "review_required")
        self.assertTrue(row["assistant_candidate_review_covered"])

    def test_prompt_explicitly_excludes_unoccupied_forklift(self):
        self.assertIn("parked or currently unoccupied conventional forklift", CLASS_PROMPT)
        self.assertIn("operator compartment", CLASS_PROMPT)

    def test_symmetric_lower_fork_pair_is_vetoed(self):
        rows = [
            {"id": "left", "bbox": [825, 928, 895, 1085]},
            {"id": "right", "bbox": [1092, 927, 1153, 1086]},
            {"id": "background", "bbox": [723, 482, 872, 610]},
        ]
        self.assertEqual(
            foreground_ego_pair_ids(rows, 1920, 1536), {"left", "right"})

    def test_ego_pair_is_removed_before_vlm_ranking(self):
        candidates = [
            {"bbox": [825, 928, 895, 1085], "point": {}, "score": 0.9,
             "passes_known_exclusion": True},
            {"bbox": [1092, 927, 1153, 1086], "point": {}, "score": 0.9,
             "passes_known_exclusion": True},
            {"bbox": [723, 482, 872, 610], "point": {}, "score": 0.9,
             "passes_known_exclusion": True},
        ]
        records = mark_foreground_ego_pairs(candidates, 1920, 1536)
        self.assertEqual(len(records), 2)
        self.assertTrue(candidates[0]["_foreground_ego_pair_veto"])
        self.assertTrue(candidates[1]["_foreground_ego_pair_veto"])
        self.assertFalse(candidates[2]["_foreground_ego_pair_veto"])

    def test_one_pass_robot_classification_routes_to_review(self):
        rows = [{
            "id": "candidate", "bbox": [1, 2, 10, 20],
            "baseline_result": {"class": "vehicle_or_forklift"},
            "result": {"class": "other_robot"},
        }]
        self.assertEqual(
            classification_disagreement_review_ids(rows, set()), {"candidate"})
        self.assertEqual(
            classification_disagreement_review_ids(rows, {"candidate"}), set())


if __name__ == "__main__":
    unittest.main()
