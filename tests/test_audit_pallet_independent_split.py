import unittest

from tools.audit_pallet_independent_split import audit


def report(rows):
    return {
        "schema_version": "labelagent.pallet_box3d_overlap_pairs.v1",
        "production_deployed": False,
        "human_pixel_truth": False,
        "approved_for_training": False,
        "workbench_writes": 0,
        "rows": rows,
    }


class PalletIndependentSplitAuditTest(unittest.TestCase):
    def test_passes_independent_conservative_rows(self):
        rows = [{
            "sequence": "010", "frame_id": "new", "camera": 0,
            "all_instances_localized": True,
            "all_conflicts_separated": False,
            "depth_split_candidate_pass": False,
            "depth_or_occlusion_handled_candidate": True,
            "depth_split_metrics": [{
                "obj_id": 20, "candidate_pass": False,
                "occluded_unobservable_candidate": True,
            }],
        }]
        result = audit(report(rows), {"old"})
        self.assertTrue(result["independent_candidate_gate_pass"])
        self.assertEqual(result["depth_split_pass_row_count"], 0)
        self.assertEqual(
            result["row_summaries"][0]["occluded_unobservable_obj_ids"],
            ["20"])
        self.assertFalse(result["human_pixel_accuracy_measurable"])

    def test_rejects_overlap_with_tuning_frames(self):
        rows = [{
            "sequence": "010", "frame_id": "old", "camera": 0,
            "all_instances_localized": True,
            "depth_split_candidate_pass": True,
            "depth_or_occlusion_handled_candidate": True,
            "depth_split_metrics": [],
        }]
        result = audit(report(rows), {"old"})
        self.assertFalse(result["independent_candidate_gate_pass"])
        self.assertEqual(result["overlapping_tuning_frames"], ["old"])

    def test_requires_every_row_to_be_safely_handled(self):
        rows = [{
            "sequence": "011", "frame_id": "new", "camera": 1,
            "all_instances_localized": True,
            "depth_split_candidate_pass": False,
            "depth_or_occlusion_handled_candidate": False,
            "depth_split_metrics": [{
                "obj_id": 2, "candidate_pass": False,
                "occluded_unobservable_candidate": False,
            }],
        }]
        self.assertFalse(audit(report(rows), set())[
            "independent_candidate_gate_pass"])


if __name__ == "__main__":
    unittest.main()
