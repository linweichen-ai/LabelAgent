import unittest

from tools.route_pallet_visibility_candidates import (
    apply_split_gate, propagation_sources, route_instance, split_metric_index,
)


class PalletVisibilityCandidateRoutesTest(unittest.TestCase):
    def test_split_gate_suppresses_visible_but_invalid_mask(self):
        decision = route_instance(
            {"lidar_visibility_classification": "visible_lidar_supported"}, None)
        gated = apply_split_gate(decision, {"candidate_pass": False})
        self.assertEqual(gated["route"], "split_candidate_failed_review")
        self.assertFalse(gated["allow_current_2d_candidate"])
        self.assertTrue(gated["suppress_current_2d_candidate"])

    def test_split_metric_index_preserves_frame_identity(self):
        report = {
            "schema_version": "labelagent.pallet_box3d_overlap_pairs.v1",
            "production_deployed": False,
            "rows": [
                {"sequence": "010", "camera": 0, "frame_id": "100",
                 "depth_split_metrics": [{"obj_id": 1, "candidate_pass": True}]},
                {"sequence": "010", "camera": 0, "frame_id": "200",
                 "depth_split_metrics": [{"obj_id": 1, "candidate_pass": False}]},
            ],
        }
        indexed = split_metric_index(report)
        self.assertTrue(indexed[("010", 0, "100", "1")]["candidate_pass"])
        self.assertFalse(indexed[("010", 0, "200", "1")]["candidate_pass"])

    def test_visible_instance_allows_candidate_but_not_truth(self):
        row = route_instance({"lidar_visibility_classification":
                              "visible_lidar_supported"}, None)
        self.assertEqual(row["route"], "depth_split_candidate")
        self.assertTrue(row["allow_current_2d_candidate"])
        self.assertFalse(row["write_human_truth"])
        self.assertFalse(row["write_training_data"])

    def test_no_return_requires_matching_temporal_support(self):
        instance = {"lidar_visibility_classification":
                    "no_lidar_return_insufficient_evidence"}
        unsupported = route_instance(instance, None)
        self.assertEqual(unsupported["route"], "insufficient_evidence_review")
        temporal = {"temporal_occlusion_supported_for_center": True, "rows": [
            {"frame_id": "1", "evidence_eligible": False,
             "target_visible_points": 9},
            {"frame_id": "2", "evidence_eligible": True,
             "target_visible_points": 3},
            {"frame_id": "3", "evidence_eligible": True,
             "target_visible_points": 0},
        ]}
        supported = route_instance(instance, temporal)
        self.assertEqual(supported["route"], "temporal_occlusion_defer_2d")
        self.assertEqual(supported["propagation_source_frame_ids"], ["2"])
        self.assertTrue(supported["suppress_current_2d_candidate"])
        self.assertFalse(supported["allow_current_2d_candidate"])

    def test_propagation_sources_exclude_ignored_and_weak_frames(self):
        report = {"rows": [
            {"frame_id": "a", "evidence_eligible": False,
             "target_visible_points": 8},
            {"frame_id": "b", "evidence_eligible": True,
             "target_visible_points": 2},
            {"frame_id": "c", "evidence_eligible": True,
             "target_visible_points": 4},
        ]}
        self.assertEqual(propagation_sources(report), ["c"])


if __name__ == "__main__":
    unittest.main()
