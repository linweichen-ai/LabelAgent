import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.evaluate_pallet_temporal_conflict_support import (
    classify_temporal_evidence, freeze_window, plan_targets,
    select_nearer_occluder, validate_window_centers,
)


def metric(obj_id, depth, handled=False):
    return {
        "obj_id": obj_id,
        "camera_depth": depth,
        "handled_candidate": handled,
    }


def pair(first, second, first_overlap, second_overlap):
    return {
        "target_obj_id": first,
        "other_obj_id": second,
        "projected_target_overlap": first_overlap,
        "projected_other_overlap": second_overlap,
    }


def report(row):
    return {
        "schema_version": "labelagent.pallet_box3d_overlap_pairs.v1",
        "status": "offline_instance_diagnostic_not_deployed",
        "human_pixel_truth": False,
        "approved_for_training": False,
        "production_deployed": False,
        "workbench_writes": 0,
        "input_sha256": {"frozen": "hash"},
        "rows": [row],
    }


class PalletTemporalConflictSupportTest(unittest.TestCase):
    def test_selects_maximum_overlap_strictly_nearer_occluder(self):
        target = metric("target", 8.0)
        row = {
            "depth_split_metrics": [
                target,
                metric("near-low-overlap", 7.0, True),
                metric("near-high-overlap", 6.0, True),
                metric("far-highest-overlap", 9.0, True),
            ],
            "pairs": [
                pair("target", "near-low-overlap", .4, .3),
                pair("near-high-overlap", "target", .2, .8),
                pair("target", "far-highest-overlap", .95, .9),
            ],
        }
        selected, reason = select_nearer_occluder(row, target)
        self.assertIsNone(reason)
        self.assertEqual(selected["occluder_obj_id"], "near-high-overlap")
        self.assertEqual(selected["target_projected_overlap"], .8)
        self.assertLess(selected["occluder_camera_depth"],
                        selected["target_camera_depth"])

    def test_unresolved_target_without_nearer_peer_is_explicitly_skipped(self):
        row = {
            "sequence": "010",
            "frame_id": "100",
            "camera": 1,
            "depth_split_metrics": [metric("target", 5.0),
                                    metric("farther", 6.0, True)],
            "pairs": [pair("target", "farther", .7, .6)],
        }
        jobs, skipped = plan_targets(report(row))
        self.assertEqual(jobs, [])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["target_obj_id"], "target")
        self.assertEqual(skipped[0]["reason"],
                         "no_nearer_conflicting_occluder")
        self.assertEqual(skipped[0]["status"], "skipped")

    def test_overlap_report_must_bind_frozen_inputs(self):
        row = {
            "sequence": "010", "frame_id": "100", "camera": 1,
            "depth_split_metrics": [metric("target", 5.0)], "pairs": [],
        }
        payload = report(row)
        del payload["input_sha256"]
        with self.assertRaisesRegex(ValueError, "frozen non-deployed"):
            plan_targets(payload)

    def test_temporal_evidence_classification_never_implies_pixel_truth(self):
        self.assertEqual(classify_temporal_evidence({
            "temporal_occlusion_supported_for_center": True,
            "temporal_presence_supported_for_center": True,
        }), ("occlusion_support", "defer_2d"))
        self.assertEqual(classify_temporal_evidence({
            "temporal_occlusion_supported_for_center": False,
            "temporal_presence_supported_for_center": True,
        }), ("label_conditioned_presence_support_only", "review_support_only"))
        self.assertEqual(classify_temporal_evidence({}),
                         ("insufficient", "insufficient"))

    def test_frozen_window_rejects_label_pcd_identity_mismatch(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = root / "010" / "labels"
            pcd = root / "010" / "pcd"
            labels.mkdir(parents=True); pcd.mkdir()
            (labels / "100.json").write_text(
                json.dumps({"frame_id": "100"}), encoding="utf-8")
            (pcd / "101.pcd").write_text("pcd", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identities differ"):
                freeze_window(root, "010")

    def test_center_frame_must_exist_in_frozen_window(self):
        jobs = [{"sequence": "010", "center_frame_id": "100"}]
        windows = {"010": (Path("."), [{"frame_id": "101"}])}
        with self.assertRaisesRegex(ValueError, "center frame is absent"):
            validate_window_centers(jobs, windows)


if __name__ == "__main__":
    unittest.main()
