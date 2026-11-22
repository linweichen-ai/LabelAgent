import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.build_pallet_visibility_candidate_bundle import build, preview_filename
from tools.evaluate_pallet_box3d_roi_sam import digest


class PalletVisibilityCandidateBundleTest(unittest.TestCase):
    def test_preview_name_preserves_frame_identity(self):
        self.assertNotEqual(
            preview_filename("010", 0, "100"),
            preview_filename("010", 0, "200"),
        )

    def test_bundle_emits_only_allowed_mask(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); masks = root / "split_masks"; masks.mkdir()
            image_root = root / "images"; image_root.mkdir()
            image = image_root / "frame.png"
            Image.new("RGB", (640, 512), "gray").save(image)
            mask = np.zeros((512, 640), np.uint8); mask[10:20, 10:20] = 255
            mask_path = masks / "010-frame9-camera0-obj1-depth-split.png"
            Image.fromarray(mask).save(mask_path)
            routes = {"schema_version": "labelagent.pallet_visibility_candidate_routes.v1",
                      "production_deployed": False, "rows": [
                {"sequence": "010", "camera": 0, "frame_id": "9", "obj_id": 1,
                 "route": "depth_split_candidate", "allow_current_2d_candidate": True,
                 "propagation_source_frame_ids": []},
                {"sequence": "010", "camera": 0, "frame_id": "9", "obj_id": 2,
                 "route": "temporal_occlusion_defer_2d",
                 "allow_current_2d_candidate": False,
                 "propagation_source_frame_ids": ["8", "10"]},
            ]}
            split = {"schema_version": "labelagent.pallet_box3d_overlap_pairs.v1",
                     "production_deployed": False, "rows": [{
                "sequence": "010", "camera": 0, "frame_id": "9",
                "depth_split_metrics": [
                    {"obj_id": 1, "candidate_pass": True, "split_pixels": 100},
                    {"obj_id": 2, "candidate_pass": False, "split_pixels": 0}],
                "depth_split_masks": {"1": {"path": str(mask_path),
                                                     "sha256": digest(mask_path)},
                                      "2": {"sha256": "unused"}},
            }]}
            manifest = {"local_root": str(image_root), "files": [{
                "sequence": "010", "camera": "camera_0", "frame_id": "9",
                "relative_path": "frame.png", "sha256": digest(image)}]}
            route_path = root / "routes.json"; split_path = root / "split.json"
            manifest_path = root / "manifest.json"
            route_path.write_text(json.dumps(routes)); split_path.write_text(json.dumps(split))
            manifest_path.write_text(json.dumps(manifest)); output = root / "bundle"
            report = build(route_path, split_path, masks, manifest_path, output)
            self.assertEqual(report["mask_candidates_emitted"], 1)
            self.assertEqual(report["current_2d_candidates_suppressed"], 1)
            self.assertEqual(len(list((output / "masks").glob("*.png"))), 1)
            saved = json.loads((output / "bundle.json").read_text())
            suppressed = next(row for row in saved["rows"] if row["obj_id"] == 2)
            self.assertIsNone(suppressed["mask_path"])
            self.assertEqual(suppressed["propagation_source_frame_ids"], ["8", "10"])
            self.assertEqual(saved["human_truth_writes"], 0)

            recovered_mask = np.zeros((512, 640), np.uint8)
            recovered_mask[30:35, 30:40] = 255
            recovered_path = root / "recovered.png"
            Image.fromarray(recovered_mask).save(recovered_path)
            recovery = {
                "schema_version": "labelagent.pallet_lidar_conflict_points.v1",
                "production_deployed": False, "workbench_writes": 0,
                "rows": [{"sequence": "010", "camera": 0, "frame_id": "9",
                          "target_obj_id": 2, "recovered": True,
                          "selected_policy": "three_positive_all_negative",
                          "selected_mask_path": str(recovered_path),
                          "selected_mask_sha256": digest(recovered_path),
                          "selected_mask_pixels": 50,
                          "selected_positive_points": [[31, 31], [35, 31]],
                          "selected_negative_points": [[40, 40]],
                          "attempts": [{
                              "policy": "three_positive_all_negative",
                              "accepted": True,
                              "split_metrics": [{
                                  "obj_id": 2, "candidate_pass": True,
                                  "split_pixels": 50,
                                  "source_mask_retention": 1.0,
                                  "projected_box_coverage": .2}]}]}]}
            recovery_path = root / "recovery.json"
            recovery_path.write_text(json.dumps(recovery))
            recovered_output = root / "recovered_bundle"
            recovered_report = build(
                route_path, split_path, masks, manifest_path, recovered_output,
                recovery_path)
            self.assertEqual(recovered_report["mask_candidates_emitted"], 2)
            self.assertEqual(recovered_report["current_2d_candidates_suppressed"], 0)
            recovered_bundle = json.loads(
                (recovered_output / "bundle.json").read_text())
            recovered_row = next(
                row for row in recovered_bundle["rows"] if row["obj_id"] == 2)
            self.assertEqual(recovered_row["route"],
                             "lidar_conflict_point_candidate")
            self.assertEqual(recovered_row["mask_pixels"], 50)
            self.assertEqual(recovered_row["recovery_policy"],
                             "three_positive_all_negative")
            self.assertEqual(recovered_row["recovery_split_metrics"]["split_pixels"], 50)
            self.assertEqual(recovered_row["recovery_positive_point_count"], 2)


if __name__ == "__main__":
    unittest.main()
