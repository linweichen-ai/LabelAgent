import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.build_pallet_visibility_candidate_bundle import build
from tools.evaluate_pallet_box3d_roi_sam import digest


class PalletVisibilityCandidateBundleTest(unittest.TestCase):
    def test_bundle_emits_only_allowed_mask(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); masks = root / "split_masks"; masks.mkdir()
            image_root = root / "images"; image_root.mkdir()
            image = image_root / "frame.png"
            Image.new("RGB", (640, 512), "gray").save(image)
            mask = np.zeros((512, 640), np.uint8); mask[10:20, 10:20] = 255
            mask_path = masks / "010-camera0-obj1-depth-split.png"
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
                "depth_split_masks": {"1": {"sha256": digest(mask_path)},
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


if __name__ == "__main__":
    unittest.main()
