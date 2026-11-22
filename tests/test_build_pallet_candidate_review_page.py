import json
from pathlib import Path
import tempfile
import unittest

from tools.build_pallet_candidate_review_page import build_page


class PalletCandidateReviewPageTest(unittest.TestCase):
    def test_page_is_read_only_and_marks_suppressed_instance(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); bundle = root / "bundle.json"; page = root / "index.html"
            bundle.write_text(json.dumps({
                "schema_version": "labelagent.pallet_visibility_candidate_bundle.v1",
                "production_deployed": False, "human_truth_writes": 0,
                "training_data_writes": 0, "candidate_count": 2,
                "mask_candidates_emitted": 1, "current_2d_candidates_suppressed": 1,
                "previews": [{"sequence": "010", "camera": 0, "frame_id": "9",
                              "path": "previews/x.png"}],
                "rows": [
                    {"sequence": "010", "camera": 0, "obj_id": 1,
                     "mask_path": "masks/1.png", "mask_pixels": 100,
                     "propagation_source_frame_ids": []},
                    {"sequence": "010", "camera": 0, "obj_id": 2,
                     "mask_path": None, "mask_pixels": 0,
                     "propagation_source_frame_ids": ["8", "10"]},
                ]}), encoding="utf-8")
            build_page(bundle, page); text = page.read_text(encoding="utf-8")
            self.assertIn("离线候选 · 未部署 · 不是真值", text)
            self.assertIn("抑制二维掩码", text)
            self.assertIn("8、10", text)
            self.assertNotIn("<button", text)
            self.assertNotIn("写回真值", text)


if __name__ == "__main__":
    unittest.main()
