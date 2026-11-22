import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image

from tools.build_ego_component_union_review import file_sha256
from tools.build_ego_component_priority_sheet import build_sheet


class EgoComponentPrioritySheetTests(unittest.TestCase):
    def _fixture(self, root: Path):
        review_dir = root / "review"
        review_dir.mkdir()
        image = review_dir / "c1-review.png"
        Image.new("RGB", (40, 30), (20, 40, 60)).save(image)
        review_path = review_dir / "component_review.json"
        review_path.write_text(json.dumps({
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False, "production_deployed": False,
            "default_enabled": False,
            "rows": [{"component_id": "c1", "review_image": image.name,
                      "review_image_sha256": file_sha256(image)}],
        }), encoding="utf-8")
        ranking_path = root / "ranking.json"
        ranking_path.write_text(json.dumps({
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False, "production_deployed": False,
            "default_enabled": False, "automatic_decisions_enabled": False,
            "component_review_sha256": file_sha256(review_path),
            "rows": [{"component_id": "c1", "review_priority_rank": 1,
                      "camera": "camera1",
                      "camera_review_priority_rank": 1,
                      "review_priority_score": 0.9,
                      "review_priority_band": "priority_1",
                      "priority_reasons": ["low_light_and_low_sharpness"],
                      "automatic_decision": None,
                      "review_image_sha256": file_sha256(image)}],
        }), encoding="utf-8")
        return ranking_path, review_path, image

    def test_rejects_non_positive_top_count_before_reading_inputs(self):
        with self.assertRaisesRegex(ValueError, "top count must be positive"):
            build_sheet(None, None, None, top_count=0)  # type: ignore[arg-type]

    def test_builds_hash_bound_sheet_without_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ranking, review, _ = self._fixture(root)
            packet = build_sheet(ranking, review, root / "output")
            self.assertEqual(packet["selected_count"], 1)
            self.assertIsNone(packet["rows"][0]["automatic_decision"])
            self.assertTrue((root / "output" / packet["sheet"]).is_file())

    def test_tampered_review_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ranking, review, image = self._fixture(root)
            Image.new("RGB", (40, 30), (255, 0, 0)).save(image)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                build_sheet(ranking, review, root / "output")

    def test_render_failure_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ranking, review, _ = self._fixture(root)
            with mock.patch(
                    "tools.build_ego_component_priority_sheet._write_sheet",
                    side_effect=OSError("simulated render failure")):
                with self.assertRaisesRegex(OSError, "simulated render failure"):
                    build_sheet(ranking, review, root / "output")
            self.assertFalse((root / "output").exists())
            self.assertEqual(list(root.glob(".output.staging-*")), [])

    def test_rejects_unknown_selection_mode(self):
        with self.assertRaisesRegex(ValueError, "unknown selection mode"):
            build_sheet(None, None, None, selection_mode="bad")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
