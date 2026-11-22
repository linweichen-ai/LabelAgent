import io
import unittest

from PIL import Image

from tools.evaluate_ego_tail_vlm_focused_pilot import (
    PANEL_BOXES,
    build_focused_montage,
    extract_review_panels,
    select_pilot_rows,
)


class EgoTailVlmFocusedPilotTests(unittest.TestCase):
    def test_extracts_exact_source_panels(self):
        image = Image.new("RGB", (1240, 360), (1, 2, 3))
        panels = extract_review_panels(image)
        self.assertEqual(set(panels), set(PANEL_BOXES))
        self.assertEqual(panels["context_overlay"].size, (255, 255))
        self.assertEqual(panels["raw_micro"].size, (260, 260))
        self.assertEqual(panels["overlay_micro"].size, (260, 260))

    def test_fails_closed_on_layout_drift(self):
        with self.assertRaisesRegex(ValueError, "unexpected"):
            extract_review_panels(Image.new("RGB", (1239, 360)))

    def test_builds_square_montage(self):
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "review.png"
            Image.new("RGB", (1240, 360), (30, 40, 50)).save(source)
            payload = build_focused_montage(source)
            with Image.open(io.BytesIO(payload)) as image:
                self.assertEqual(image.size, (768, 768))
                self.assertEqual(image.format, "PNG")

    def test_selects_requested_tail_in_requested_order(self):
        policy = {
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False, "production_deployed": False,
            "default_enabled": False, "automatic_decisions_enabled": False,
            "policy": {"camera1_bottom_tail_evidence_coverage": "complete",
                       "camera1_defer_after_rank": 2},
            "counts": {"camera1_bottom_total": 2},
            "rows": [
                {"component_id": "c3", "camera": "camera1",
                 "camera_review_priority_rank": 3,
                 "manual_review_queue": "deferred_manual_review",
                 "policy_reason": "reviewed"},
                {"component_id": "c4", "camera": "camera1",
                 "camera_review_priority_rank": 4,
                 "manual_review_queue": "priority_manual_review",
                 "policy_reason": "reviewed"},
            ],
        }
        review = {
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False, "production_deployed": False,
            "default_enabled": False,
            "rows": [
                {"component_id": "c3", "review_image": "c3.jpg",
                 "review_image_sha256": "a" * 64},
                {"component_id": "c4", "review_image": "c4.jpg",
                 "review_image_sha256": "b" * 64},
            ],
        }
        rows = select_pilot_rows(policy, review, ["c4", "c3"])
        self.assertEqual([row["component_id"] for row in rows], ["c4", "c3"])

    def test_rejects_duplicate_or_outside_ids(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            select_pilot_rows({}, {}, ["c3", "c3"])


if __name__ == "__main__":
    unittest.main()
