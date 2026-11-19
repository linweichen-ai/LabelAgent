import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.review_pallet_pair_semantics import _validate_decision, review_pairs


def rectangle(shape_id, left, top, right, bottom):
    points = [[left, top], [right, top], [right, bottom], [left, bottom]]
    return {"id": shape_id, "label": "pallet", "points": points,
            "holes": [], "parts": [{"points": points, "holes": []}]}


class PalletPairSemanticReviewTest(unittest.TestCase):
    def test_safe_decision_requires_visible_pallet_and_clear_evidence(self):
        decision = {
            "larger_candidate_is_visible_pallet": False,
            "smaller_candidate_is_visible_pallet_part": True,
            "visual_evidence_is_clear": True,
            "same_physical_instance": "yes",
            "smaller_is_redundant_fragment": True,
            "safe_to_consolidate": True,
            "confidence": .99,
            "reason": "Synthetic invalid combination.",
        }
        with self.assertRaisesRegex(ValueError, "unsafe semantic"):
            _validate_decision(decision)

    def test_review_is_read_only_and_cannot_authorize_automatic_merge(self):
        payload = {"0": [
            rectangle("large", 10, 10, 130, 80),
            rectangle("small", 20, 20, 50, 35),
        ]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.json"
            candidates.write_text(json.dumps(payload), encoding="utf-8")
            before = candidates.read_bytes()
            image = root / "frame.jpg"
            Image.new("RGB", (180, 110), "gray").save(image)
            pairs = root / "pairs.json"
            pairs.write_text(json.dumps({"rows": [{
                "rank": 1, "topic": "/CVAT/Task3/images", "frame": 0,
                "small_id": "small", "large_id": "large",
                "image_path": str(image),
                "conservative_duplicate_gate": True,
            }]}), encoding="utf-8")

            def fake_chat(config, prompt, original, *, extra_images, raw,
                          response_schema):
                self.assertTrue(original)
                self.assertEqual(len(extra_images), 1)
                self.assertIn("same single physical pallet", prompt)
                raw["model_route"] = "test"
                return {
                    "larger_candidate_is_visible_pallet": True,
                    "smaller_candidate_is_visible_pallet_part": True,
                    "visual_evidence_is_clear": True,
                    "same_physical_instance": "yes",
                    "smaller_is_redundant_fragment": True,
                    "safe_to_consolidate": True,
                    "confidence": .95,
                    "reason": "Both masks trace one pallet.",
                }

            report = review_pairs(
                candidates, pairs, root / "output", [1], {}, chat=fake_chat)

            self.assertEqual(candidates.read_bytes(), before)
            self.assertEqual(report["visible_pallet_support_count"], 1)
            self.assertEqual(report["clear_visual_evidence_count"], 1)
            self.assertEqual(report["semantic_safe_count"], 1)
            self.assertFalse(report["automatic_merge_allowed"])
            self.assertFalse(report["human_truth"])
            self.assertTrue(Path(report["report_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
