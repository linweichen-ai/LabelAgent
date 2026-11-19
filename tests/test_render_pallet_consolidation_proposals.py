import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.render_pallet_consolidation_proposals import render_proposals


def rectangle(shape_id, left, top, right, bottom):
    points = [[left, top], [right, top], [right, bottom], [left, bottom]]
    return {"id": shape_id, "label": "pallet", "points": points,
            "holes": [], "parts": [{"points": points, "holes": []}]}


class PalletConsolidationProposalTest(unittest.TestCase):
    def test_renders_safe_union_without_writing_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "frame.jpg"
            Image.new("RGB", (160, 100), "gray").save(image_path)
            payload = {"0": [
                rectangle("large", 10, 10, 120, 80),
                rectangle("small", 20, 20, 45, 35),
            ]}
            candidate_path = root / "candidates.json"
            candidate_path.write_text(json.dumps(payload), encoding="utf-8")
            before = candidate_path.read_bytes()
            candidate_sha = hashlib.sha256(before).hexdigest()
            pair_payload = {
                "rows": [{
                    "rank": 1, "topic": "/CVAT/Task3/images", "frame": 0,
                    "small_id": "small", "large_id": "large",
                    "conservative_duplicate_gate": True,
                    "image_path": str(image_path),
                }]}
            pair_path = root / "pairs.json"
            pair_path.write_text(json.dumps(pair_payload), encoding="utf-8")
            pair_sha = hashlib.sha256(pair_path.read_bytes()).hexdigest()
            semantic_payload = {
                "candidate_sha256": candidate_sha,
                "pair_report": str(pair_path),
                "pair_report_sha256": pair_sha,
                "results": [{
                    "rank": 1, "topic": "/CVAT/Task3/images", "frame": 0,
                    "small_id": "small", "large_id": "large",
                    "geometry_gate": True,
                    "decision": {
                        "larger_candidate_is_visible_pallet": True,
                        "smaller_candidate_is_visible_pallet_part": True,
                        "visual_evidence_is_clear": True,
                        "same_physical_instance": "yes",
                        "smaller_is_redundant_fragment": True,
                        "safe_to_consolidate": True,
                        "confidence": .95,
                    },
                }],
            }
            semantic_path = root / "semantic.json"
            semantic_path.write_text(json.dumps(semantic_payload), encoding="utf-8")

            report = render_proposals(
                candidate_path, semantic_path, root / "output")

            self.assertEqual(candidate_path.read_bytes(), before)
            self.assertEqual(report["proposal_count"], 1)
            self.assertEqual(report["proposal_gate_pass_count"], 1)
            self.assertFalse(report["automatic_apply_allowed"])
            self.assertFalse(report["human_truth"])
            self.assertTrue(Path(report["contact_sheet"]).is_file())
            self.assertTrue(Path(report["report_path"]).is_file())

    def test_rejects_candidate_file_not_bound_to_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_path = root / "candidates.json"
            candidate_path.write_text("{}", encoding="utf-8")
            semantic_path = root / "semantic.json"
            semantic_path.write_text(json.dumps({
                "candidate_sha256": "wrong", "pair_report": "missing",
                "pair_report_sha256": "wrong", "results": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not bound"):
                render_proposals(candidate_path, semantic_path, root / "output")


if __name__ == "__main__":
    unittest.main()
