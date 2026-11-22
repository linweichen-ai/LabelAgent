import json
from pathlib import Path
import tempfile
import unittest

from tools.audit_imported_cvat_truth_sources import audit


def shape(source: str, label: str = "pallet") -> dict:
    return {
        "id": 1,
        "label": label,
        "points": [[0, 0], [1, 0], [1, 1]],
        "parts": [{"points": [[0, 0], [1, 0], [1, 1]], "holes": []}],
        "holes": [],
        "cvat_metadata": {"source": source},
    }


class ImportedCvatTruthSourceAuditTests(unittest.TestCase):
    def write_source(self, root: Path, value: object) -> Path:
        path = root / "truth.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_mixed_frame_is_not_human_only(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_source(
                Path(directory), {"0": [shape("manual"), shape("auto", "floor")]}
            )
            result = audit(source)
        self.assertEqual(result["status"], "passed_source_inventory")
        self.assertEqual(result["frame_sets"]["mixed_manual_auto"], [0])
        self.assertEqual(result["decision"]["whole_frame_human_truth_candidates"], 0)
        self.assertFalse(result["decision"]["whole_frame_training_eligible"])

    def test_all_manual_frame_still_needs_hash_bound_signoff(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_source(Path(directory), {"7": [shape("manual")]})
            result = audit(source)
        self.assertEqual(result["frame_sets"]["all_manual"], [7])
        self.assertEqual(result["decision"]["whole_frame_human_truth_candidates"], 1)
        self.assertFalse(result["decision"]["whole_frame_training_eligible"])
        self.assertIn("reviewer signoff receipt is absent", result["review_gate"]["receipt_errors"])

    def test_valid_signoff_can_approve_only_all_manual_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_source(root, {"2": [shape("manual")]})
            first = audit(source)
            receipt = root / "receipt.json"
            receipt.write_text(
                json.dumps(
                    {
                        "schema_version": "labelagent.cvat-human-truth-signoff.v1",
                        "source_sha256": first["source"]["sha256"],
                        "approved": True,
                        "reviewer": "reviewer-a",
                        "approved_frame_ids": [2],
                    }
                ),
                encoding="utf-8",
            )
            result = audit(source, receipt)
        self.assertTrue(result["decision"]["whole_frame_training_eligible"])

    def test_unknown_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_source(Path(directory), {"0": [shape("generated")]})
            result = audit(source)
        self.assertEqual(result["status"], "failed_closed")
        self.assertEqual(result["frame_sets"]["contains_unknown_source"], [0])
        self.assertFalse(result["decision"]["whole_frame_training_eligible"])


if __name__ == "__main__":
    unittest.main()
