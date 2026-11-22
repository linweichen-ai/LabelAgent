import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.validate_cvat_human_signoff_import import validate


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class HumanSignoffImportTests(unittest.TestCase):
    def fixture(self, root: Path):
        review_items = []
        rows = []
        for index in range(81):
            candidate_id = f"candidate-{index:02d}"
            identity = {
                "candidate_id": candidate_id,
                "frame_id": index,
                "source_image_sha256": digest(f"image-{index}"),
                "canonical_geometry_sha256": digest(f"geometry-{index}"),
                "focused_image_sha256": digest(f"focused-{index}"),
            }
            review_items.append(identity)
            rows.append(
                {
                    **identity,
                    "decision": ("pass", "reject", "uncertain")[index % 3],
                    "reviewer": "fixture-reviewer",
                    "reviewed_at": "2026-11-22T12:34:56+08:00",
                    "note": None,
                }
            )
        review = root / "review.json"
        review.write_text(
            json.dumps(
                {
                    "schema_version": "labelagent.cvat-manual-instance-review-pack.v1",
                    "status": "pending_independent_review",
                    "test24_intersection_count": 0,
                    "training_authorized": False,
                    "review_items": review_items,
                }
            ),
            encoding="utf-8",
        )
        signoff = root / "signoff.json"
        value = {
            "schema_version": "labelagent.cvat-human-instance-signoff.v1",
            "status": "pending_human_review",
            "review_manifest_sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
            "candidate_count": 81,
            "allowed_decisions": ["pass", "reject", "uncertain"],
            "rows": rows,
            "human_signoff_count": 81,
            "training_authorized": False,
            "automatic_approval": False,
        }
        signoff.write_text(json.dumps(value), encoding="utf-8")
        return review, signoff, value

    def write_signoff(self, path: Path, value: dict) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_valid_complete_signoff_remains_not_training_authorized(self):
        with tempfile.TemporaryDirectory() as directory:
            review, signoff, _ = self.fixture(Path(directory))
            normalized, receipt = validate(review, signoff)
        self.assertEqual(len(normalized["rows"]), 81)
        self.assertEqual(receipt["validated_row_count"], 81)
        self.assertEqual(receipt["human_signoff_count"], 81)
        self.assertEqual(receipt["decision_counts"], {
            "pass": 27, "reject": 27, "uncertain": 27,
        })
        self.assertFalse(normalized["approved_for_training"])
        self.assertFalse(receipt["training_authorized"])
        self.assertNotIn("rows", receipt)
        self.assertFalse(receipt["reviewer_identities_published"])
        self.assertNotIn("rows", receipt)

    def test_error_messages_do_not_publish_private_candidate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            review, signoff, value = self.fixture(Path(directory))
            private_id = value["rows"][0]["candidate_id"]
            value["rows"][0]["decision"] = None
            self.write_signoff(signoff, value)
            with self.assertRaises(ValueError) as captured:
                validate(review, signoff)
        self.assertNotIn(private_id, str(captured.exception))
        self.assertIn("row 0", str(captured.exception))

    def test_empty_signature_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            review, signoff, value = self.fixture(Path(directory))
            value["rows"][0]["reviewer"] = "  "
            self.write_signoff(signoff, value)
            with self.assertRaisesRegex(ValueError, "reviewer is absent"):
                validate(review, signoff)

    def test_bad_timestamp_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            review, signoff, value = self.fixture(Path(directory))
            value["rows"][0]["reviewed_at"] = "2026-02-30T12:00:00+08:00"
            self.write_signoff(signoff, value)
            with self.assertRaisesRegex(ValueError, "timestamp is invalid"):
                validate(review, signoff)

    def test_missing_decision_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            review, signoff, value = self.fixture(Path(directory))
            value["rows"][0]["decision"] = None
            self.write_signoff(signoff, value)
            with self.assertRaisesRegex(ValueError, "decision is absent or invalid"):
                validate(review, signoff)

    def test_hash_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            review, signoff, value = self.fixture(Path(directory))
            value["rows"][0]["focused_image_sha256"] = "0" * 64
            self.write_signoff(signoff, value)
            with self.assertRaisesRegex(ValueError, "binding drift"):
                validate(review, signoff)

    def test_duplicate_conflicting_row_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            review, signoff, value = self.fixture(Path(directory))
            duplicate = dict(value["rows"][0])
            duplicate["decision"] = "reject"
            value["rows"][-1] = duplicate
            self.write_signoff(signoff, value)
            with self.assertRaisesRegex(ValueError, "repeated or conflicting"):
                validate(review, signoff)

    def test_missing_row_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            review, signoff, value = self.fixture(Path(directory))
            value["rows"].pop()
            self.write_signoff(signoff, value)
            with self.assertRaisesRegex(ValueError, "exactly 81 rows"):
                validate(review, signoff)

    def test_unknown_id_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            review, signoff, value = self.fixture(Path(directory))
            value["rows"][-1]["candidate_id"] = "unknown-private-id"
            self.write_signoff(signoff, value)
            with self.assertRaisesRegex(ValueError, "unknown candidate id"):
                validate(review, signoff)

    def test_training_approval_in_input_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            review, signoff, value = self.fixture(Path(directory))
            value["approved_for_training"] = True
            self.write_signoff(signoff, value)
            with self.assertRaisesRegex(ValueError, "must not approve training"):
                validate(review, signoff)


if __name__ == "__main__":
    unittest.main()
