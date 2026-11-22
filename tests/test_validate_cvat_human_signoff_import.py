from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest

from tools.validate_cvat_human_signoff_import import (
    MAX_INPUT_BYTES,
    _validate_output_paths,
    _write_fresh_outputs,
    cli,
    validate,
)


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
        self.assertEqual(normalized["human_pass_count"], 27)
        self.assertEqual(len(normalized["human_pass_candidate_ids"]), 27)
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

    def test_duplicate_json_keys_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            review, signoff, _ = self.fixture(Path(directory))
            raw = signoff.read_text(encoding="utf-8")
            signoff.write_text(
                raw.replace(
                    '"candidate_count": 81',
                    '"candidate_count": 81, "candidate_count": 81',
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate keys"):
                validate(review, signoff)

    def test_oversized_input_fails_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            review, signoff, _ = self.fixture(Path(directory))
            signoff.write_bytes(b" " * (MAX_INPUT_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "input size limit"):
                validate(review, signoff)

    def test_duplicate_identical_row_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            review, signoff, value = self.fixture(Path(directory))
            value["rows"][-1] = dict(value["rows"][0])
            self.write_signoff(signoff, value)
            with self.assertRaisesRegex(ValueError, "repeated or conflicting"):
                validate(review, signoff)

    def test_invalid_timezone_offset_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            review, signoff, value = self.fixture(Path(directory))
            value["rows"][0]["reviewed_at"] = "2026-11-22T12:34:56+24:00"
            self.write_signoff(signoff, value)
            with self.assertRaisesRegex(ValueError, "timestamp is invalid"):
                validate(review, signoff)

    def test_each_binding_is_required_and_hash_bound(self):
        fields = (
            "candidate_id",
            "frame_id",
            "source_image_sha256",
            "canonical_geometry_sha256",
            "focused_image_sha256",
        )
        for field in fields:
            with self.subTest(missing=field), tempfile.TemporaryDirectory() as directory:
                review, signoff, value = self.fixture(Path(directory))
                del value["rows"][0][field]
                self.write_signoff(signoff, value)
                with self.assertRaises(ValueError):
                    validate(review, signoff)
        for field in fields[1:]:
            with self.subTest(drift=field), tempfile.TemporaryDirectory() as directory:
                review, signoff, value = self.fixture(Path(directory))
                value["rows"][0][field] = 999 if field == "frame_id" else "0" * 64
                self.write_signoff(signoff, value)
                with self.assertRaisesRegex(ValueError, "binding drift"):
                    validate(review, signoff)

    def test_human_pass_set_is_private_and_inputs_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            review, signoff, value = self.fixture(Path(directory))
            before = (review.read_bytes(), signoff.read_bytes())
            normalized, receipt = validate(review, signoff)
            expected = sorted(
                row["candidate_id"] for row in value["rows"] if row["decision"] == "pass"
            )
            self.assertEqual(normalized["human_pass_candidate_ids"], expected)
            self.assertEqual(receipt["human_pass_count"], len(expected))
            self.assertNotIn("human_pass_candidate_ids", receipt)
            self.assertEqual(
                normalized["source_signoff_sha256"],
                hashlib.sha256(before[1]).hexdigest(),
            )
            self.assertEqual(before, (review.read_bytes(), signoff.read_bytes()))
            self.assertTrue(
                all(row["approved_for_training"] is False for row in normalized["rows"])
            )

    def test_invalid_decisions_and_incomplete_signatures_fail_closed(self):
        invalid_fields = {
            "decision": (None, "", "approve", True),
            "reviewer": (None, "", "  ", 42),
            "reviewed_at": (
                None,
                "",
                "2026-11-22",
                "2026-11-22T12:00:00",
                "2026-11-22T12:00:00+01:99",
            ),
        }
        for field, invalid_values in invalid_fields.items():
            for invalid in invalid_values:
                with self.subTest(field=field, value=invalid), tempfile.TemporaryDirectory() as directory:
                    review, signoff, value = self.fixture(Path(directory))
                    value["rows"][0][field] = invalid
                    self.write_signoff(signoff, value)
                    with self.assertRaises(ValueError):
                        validate(review, signoff)

    def test_same_output_path_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "same.json"
            with self.assertRaisesRegex(ValueError, "paths must differ"):
                _write_fresh_outputs(output, output, {"private": True}, {"public": True})
            self.assertFalse(output.exists())

    def test_fresh_outputs_share_the_same_private_payload_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / "private.json"
            receipt = root / "receipt.json"
            final_receipt = _write_fresh_outputs(
                private,
                receipt,
                {"private": True},
                {"public": True},
            )
            self.assertEqual(
                final_receipt["private_import_sha256"],
                hashlib.sha256(private.read_bytes()).hexdigest(),
            )
            self.assertEqual(json.loads(receipt.read_text(encoding="utf-8")), final_receipt)

    def test_second_output_failure_rolls_back_first_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / "private.json"
            receipt = root / "existing-receipt.json"
            receipt.write_text("protected", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _write_fresh_outputs(
                    private,
                    receipt,
                    {"private": True},
                    {"public": True},
                )
            self.assertFalse(private.exists())
            self.assertEqual(receipt.read_text(encoding="utf-8"), "protected")

    def test_training_approval_in_input_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            review, signoff, value = self.fixture(Path(directory))
            value["approved_for_training"] = True
            self.write_signoff(signoff, value)
            with self.assertRaisesRegex(ValueError, "must not approve training"):
                validate(review, signoff)

    def test_row_level_training_approval_injection_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            review, signoff, value = self.fixture(Path(directory))
            value["rows"][0]["training_authorized"] = True
            self.write_signoff(signoff, value)
            with self.assertRaisesRegex(ValueError, "must not approve training"):
                validate(review, signoff)

    def test_contradictory_human_signoff_count_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            review, signoff, value = self.fixture(Path(directory))
            value["human_signoff_count"] = 0
            self.write_signoff(signoff, value)
            with self.assertRaisesRegex(ValueError, "count must equal"):
                validate(review, signoff)

    def test_private_output_outside_repro_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaisesRegex(ValueError, "must stay under"):
                _validate_output_paths(
                    root / "private.json",
                    root / "receipt.json",
                    Path(__file__).resolve().parents[1],
                )

    def test_cli_error_hides_sensitive_input_path_and_traceback(self):
        repository_root = Path(__file__).resolve().parents[1]
        sensitive = Path(tempfile.gettempdir()) / "private-person-name-signoff.json"
        private = repository_root / ".repro" / "cli-error-private-never-written.json"
        receipt = repository_root / ".repro" / "cli-error-receipt-never-written.json"
        stderr = io.StringIO()
        argv = [
            "validate_cvat_human_signoff_import.py",
            "--review-manifest",
            str(sensitive),
            "--human-signoff",
            str(sensitive),
            "--private-output",
            str(private),
            "--receipt-output",
            str(receipt),
        ]
        with patch("sys.argv", argv), redirect_stderr(stderr):
            self.assertEqual(cli(), 2)
        message = stderr.getvalue()
        self.assertNotIn(str(sensitive), message)
        self.assertNotIn("Traceback", message)
        self.assertFalse(private.exists())
        self.assertFalse(receipt.exists())


if __name__ == "__main__":
    unittest.main()
