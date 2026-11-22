"""Validate a complete, hash-bound CVAT human instance signoff.

The validator is deliberately a one-way safety gate: it verifies a private
human signoff against the frozen 81-row review manifest and emits only a
privacy-safe receipt.  It never fills decisions and never authorizes training.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


REVIEW_SCHEMA = "labelagent.cvat-manual-instance-review-pack.v1"
SIGNOFF_SCHEMA = "labelagent.cvat-human-instance-signoff.v1"
RECEIPT_SCHEMA = "labelagent.cvat-human-instance-signoff-import-receipt.v1"
EXPECTED_COUNT = 81
ALLOWED_DECISIONS = ("pass", "reject", "uncertain")
BOUND_FIELDS = (
    "frame_id",
    "source_image_sha256",
    "canonical_geometry_sha256",
    "focused_image_sha256",
)
ISO_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)
SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
MAX_INPUT_BYTES = 4 * 1024 * 1024


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON object contains duplicate keys")
        result[key] = value
    return result


def _read_object(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError(f"{label} exceeds the input size limit")
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or ISO_TIMESTAMP.fullmatch(value) is None:
        return False
    if not value.endswith("Z"):
        offset = value[-6:]
        if int(offset[1:3]) > 23 or int(offset[4:6]) > 59:
            return False
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _review_identity(review: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if (
        review.get("schema_version") != REVIEW_SCHEMA
        or review.get("status") != "pending_independent_review"
        or review.get("test24_intersection_count") != 0
        or review.get("training_authorized") is not False
    ):
        raise ValueError("review manifest identity or isolation boundary differs")
    items = review.get("review_items")
    if not isinstance(items, list) or len(items) != EXPECTED_COUNT:
        raise ValueError(f"review manifest must contain exactly {EXPECTED_COUNT} items")

    result: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("review item must be an object")
        candidate_id = item.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("review candidate id is invalid")
        if candidate_id in result:
            raise ValueError("review candidate id is repeated")
        frame_id = item.get("frame_id")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            raise ValueError(f"review candidate frame id is invalid at row {index}")
        for field in BOUND_FIELDS[1:]:
            if not _valid_sha256(item.get(field)):
                raise ValueError(f"review candidate binding is invalid: {field}")
        result[candidate_id] = {
            "frame_id": frame_id,
            **{field: item[field].lower() for field in BOUND_FIELDS[1:]},
        }
    return result


def validate(
    review_manifest_path: Path,
    signoff_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the private signoff and return a normalized private value and safe receipt."""
    review_manifest_path = review_manifest_path.resolve()
    signoff_path = signoff_path.resolve()
    review, review_digest = _read_object(review_manifest_path, "review manifest")
    expected = _review_identity(review)
    signoff, signoff_digest = _read_object(signoff_path, "human signoff")

    if signoff.get("schema_version") != SIGNOFF_SCHEMA:
        raise ValueError("human signoff schema differs")
    if signoff.get("review_manifest_sha256") != review_digest:
        raise ValueError("human signoff review manifest hash differs")
    if signoff.get("candidate_count") != EXPECTED_COUNT:
        raise ValueError("human signoff candidate count differs")
    if signoff.get("human_signoff_count") != EXPECTED_COUNT:
        raise ValueError("human signoff count must equal the complete candidate count")
    if signoff.get("allowed_decisions") != list(ALLOWED_DECISIONS):
        raise ValueError("human signoff allowed decisions differ")
    if signoff.get("training_authorized") is not False:
        raise ValueError("human signoff must not authorize training")
    approved_for_training = signoff.get("approved_for_training")
    if approved_for_training is not None and approved_for_training is not False:
        raise ValueError("human signoff must not approve training")
    if signoff.get("automatic_approval") is not False:
        raise ValueError("human signoff automatic approval must remain disabled")

    rows = signoff.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_COUNT:
        raise ValueError(f"human signoff must contain exactly {EXPECTED_COUNT} rows")

    normalized_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    decisions: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("human signoff row must be an object")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("human signoff candidate id is invalid")
        if candidate_id in seen:
            raise ValueError(f"human signoff candidate is repeated or conflicting at row {index}")
        if candidate_id not in expected:
            raise ValueError(f"human signoff contains unknown candidate id at row {index}")
        seen.add(candidate_id)

        identity = expected[candidate_id]
        for field in BOUND_FIELDS:
            actual = row.get(field)
            if field == "frame_id":
                if isinstance(actual, bool) or not isinstance(actual, int) or actual < 0:
                    raise ValueError(
                        f"human signoff binding is invalid at row {index}: {field}"
                    )
            else:
                if not _valid_sha256(actual):
                    raise ValueError(f"human signoff binding is invalid at row {index}: {field}")
                actual = actual.lower()
            if actual != identity[field]:
                raise ValueError(f"human signoff binding drift at row {index}: {field}")

        decision = row.get("decision")
        if decision not in ALLOWED_DECISIONS:
            raise ValueError(f"human signoff decision is absent or invalid at row {index}")
        reviewer = row.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError(f"human signoff reviewer is absent at row {index}")
        reviewed_at = row.get("reviewed_at")
        if not _valid_timestamp(reviewed_at):
            raise ValueError(f"human signoff timestamp is invalid at row {index}")
        for field in (
            "approved_for_training",
            "training_authorized",
            "automatic_approval",
        ):
            value = row.get(field)
            if value is not None and value is not False:
                raise ValueError(
                    f"human signoff row must not approve training at row {index}"
                )

        note = row.get("note")
        if note is not None and not isinstance(note, str):
            raise ValueError(f"human signoff note is invalid at row {index}")
        decisions[decision] += 1
        normalized_rows.append(
            {
                "candidate_id": candidate_id,
                **identity,
                "decision": decision,
                "reviewer": reviewer.strip(),
                "reviewed_at": reviewed_at,
                "note": note,
                "approved_for_training": False,
            }
        )

    missing = set(expected) - seen
    if missing:
        raise ValueError(f"human signoff is missing {len(missing)} review rows")

    human_pass_candidate_ids = sorted(
        row["candidate_id"] for row in normalized_rows if row["decision"] == "pass"
    )
    normalized = {
        "schema_version": "labelagent.cvat-human-instance-signoff-import.v1",
        "status": "validated_human_signoff_not_training_authorized",
        "review_manifest_sha256": review_digest,
        "source_signoff_sha256": signoff_digest,
        "candidate_count": EXPECTED_COUNT,
        "human_signoff_count": EXPECTED_COUNT,
        "decision_counts": {key: decisions[key] for key in ALLOWED_DECISIONS},
        "human_pass_candidate_ids": human_pass_candidate_ids,
        "human_pass_count": len(human_pass_candidate_ids),
        "rows": normalized_rows,
        "approved_for_training": False,
        "training_authorized": False,
        "automatic_approval": False,
    }
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": normalized["status"],
        "review_manifest_sha256": review_digest,
        "source_signoff_sha256": normalized["source_signoff_sha256"],
        "candidate_count": EXPECTED_COUNT,
        "validated_row_count": EXPECTED_COUNT,
        "human_signoff_count": EXPECTED_COUNT,
        "decision_counts": normalized["decision_counts"],
        "human_pass_count": normalized["human_pass_count"],
        "approved_for_training": False,
        "training_authorized": False,
        "automatic_approval": False,
        "private_candidate_ids_published": False,
        "private_geometry_published": False,
        "raw_images_published": False,
        "reviewer_identities_published": False,
    }
    return normalized, receipt


def _write_fresh_outputs(
    private_output: Path,
    receipt_output: Path,
    normalized: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    if private_output == receipt_output:
        raise ValueError("private and receipt output paths must differ")
    private_payload = (
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    final_receipt = {
        **receipt,
        "private_import_sha256": hashlib.sha256(private_payload).hexdigest(),
    }
    receipt_payload = (
        json.dumps(final_receipt, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    created: list[Path] = []
    try:
        private_output.parent.mkdir(parents=True, exist_ok=True)
        receipt_output.parent.mkdir(parents=True, exist_ok=True)
        with private_output.open("xb") as stream:
            created.append(private_output)
            stream.write(private_payload)
            stream.flush()
            os.fsync(stream.fileno())
        with receipt_output.open("xb") as stream:
            created.append(receipt_output)
            stream.write(receipt_payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception as error:
        rollback_failed = False
        for path in reversed(created):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                rollback_failed = True
        if rollback_failed:
            raise OSError("output rollback failed") from error
        raise
    return final_receipt


def _validate_output_paths(
    private_output: Path,
    receipt_output: Path,
    repository_root: Path,
) -> None:
    if private_output.exists() or receipt_output.exists():
        raise ValueError("choose fresh output paths")
    if private_output == receipt_output:
        raise ValueError("private and receipt output paths must differ")
    private_root = (repository_root / ".repro").resolve()
    if not private_output.is_relative_to(private_root):
        raise ValueError("private output must stay under the repository .repro directory")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-manifest", required=True, type=Path)
    parser.add_argument("--human-signoff", required=True, type=Path)
    parser.add_argument("--private-output", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    args = parser.parse_args()

    private_output = args.private_output.resolve()
    receipt_output = args.receipt_output.resolve()
    repository_root = Path(__file__).resolve().parents[1]
    _validate_output_paths(private_output, receipt_output, repository_root)
    normalized, receipt = validate(args.review_manifest, args.human_signoff)
    final_receipt = _write_fresh_outputs(
        private_output, receipt_output, normalized, receipt
    )
    print(json.dumps(final_receipt, ensure_ascii=False))
    return 0


def cli() -> int:
    try:
        return main()
    except ValueError as error:
        print(f"signoff import rejected: {error}", file=sys.stderr)
        return 2
    except OSError:
        print("signoff import failed during a protected I/O operation", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(cli())
