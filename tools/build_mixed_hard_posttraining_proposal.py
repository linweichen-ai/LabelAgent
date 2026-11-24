#!/usr/bin/env python3
"""Build a fail-closed post-training *proposal* from mixed-hard review.

The output is deliberately not a training dataset and never authorizes a
training run.  It only binds a complete, unambiguous human review to the
review packet so that target authoring and a separate training authorization
can happen later without losing provenance.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.build_mixed_hard_adjudication_workspace import (
    read_json,
    validate_packet,
    validate_submission,
)


SCHEMA_VERSION = "labelagent.mixed-hard-posttraining-proposal.v1"
UNCERTAIN = "uncertain"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_timestamp_with_timezone(value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("reviewed_at is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("reviewed_at must be a valid ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("reviewed_at must include a timezone")


def build_posttraining_proposal(
        packet_dir: Path | str,
        adjudication_path: Path | str) -> dict[str, Any]:
    packet_dir = Path(packet_dir)
    adjudication_path = Path(adjudication_path)
    manifest, manifest_sha256 = validate_packet(packet_dir)
    adjudication = read_json(adjudication_path)
    validate_submission(manifest, manifest_sha256, adjudication)

    if adjudication.get("status") != "declared_human_review_complete":
        raise ValueError("adjudication is not declared complete")
    _require_timestamp_with_timezone(adjudication.get("reviewed_at"))

    review_by_id = {row["sample_id"]: row for row in adjudication["rows"]}
    proposal_rows: list[dict[str, Any]] = []
    for source in manifest["rows"]:
        review = review_by_id[source["sample_id"]]
        ambiguous = [
            key for key in (
                "candidate_decision", "slope_presence", "class_conflict",
                "pallet_quality")
            if review.get(key) == UNCERTAIN
        ]
        if ambiguous:
            raise ValueError(
                f"uncertain review cannot enter a post-training proposal: "
                f"{source['sample_id']} ({', '.join(ambiguous)})")
        proposal_rows.append({
            "sample_id": source["sample_id"],
            "source_image_sha256": source["source_image_sha256"],
            "cohort": source["cohort"],
            "human_review": {
                "candidate_decision": review["candidate_decision"],
                "slope_presence": review["slope_presence"],
                "class_conflict": review["class_conflict"],
                "pallet_quality": review["pallet_quality"],
                "difficulty_flags": list(review.get("difficulty_flags") or []),
            },
            "target_authoring_status": "pending",
            "approved_for_training": False,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pending_target_authoring_and_training_authorization",
        "review_manifest_sha256": manifest_sha256,
        "adjudication_sha256": _sha256(adjudication_path),
        "reviewer_declared_id": adjudication["reviewer_declared_id"],
        "reviewed_at": adjudication["reviewed_at"],
        "sample_count": len(proposal_rows),
        "rows": proposal_rows,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "training_authorized": False,
        "automatic_approval": False,
        "test24_used": False,
        "training_dataset_materialized": False,
        "model_training_started": False,
        "production_deployed": False,
        "required_next_steps": [
            "author model targets from the reviewed decisions",
            "independently review the authored targets",
            "issue a separate explicit training authorization",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--adjudication", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        proposal = build_posttraining_proposal(args.packet, args.adjudication)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({
            "status": "blocked",
            "error": str(exc),
            "approved_for_training": False,
            "model_training_started": False,
        }, ensure_ascii=False), file=sys.stderr)
        return 2
    if args.output.exists():
        raise ValueError("output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(proposal, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": proposal["status"],
        "sample_count": proposal["sample_count"],
        "approved_for_training": False,
        "model_training_started": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
