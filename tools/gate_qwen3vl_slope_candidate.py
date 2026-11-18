"""Gate a slope-focused Qwen3-VL candidate without promoting assistant review to truth."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


SCHEMA = "labelagent.qwen3vl-slope-review-score.v1"


def _validated_summary(payload: dict) -> dict:
    if (payload.get("schema_version") != SCHEMA
            or payload.get("human_truth") is not False
            or payload.get("human_confirmed") is not False
            or payload.get("production_deployed") is not False):
        raise ValueError("score is not a bounded non-human slope review")
    summary = payload.get("summary")
    details = summary.get("details") if isinstance(summary, dict) else None
    if not isinstance(details, list) or not details:
        raise ValueError("score details are missing")
    sample_ids = [row.get("sample_id") for row in details]
    if (any(not isinstance(item, str) or not item for item in sample_ids)
            or len(sample_ids) != len(set(sample_ids))):
        raise ValueError("score sample IDs are invalid or duplicated")
    for name in ("true_positive_candidates", "false_positive_candidates",
                 "false_negative_candidates", "true_negative_candidates",
                 "uncertain_candidates"):
        if type(summary.get(name)) is not int or summary[name] < 0:
            raise ValueError(f"score count {name} is invalid")
    return summary


def gate(baseline: dict, candidate: dict) -> dict:
    before = _validated_summary(baseline)
    after = _validated_summary(candidate)
    before_rows = {row["sample_id"]: row for row in before["details"]}
    after_rows = {row["sample_id"]: row for row in after["details"]}
    if set(before_rows) != set(after_rows):
        raise ValueError("baseline and candidate sample sets differ")
    for sample_id in before_rows:
        if (before_rows[sample_id].get("assistant_visible_slope_candidate")
                is not after_rows[sample_id].get("assistant_visible_slope_candidate")):
            raise ValueError("assistant review changed between baseline and candidate")

    guards = {
        "false_positives_not_increased": (
            after["false_positive_candidates"] <= before["false_positive_candidates"]),
        "true_positives_increased": (
            after["true_positive_candidates"] > before["true_positive_candidates"]),
        "false_negatives_reduced": (
            after["false_negative_candidates"] < before["false_negative_candidates"]),
        "uncertain_not_increased": (
            after["uncertain_candidates"] <= before["uncertain_candidates"]),
    }
    passed = all(guards.values())
    return {
        "status": ("assistant_candidate_passed_for_human_review" if passed
                   else "assistant_candidate_rejected"),
        "passed": passed,
        "human_truth": False,
        "human_confirmed": False,
        "production_deployed": False,
        "production_approved": False,
        "test24_authorized": False,
        "guards": guards,
        "baseline": {
            "true_positive_candidates": before["true_positive_candidates"],
            "false_positive_candidates": before["false_positive_candidates"],
            "false_negative_candidates": before["false_negative_candidates"],
            "uncertain_candidates": before["uncertain_candidates"],
        },
        "candidate": {
            "true_positive_candidates": after["true_positive_candidates"],
            "false_positive_candidates": after["false_positive_candidates"],
            "false_negative_candidates": after["false_negative_candidates"],
            "uncertain_candidates": after["uncertain_candidates"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = gate(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")))
    receipt = {
        "schema_version": "labelagent.qwen3vl-slope-candidate-gate.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
