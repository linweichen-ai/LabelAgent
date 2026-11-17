"""Score Qwen3-VL slope presence against a bounded non-human visual review."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def score(result: dict, review: dict) -> dict:
    if (review.get("schema_version") != "labelagent.slope-visual-review.v1"
            or review.get("human_truth") is not False
            or review.get("human_confirmed") is not False
            or review.get("split") != "dev8"
            or review.get("source_sequence") != "008"):
        raise ValueError("review is not a bounded non-human dev8 review")
    rows = result.get("per_sample")
    decisions = review.get("decisions")
    if not isinstance(rows, list) or not isinstance(decisions, list):
        raise ValueError("result or review rows are missing")
    predictions = {}
    for row in rows:
        sample_id = row.get("sample_id")
        assessments = (row.get("content") or {}).get("assessments") or {}
        evidence = (assessments.get("slope") or {}).get("evidence")
        if (row.get("status") != "ok" or evidence not in {
                "present", "absent", "uncertain"} or sample_id in predictions):
            raise ValueError("result slope prediction is invalid or duplicated")
        predictions[sample_id] = evidence
    expected = {}
    for row in decisions:
        sample_id = row.get("sample_id")
        visible = row.get("assistant_visible_slope_candidate")
        if not isinstance(visible, bool) or sample_id in expected:
            raise ValueError("review decision is invalid or duplicated")
        expected[sample_id] = visible
    if set(predictions) != set(expected) or len(expected) != 8:
        raise ValueError("result and review sample sets differ")
    tp = fp = fn = tn = uncertain = 0
    details = []
    for sample_id in sorted(expected):
        evidence = predictions[sample_id]
        visible = expected[sample_id]
        if evidence == "uncertain":
            uncertain += 1
        elif evidence == "present" and visible:
            tp += 1
        elif evidence == "present":
            fp += 1
        elif visible:
            fn += 1
        else:
            tn += 1
        details.append({
            "sample_id": sample_id,
            "assistant_visible_slope_candidate": visible,
            "candidate_evidence": evidence,
        })
    return {
        "samples": len(expected),
        "assistant_visible_slope_candidates": sum(expected.values()),
        "true_positive_candidates": tp,
        "false_positive_candidates": fp,
        "false_negative_candidates": fn,
        "true_negative_candidates": tn,
        "uncertain_candidates": uncertain,
        "precision_on_assistant_review": tp / (tp + fp) if tp + fp else None,
        "recall_on_assistant_review": tp / (tp + fn) if tp + fn else None,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = json.loads(args.result.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    summary = score(result, review)
    payload = {
        "schema_version": "labelagent.qwen3vl-slope-review-score.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "assistant_review_scored_not_human_truth",
        "human_truth": False,
        "human_confirmed": False,
        "production_deployed": False,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
