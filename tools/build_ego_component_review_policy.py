#!/usr/bin/env python3
"""Freeze a conservative, offline manual-review policy for ego components."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


POLICY_BOUNDARY = (
    "Assistant-only same-dataset evidence. This policy only orders manual "
    "review; it never auto-accepts, creates human truth, approves training, "
    "or changes production behavior."
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_inputs(ranking: dict, evaluation: dict) -> tuple[list[dict], dict[str, int]]:
    _require(ranking.get("candidate_only") is True, "ranking is not candidate-only")
    _require(ranking.get("human_truth") is False, "ranking incorrectly claims human truth")
    _require(ranking.get("automatic_decisions_enabled") is False,
             "ranking enables automatic decisions")
    rows = ranking.get("rows")
    _require(isinstance(rows, list) and rows, "ranking rows are missing")
    _require(ranking.get("component_count") == len(rows), "component count mismatch")

    _require(evaluation.get("assistant_review_only") is True,
             "evaluation is not assistant-review-only")
    _require(evaluation.get("human_truth") is False,
             "evaluation incorrectly claims human truth")
    _require(evaluation.get("reviewer_blinded_to_risk_band") is True,
             "reviewer was not blinded to the risk band")
    _require(evaluation.get("hard_reject_enrichment_observed") is True,
             "hard-reject enrichment was not observed")
    _require(evaluation.get("top_plus_middle_likely_reject_capture_rate") == 1.0,
             "top+middle did not capture every likely reject")

    by_camera: dict[str, list[dict]] = {"camera1": [], "camera3": []}
    seen = set()
    for row in rows:
        _require(isinstance(row, dict), "ranking row is not an object")
        component_id = row.get("component_id")
        camera = row.get("camera")
        rank = row.get("camera_review_priority_rank")
        _require(isinstance(component_id, str) and component_id,
                 "component id is missing")
        _require(component_id not in seen, f"duplicate component id: {component_id}")
        seen.add(component_id)
        _require(camera in by_camera, f"unsupported camera: {camera}")
        _require(isinstance(rank, int) and rank > 0,
                 f"invalid camera rank: {component_id}")
        _require(row.get("automatic_decision") is None,
                 f"automatic decision present: {component_id}")
        by_camera[camera].append(row)

    counts = {camera: len(camera_rows) for camera, camera_rows in by_camera.items()}
    _require(all(counts.values()), "both camera1 and camera3 are required")
    expected_counts = ranking.get("camera_component_counts")
    _require(expected_counts == counts, "camera component counts mismatch")
    for camera, camera_rows in by_camera.items():
        ranks = sorted(int(row["camera_review_priority_rank"]) for row in camera_rows)
        _require(ranks == list(range(1, counts[camera] + 1)),
                 f"{camera} ranks are not contiguous")
    return rows, counts


def build_policy(ranking: dict, evaluation: dict, *, ranking_sha256: str,
                 evaluation_sha256: str) -> dict:
    rows, counts = _validate_inputs(ranking, evaluation)
    camera1_defer_after = math.floor(counts["camera1"] * 0.80)
    policy_rows = []
    for row in rows:
        camera = row["camera"]
        camera_rank = int(row["camera_review_priority_rank"])
        if camera == "camera1" and camera_rank > camera1_defer_after:
            queue = "deferred_manual_review"
            reason = "camera1_bottom_20pct_assistant_blind_audit"
        else:
            queue = "priority_review"
            reason = ("camera3_all_bands_remain_difficult" if camera == "camera3"
                      else "camera1_not_in_deferred_tail")
        policy_rows.append({
            "component_id": row["component_id"],
            "sample_id": row.get("sample_id"),
            "camera": camera,
            "camera_review_priority_rank": camera_rank,
            "review_priority_score": row.get("review_priority_score"),
            "manual_review_queue": queue,
            "policy_reason": reason,
            "automatic_decision": None,
            "human_truth": False,
            "training_eligible": False,
            "production_deployed": False,
        })

    priority = sum(row["manual_review_queue"] == "priority_review"
                   for row in policy_rows)
    deferred = len(policy_rows) - priority
    return {
        "schema_version": 1,
        "candidate_only": True,
        "assistant_review_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "production_deployed": False,
        "default_enabled": False,
        "automatic_decisions_enabled": False,
        "ranking_sha256": ranking_sha256,
        "evaluation_sha256": evaluation_sha256,
        "policy": {
            "camera3": "priority_review_all",
            "camera1": "defer_bottom_20pct_for_manual_review_only",
            "camera1_defer_after_rank": camera1_defer_after,
            "unseen_or_invalid_input": "fail_closed",
        },
        "counts": {
            "total": len(policy_rows),
            "priority_review": priority,
            "deferred_manual_review": deferred,
            "automatic_decisions": 0,
            "by_camera": counts,
        },
        "policy_boundary": POLICY_BOUNDARY,
        "rows": policy_rows,
    }


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", delete=False,
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    ranking = json.loads(args.ranking.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    payload = build_policy(
        ranking, evaluation,
        ranking_sha256=file_sha256(args.ranking),
        evaluation_sha256=file_sha256(args.evaluation),
    )
    atomic_write_json(args.output, payload)
    print(json.dumps(payload["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
