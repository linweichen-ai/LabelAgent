#!/usr/bin/env python3
"""Audit whether the bounded expert queue covers slope disagreement candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.audit_second_assistant_slope_triage import (
    SCHEMA_VERSION as TRIAGE_SCHEMA,
    audit_second_assistant_triage,
)
from tools.build_mixed_hard_expert_queue import SCHEMA_VERSION as QUEUE_SCHEMA


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_route_coverage(
        packet_dir: Path | str,
        qwen_path: Path | str,
        queue_path: Path | str,
        triage_path: Path | str) -> dict[str, Any]:
    """Return a path-free, candidate-only coverage report.

    The assistant triage is not promoted to truth.  Missed rows are emitted as
    a supplemental *review proposal* only; this function never authorizes or
    dispatches an external request.
    """
    packet_dir, qwen_path, queue_path, triage_path = map(
        Path, (packet_dir, qwen_path, queue_path, triage_path))
    triage_audit = audit_second_assistant_triage(
        packet_dir, qwen_path, triage_path)
    qwen, queue, triage = map(
        _read_json, (qwen_path, queue_path, triage_path))

    if queue.get("schema_version") != QUEUE_SCHEMA:
        raise ValueError("unsupported expert queue schema")
    for key, expected in {
        "external_upload_authorized": False,
        "ready_for_shadow_execution": False,
        "ready_for_quality_ab": False,
        "images_copied": False,
        "image_paths_included": False,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "test24_used": False,
    }.items():
        if queue.get(key) != expected:
            raise ValueError(f"unsafe expert queue boundary: {key}")
    if queue.get("source_suggestions_sha256") != _sha256(qwen_path):
        raise ValueError("expert queue is bound to another Qwen suggestion file")

    qwen_rows = {row.get("sample_id"): row for row in qwen.get("rows", [])}
    if (len(qwen_rows) != 34 or None in qwen_rows
            or qwen.get("samples_total") != 34):
        raise ValueError("Qwen suggestion set is incomplete")
    samples = queue.get("samples")
    if (not isinstance(samples, list)
            or queue.get("sample_count") != len(samples)
            or len({row.get("sample_id") for row in samples}) != len(samples)):
        raise ValueError("expert queue sample count or identity differs")
    queued_ids: set[str] = set()
    for sample in samples:
        sample_id = sample.get("sample_id")
        source = qwen_rows.get(sample_id)
        if source is None:
            raise ValueError(f"unknown expert queue sample: {sample_id}")
        if sample.get("source_image_sha256") != source.get("copied_image_sha256"):
            raise ValueError(f"expert queue image binding differs: {sample_id}")
        if (sample.get("external_upload_authorized") is not False
                or sample.get("external_dispatch_performed") is not False
                or sample.get("human_reference_available") is not False):
            raise ValueError(f"unsafe expert queue sample: {sample_id}")
        queued_ids.add(sample_id)

    triage_rows = triage.get("rows")
    if (triage.get("schema_version") != TRIAGE_SCHEMA
            or not isinstance(triage_rows, list)):
        raise ValueError("unsupported assistant triage schema")
    coverage = Counter()
    missed_by_confidence = Counter()
    flag_totals = Counter()
    flag_misses = Counter()
    missed: list[dict[str, Any]] = []
    for row in triage_rows:
        sample_id = row["sample_id"]
        confidence = row["assistant_confidence"]
        flags = sorted(set(row.get("difficulty_flags") or []))
        routed = sample_id in queued_ids
        coverage[(confidence, "routed" if routed else "missed")] += 1
        flag_totals.update(flags)
        if routed:
            continue
        missed_by_confidence[confidence] += 1
        flag_misses.update(flags)
        source = qwen_rows[sample_id]
        missed.append({
            "sample_id": sample_id,
            "qwen_image_sha256": source["copied_image_sha256"],
            "assistant_review_image_sha256": row["source_image_sha256"],
            "assistant_confidence": confidence,
            "local_qwen_slope": row["local_qwen_slope"],
            "difficulty_flags": flags,
            "proposal_only": True,
            "external_upload_authorized": False,
            "external_dispatch_performed": False,
            "human_reference_available": False,
        })
    missed.sort(key=lambda row: row["sample_id"])
    triage_count = len(triage_rows)
    routed_count = triage_count - len(missed)
    augmented_count = len(queued_ids | {row["sample_id"] for row in missed})
    return {
        "schema_version": "labelagent.mixed-hard-route-coverage-audit.v1",
        "status": ("coverage_gap_detected" if missed
                   else "assistant_disagreements_covered"),
        "source_suggestions_sha256": _sha256(qwen_path),
        "expert_queue_sha256": _sha256(queue_path),
        "assistant_triage_sha256": _sha256(triage_path),
        "review_manifest_sha256": triage_audit["review_manifest_sha256"],
        "source_sample_count": len(qwen_rows),
        "baseline_queue_count": len(queued_ids),
        "baseline_external_call_ratio": round(len(queued_ids) / len(qwen_rows), 6),
        "priority_disagreement_count": triage_count,
        "priority_routed_count": routed_count,
        "priority_missed_count": len(missed),
        "priority_coverage_ratio": round(routed_count / triage_count, 6),
        "high_medium_missed_count": sum(
            missed_by_confidence[level] for level in ("high", "medium")),
        "coverage_by_confidence": {
            level: {
                "routed": coverage[(level, "routed")],
                "missed": coverage[(level, "missed")],
            } for level in ("high", "medium", "low")
        },
        "difficulty_flag_totals": dict(sorted(flag_totals.items())),
        "difficulty_flag_misses": dict(sorted(flag_misses.items())),
        "supplemental_review_proposal_count": len(missed),
        "supplemental_review_proposal": missed,
        "augmented_shadow_candidate_count": augmented_count,
        "augmented_shadow_candidate_ratio": round(
            augmented_count / len(qwen_rows), 6),
        "coverage_gate_passed": not missed,
        "dispatch_performed": False,
        "external_upload_performed": False,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "training_authorized": False,
        "automatic_approval": False,
        "test24_used": False,
        "human_review_required": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--qwen", required=True, type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--triage", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = audit_route_coverage(
            args.packet, args.qwen, args.queue, args.triage)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise ValueError("output already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
