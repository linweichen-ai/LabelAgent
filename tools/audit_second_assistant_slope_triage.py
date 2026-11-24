#!/usr/bin/env python3
"""Audit a hash-bound second-assistant slope disagreement queue."""

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

from tools.build_mixed_hard_adjudication_workspace import validate_packet


SCHEMA_VERSION = "labelagent.local-second-assistant-slope-triage.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_second_assistant_triage(
        packet_dir: Path | str,
        qwen_path: Path | str,
        triage_path: Path | str) -> dict[str, Any]:
    packet_dir, qwen_path, triage_path = (
        Path(packet_dir), Path(qwen_path), Path(triage_path))
    manifest, manifest_sha256 = validate_packet(packet_dir)
    qwen, triage = _read_json(qwen_path), _read_json(triage_path)

    if triage.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported second-assistant triage schema")
    for key, expected in {
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "training_authorized": False,
        "automatic_approval": False,
        "test24_used": False,
        "external_api_calls": 0,
        "factory_data_uploaded": False,
        "production_deployed": False,
        "human_review_required": True,
    }.items():
        if triage.get(key) != expected:
            raise ValueError(f"unsafe second-assistant boundary: {key}")
    if triage.get("review_manifest_sha256") != manifest_sha256:
        raise ValueError("triage is bound to another review manifest")
    if triage.get("local_qwen_suggestions_sha256") != _sha256(qwen_path):
        raise ValueError("triage is bound to another Qwen suggestion file")
    if triage.get("reviewed_sample_count") != len(manifest["rows"]):
        raise ValueError("reviewed sample count differs from manifest")

    manifest_rows = {row["sample_id"]: row for row in manifest["rows"]}
    qwen_rows = {row["sample_id"]: row for row in qwen.get("rows", [])}
    rows = triage.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("second-assistant queue is empty")
    if triage.get("priority_disagreement_count") != len(rows):
        raise ValueError("priority disagreement count differs")

    seen: set[str] = set()
    qwen_states: Counter[str] = Counter()
    confidences: Counter[str] = Counter()
    difficulty_flags: Counter[str] = Counter()
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id in seen or sample_id not in manifest_rows:
            raise ValueError("unknown or duplicate triage sample")
        seen.add(sample_id)
        source = manifest_rows[sample_id]
        if row.get("source_image_sha256") != source.get("source_image_sha256"):
            raise ValueError(f"source image binding differs: {sample_id}")
        qwen_row = qwen_rows.get(sample_id)
        if not qwen_row:
            raise ValueError(f"Qwen suggestion is missing: {sample_id}")
        qwen_slope = (qwen_row.get("suggestion") or {}).get(
            "assessments", {}).get("slope")
        if row.get("local_qwen_slope") != qwen_slope:
            raise ValueError(f"Qwen slope binding differs: {sample_id}")
        if row.get("assistant_slope_candidate") != "present":
            raise ValueError(f"queue contains a non-positive assistant row: {sample_id}")
        confidence = row.get("assistant_confidence")
        if confidence not in {"low", "medium", "high"}:
            raise ValueError(f"invalid assistant confidence: {sample_id}")
        if not str(row.get("rationale") or "").strip():
            raise ValueError(f"assistant rationale is missing: {sample_id}")
        qwen_states[qwen_slope] += 1
        confidences[confidence] += 1
        difficulty_flags.update(row.get("difficulty_flags") or [])

    return {
        "schema_version": "labelagent.local-second-assistant-slope-triage-audit.v1",
        "status": "valid_assistant_disagreement_queue",
        "review_manifest_sha256": manifest_sha256,
        "local_qwen_suggestions_sha256": _sha256(qwen_path),
        "assistant_triage_sha256": _sha256(triage_path),
        "reviewed_sample_count": len(manifest_rows),
        "priority_disagreement_count": len(rows),
        "qwen_slope_states": dict(sorted(qwen_states.items())),
        "assistant_confidences": dict(sorted(confidences.items())),
        "difficulty_flags": dict(sorted(difficulty_flags.items())),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "training_authorized": False,
        "test24_used": False,
        "external_api_calls": 0,
        "factory_data_uploaded": False,
        "human_review_required": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--qwen", required=True, type=Path)
    parser.add_argument("--triage", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        audit = audit_second_assistant_triage(
            args.packet, args.qwen, args.triage)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2
    payload = json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise ValueError("output already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
