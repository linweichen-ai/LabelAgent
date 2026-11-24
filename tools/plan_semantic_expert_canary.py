#!/usr/bin/env python3
"""Freeze a path-free, unauthorized technical canary plan from a cascade."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_combined_expert_shadow import stratified_canary_order


SHA256 = re.compile(r"^[0-9a-f]{64}$")
EVALUATION_SCHEMA = "labelagent.local-qwen-prompt-cascade-evaluation.v1"
SCHEMA_VERSION = "labelagent.semantic-expert-canary-plan.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("semantic expert queue is empty")
    seen: set[str] = set()
    validated = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("semantic expert queue row is invalid")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError("semantic expert queue sample identity is invalid")
        seen.add(sample_id)
        if (not isinstance(row.get("cohort"), str) or not row["cohort"]
                or SHA256.fullmatch(str(row.get("source_image_sha256") or "")) is None
                or SHA256.fullmatch(str(row.get("copied_image_sha256") or "")) is None):
            raise ValueError("semantic expert queue binding is invalid")
        if (row.get("external_upload_authorized") is not False
                or row.get("candidate_only") is not True):
            raise ValueError("semantic expert queue is already authorized or unsafe")
        if any("path" in str(key).casefold() for key in row):
            raise ValueError("semantic expert queue must remain path-free")
        validated.append(dict(row))
    return validated


def build_canary_plan(evaluation_path: Path | str, *, canary_count: int = 8) -> dict[str, Any]:
    evaluation_path = Path(evaluation_path)
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    context = payload.get("semantic_context_cascade") or {}
    if (payload.get("schema_version") != EVALUATION_SCHEMA
            or payload.get("candidate_only") is not True
            or payload.get("human_truth") is not False
            or payload.get("approved_for_training") is not False
            or payload.get("test24_used") is not False
            or payload.get("production_deployed") is not False
            or payload.get("external_upload_performed") is not False
            or context.get("external_authorization_granted") is not False):
        raise ValueError("cascade evaluation boundary is unsafe")
    rows = _validate_rows(context.get("external_queue"))
    if (context.get("max_external_calls_if_authorized") != len(rows)
            or context.get("cascade_external_count") != len(rows)):
        raise ValueError("cascade expert queue count differs")
    ordered = stratified_canary_order(rows, canary_count)
    canary, remainder = ordered[:canary_count], ordered[canary_count:]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "authorization_required",
        "cascade_evaluation_sha256": _sha256(evaluation_path),
        "external_candidate_count": len(rows),
        "canary_count": len(canary),
        "remainder_count": len(remainder),
        "canary": canary,
        "remainder": remainder,
        "technical_gate": {
            "all_requests_passed": True,
            "usage_complete": True,
            "maximum_median_latency_ms": 60000,
            "maximum_average_tokens_per_call": 30000,
            "automatic_retry_enabled": False,
            "maximum_attempts_per_image": 1,
        },
        "external_authorization_granted": False,
        "dispatch_performed": False,
        "external_upload_performed": False,
        "factory_data_uploaded": False,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "test24_used": False,
        "production_deployed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--canary-count", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = build_canary_plan(
            args.evaluation, canary_count=args.canary_count)
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
