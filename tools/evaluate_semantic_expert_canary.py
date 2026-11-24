#!/usr/bin/env python3
"""Evaluate a bounded semantic-expert canary without dispatching more calls."""

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

from tools.build_mixed_hard_expert_queue import SCHEMA_VERSION as QUEUE_SCHEMA
from tools.plan_semantic_expert_canary import SCHEMA_VERSION as PLAN_SCHEMA
from tools.run_combined_expert_shadow import canary_guard
from tools.run_mixed_hard_external_shadow import RESULT_SCHEMA


SCHEMA_VERSION = "labelagent.semantic-expert-canary-evaluation.v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def evaluate_canary(queue_bytes: bytes, plan_bytes: bytes,
                    result_bytes: bytes) -> dict[str, Any]:
    try:
        queue = json.loads(queue_bytes.decode("utf-8"))
        plan = json.loads(plan_bytes.decode("utf-8"))
        result = json.loads(result_bytes.decode("utf-8"))
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("queue, plan and result must be UTF-8 JSON bytes") from exc

    samples = queue.get("samples") if isinstance(queue, dict) else None
    canary = plan.get("canary") if isinstance(plan, dict) else None
    gate = plan.get("technical_gate") if isinstance(plan, dict) else None
    if (queue.get("schema_version") != QUEUE_SCHEMA
            or not isinstance(samples, list) or len(samples) != 19
            or queue.get("sample_count") != 19
            or queue.get("canary_count") != 8
            or queue.get("remainder_count") != 11
            or queue.get("semantic_canary_plan_sha256") != _sha256(plan_bytes)
            or queue.get("candidate_only") is not True
            or queue.get("human_truth") is not False
            or queue.get("approved_for_training") is not False
            or queue.get("test24_used") is not False):
        raise ValueError("execution queue identity or safety boundary differs")
    if (plan.get("schema_version") != PLAN_SCHEMA
            or not isinstance(canary, list) or len(canary) != 8
            or plan.get("canary_count") != 8
            or plan.get("remainder_count") != 11
            or not isinstance(gate, dict)
            or gate.get("all_requests_passed") is not True
            or gate.get("usage_complete") is not True
            or gate.get("maximum_median_latency_ms") != 60000
            or gate.get("maximum_average_tokens_per_call") != 30000
            or gate.get("automatic_retry_enabled") is not False
            or gate.get("maximum_attempts_per_image") != 1
            or plan.get("candidate_only") is not True
            or plan.get("human_truth") is not False
            or plan.get("approved_for_training") is not False
            or plan.get("test24_used") is not False):
        raise ValueError("canary plan identity or safety boundary differs")

    expected = samples[:8]
    if ([row.get("sample_id") for row in expected]
            != [row.get("sample_id") for row in canary]
            or any(row.get("execution_stage") != "canary" for row in expected)
            or any(row.get("execution_stage") != "remainder"
                   for row in samples[8:])):
        raise ValueError("canary order differs from the execution queue")

    rows = result.get("rows") if isinstance(result, dict) else None
    if (result.get("schema_version") != RESULT_SCHEMA
            or result.get("queue_sha256") != _sha256(queue_bytes)
            or result.get("samples_total") != 19
            or result.get("samples_attempted") != 8
            or not isinstance(rows, list) or len(rows) != 8
            or result.get("automatic_retry_enabled") is not False
            or result.get("candidate_only") is not True
            or result.get("human_truth") is not False
            or result.get("approved_for_training") is not False
            or result.get("automatic_truth_write") is not False
            or result.get("test24_used") is not False):
        raise ValueError("canary result identity or safety boundary differs")

    expected_ids = [row["sample_id"] for row in expected]
    if [row.get("sample_id") for row in rows] != expected_ids:
        raise ValueError("canary result order or membership differs")
    for source, row in zip(expected, rows):
        if (row.get("source_image_sha256") != source.get("source_image_sha256")
                or SHA256.fullmatch(str(row.get("source_image_sha256") or "")) is None
                or row.get("status") != "passed"
                or row.get("external_attempts") != 1
                or row.get("candidate_only") is not True
                or row.get("human_truth") is not False
                or row.get("approved_for_training") is not False
                or row.get("test24_used") is not False):
            raise ValueError("canary row identity or safety boundary differs")

    verdict = canary_guard(
        result,
        expected_attempts=8,
        max_median_latency_ms=gate["maximum_median_latency_ms"],
        max_average_tokens=gate["maximum_average_tokens_per_call"],
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "queue_sha256": _sha256(queue_bytes),
        "canary_plan_sha256": _sha256(plan_bytes),
        "canary_result_sha256": _sha256(result_bytes),
        "canary_count": 8,
        "remainder_count": 11,
        "technical_gate": verdict,
        "technical_gate_passed": verdict["passed"],
        "remainder_technically_eligible": verdict["passed"],
        "remainder_dispatched": False,
        "additional_external_calls_performed": 0,
        "automatic_continuation_enabled": False,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "test24_used": False,
        "production_deployed": False,
        "note": (
            "A passing technical gate does not dispatch the remaining samples "
            "and does not replace a matching explicit upload authorization."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = evaluate_canary(
            args.queue.read_bytes(), args.plan.read_bytes(),
            args.result.read_bytes())
        encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            if args.output.exists():
                raise ValueError("output already exists")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(args.output)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
