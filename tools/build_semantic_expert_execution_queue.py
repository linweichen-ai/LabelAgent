#!/usr/bin/env python3
"""Convert an unauthorized semantic canary plan into a path-free run queue."""

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

from tools.build_mixed_hard_expert_queue import SCHEMA_VERSION
from tools.plan_semantic_expert_canary import SCHEMA_VERSION as PLAN_SCHEMA


SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_execution_queue(plan_bytes: bytes) -> dict[str, Any]:
    if not isinstance(plan_bytes, bytes):
        raise ValueError("semantic canary plan bytes are required")
    plan = json.loads(plan_bytes.decode("utf-8"))
    canary, remainder = plan.get("canary"), plan.get("remainder")
    if (plan.get("schema_version") != PLAN_SCHEMA
            or plan.get("status") != "authorization_required"
            or not isinstance(canary, list) or not isinstance(remainder, list)
            or plan.get("canary_count") != len(canary)
            or plan.get("remainder_count") != len(remainder)
            or plan.get("external_candidate_count") != len(canary) + len(remainder)
            or plan.get("external_authorization_granted") is not False
            or plan.get("dispatch_performed") is not False
            or plan.get("external_upload_performed") is not False
            or plan.get("factory_data_uploaded") is not False
            or plan.get("candidate_only") is not True
            or plan.get("human_truth") is not False
            or plan.get("approved_for_training") is not False
            or plan.get("test24_used") is not False
            or plan.get("production_deployed") is not False):
        raise ValueError("semantic canary plan identity or boundary differs")
    ordered = canary + remainder
    samples, seen = [], set()
    for index, row in enumerate(ordered):
        if not isinstance(row, dict):
            raise ValueError("semantic canary row is invalid")
        sample_id = row.get("sample_id")
        original_sha = str(row.get("source_image_sha256") or "")
        upload_source_sha = str(row.get("copied_image_sha256") or "")
        if (not isinstance(sample_id, str) or not sample_id or sample_id in seen
                or not isinstance(row.get("cohort"), str) or not row["cohort"]
                or SHA256.fullmatch(original_sha) is None
                or SHA256.fullmatch(upload_source_sha) is None
                or row.get("external_upload_authorized") is not False
                or row.get("candidate_only") is not True
                or any("path" in str(key).casefold() for key in row)):
            raise ValueError("semantic canary row identity or boundary differs")
        seen.add(sample_id)
        samples.append({
            "sample_id": sample_id,
            "cohort": row["cohort"],
            # The existing runner compares this field to the actual bytes read
            # from the review workspace.  Bind it to the copied JPEG, while
            # retaining the acquisition hash separately for provenance.
            "source_image_sha256": upload_source_sha,
            "original_source_image_sha256": original_sha,
            "route_source": row.get("route_source"),
            "execution_stage": "canary" if index < len(canary) else "remainder",
            "external_upload_authorized": False,
            "external_dispatch_performed": False,
            "human_reference_available": False,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "semantic_canary_plan_sha256": _sha256(plan_bytes),
        "sample_count": len(samples),
        "source_sample_count": len(samples),
        "canary_count": len(canary),
        "remainder_count": len(remainder),
        "maximum_external_calls_if_authorized": len(samples),
        "maximum_canary_attempts_if_authorized": len(canary),
        "external_call_ratio": None,
        "external_upload_authorized": False,
        "ready_for_shadow_execution": False,
        "ready_for_quality_ab": False,
        "images_copied": False,
        "image_paths_included": False,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "test24_used": False,
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = build_execution_queue(args.plan.read_bytes())
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
