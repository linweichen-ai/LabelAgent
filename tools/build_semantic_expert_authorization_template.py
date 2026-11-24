#!/usr/bin/env python3
"""Build a fail-closed authorization template for a semantic expert queue."""

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
from tools.run_mixed_hard_external_shadow import AUTH_SCHEMA


SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_authorization_template(queue_bytes: bytes) -> dict[str, Any]:
    if not isinstance(queue_bytes, bytes):
        raise ValueError("execution queue bytes are required")
    queue = json.loads(queue_bytes.decode("utf-8"))
    samples = queue.get("samples") if isinstance(queue, dict) else None
    if (queue.get("schema_version") != QUEUE_SCHEMA
            or not isinstance(samples, list) or not samples
            or queue.get("sample_count") != len(samples)
            or queue.get("canary_count") != 8
            or queue.get("remainder_count") != len(samples) - 8
            or queue.get("maximum_external_calls_if_authorized") != len(samples)
            or queue.get("maximum_canary_attempts_if_authorized") != 8
            or queue.get("external_upload_authorized") is not False
            or queue.get("ready_for_shadow_execution") is not False
            or queue.get("images_copied") is not False
            or queue.get("image_paths_included") is not False
            or queue.get("candidate_only") is not True
            or queue.get("human_truth") is not False
            or queue.get("approved_for_training") is not False
            or queue.get("test24_used") is not False):
        raise ValueError("execution queue identity or boundary differs")
    seen = set()
    for row in samples:
        sample_id = row.get("sample_id") if isinstance(row, dict) else None
        if (not isinstance(sample_id, str) or not sample_id or sample_id in seen
                or SHA256.fullmatch(str(row.get("source_image_sha256") or "")) is None
                or row.get("external_upload_authorized") is not False
                or any("path" in str(key).casefold() for key in row)):
            raise ValueError("execution queue sample boundary differs")
        seen.add(sample_id)
    return {
        "schema_version": AUTH_SCHEMA,
        "scope": "mixed_hard_external_shadow",
        "queue_sha256": _sha256(queue_bytes),
        "authorized": False,
        "authorized_by": None,
        "authorized_at": None,
        "authorized_sample_count": len(samples),
        "authorized_upload_count": 0,
        "max_external_calls": len(samples),
        "first_stage_max_new_attempts": 8,
        "automatic_retry_enabled": False,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "test24_used": False,
        "production_deployed": False,
        "instructions": (
            "Do not set authorized=true without a new explicit user authorization "
            "that names this hash-bound 19-image development queue."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = build_authorization_template(args.queue.read_bytes())
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
