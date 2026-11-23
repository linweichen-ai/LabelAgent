#!/usr/bin/env python3
"""Read-only post-deploy verification for the formal minimal overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.parse import quote
from urllib.request import Request, urlopen

try:
    from tools.deploy_formal_minimal_overlay import load_plan
except ModuleNotFoundError:
    from deploy_formal_minimal_overlay import load_plan


def get_json(url: str, timeout: float) -> dict:
    with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected HTTP status {response.status}")
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("endpoint did not return a JSON object")
    return value


def verify(candidate_root: Path, target_root: Path, base_url: str,
           packet_id: str, expected_priority: list[int], timeout: float) -> dict:
    manifest, plan = load_plan(candidate_root, target_root)
    pending = [row["relative"] for row in plan if row["needs_update"]]
    if pending:
        raise RuntimeError(f"formal target is not fully deployed: {', '.join(pending)}")

    base = base_url.rstrip("/")
    status = get_json(f"{base}/api/run/status", timeout)
    if status.get("status") != "idle":
        raise RuntimeError("formal service is not idle after deployment")

    pallet = get_json(f"{base}/api/pallet-candidate-review", timeout)
    if (not isinstance(pallet.get("available"), bool)
            or pallet.get("read_only") is not True
            or pallet.get("production_deployed") is not False
            or pallet.get("human_truth_writes") != 0
            or pallet.get("training_data_writes") != 0):
        raise RuntimeError("optional pallet review status violates read-only boundaries")

    packet = get_json(f"{base}/api/review_packets/{quote(packet_id, safe='')}", timeout)
    rows = packet.get("rows")
    if not isinstance(rows, list) or packet.get("priority_reviews_loaded") != len(expected_priority):
        raise RuntimeError("priority review packet was not loaded completely")
    ranked = []
    for row in rows:
        priority = row.get("review_priority") if isinstance(row, dict) else None
        if priority is None:
            continue
        if (priority.get("assistant_only") is not True
                or priority.get("human_truth") is not False
                or priority.get("approved_for_training") is not False
                or priority.get("auto_submit_allowed") is not False):
            raise RuntimeError("priority row violates candidate-only boundaries")
        ranked.append((priority.get("priority_rank"), row.get("frame")))
    priority_frames = [frame for _, frame in sorted(ranked)]
    if priority_frames != expected_priority:
        raise RuntimeError(f"priority frame mismatch: {priority_frames}")
    summary = packet.get("summary") or {}
    if (packet.get("read_only_packet") is not True
            or packet.get("submits_human_truth") is not False
            or summary.get("human_truth_frames") != 0
            or summary.get("training_approved_frames") != 0):
        raise RuntimeError("review packet violates truth or training isolation")

    return {
        "schema_version": "labelagent.formal-minimal-overlay-postdeploy-verification.v1",
        "passed": True,
        "candidate_schema": manifest["schema_version"],
        "verified_files": [row["relative"] for row in plan],
        "formal_service_status": "idle",
        "pallet_review_available": pallet["available"],
        "review_packet": packet_id,
        "review_frames": len(rows),
        "priority_frames": priority_frames,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "write_requests": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--packet-id", default="slope-round160")
    parser.add_argument("--priority-frames", default="95,140")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    expected = [int(value.strip()) for value in args.priority_frames.split(",") if value.strip()]
    receipt = verify(args.candidate_root.resolve(strict=True), args.target_root.resolve(strict=True),
                     args.base_url, args.packet_id, expected, args.timeout)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
