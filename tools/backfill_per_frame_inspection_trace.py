#!/usr/bin/env python3
"""Atomically bind the latest run trace to its persisted machine-candidate frames."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--state", type=Path)
    source.add_argument("--file-sessions", type=Path)
    parser.add_argument("--last-run", type=Path, required=True)
    parser.add_argument("--expected-state-sha256")
    parser.add_argument("--expected-sessions-sha256")
    parser.add_argument("--expected-run-sha256", required=True)
    args = parser.parse_args()
    if digest(args.last_run) != args.expected_run_sha256.lower():
        raise SystemExit("last_run SHA-256 changed; refusing backfill")
    run = json.loads(args.last_run.read_text(encoding="utf-8"))
    if run.get("status") != "completed" or run.get("failed_frames"):
        raise SystemExit("latest run is not a clean completed batch")
    if args.state:
        if not args.expected_state_sha256 or digest(args.state) != args.expected_state_sha256.lower():
            raise SystemExit("state SHA-256 changed; refusing backfill")
        destination, root = args.state, json.loads(args.state.read_text(encoding="utf-8"))
        state = root
        expected_destination_sha = args.expected_state_sha256.lower()
        if run.get("file") != state.get("video_path") or run.get("topic") != state.get("topic"):
            raise SystemExit("latest run is not bound to the loaded state file/topic")
    else:
        if (not args.expected_sessions_sha256
                or digest(args.file_sessions) != args.expected_sessions_sha256.lower()):
            raise SystemExit("file_sessions SHA-256 changed; refusing backfill")
        destination = args.file_sessions
        root = json.loads(args.file_sessions.read_text(encoding="utf-8"))
        state = root.get(run.get("file"))
        expected_destination_sha = args.expected_sessions_sha256.lower()
        if not isinstance(state, dict) or run.get("topic") != state.get("topic"):
            raise SystemExit("latest run is not bound to a persisted file session/topic")
    truth_before = json_digest(state.get("truth", {}))
    candidates_before = json_digest(state.get("candidates", {}))
    grouped = {}
    for item in run.get("trace", []):
        frame = item.get("frame")
        if (type(frame) is int and item.get("file") == run["file"]
                and item.get("topic") == run["topic"]
                and item.get("purpose") != "生成真值描述"):
            grouped.setdefault(str(frame), []).append(item)
    updated = []
    for row in run.get("frames", []):
        key = str(row.get("frame"))
        calls = grouped.get(key, [])
        agent = state.get("agent_runs", {}).get(key)
        if (not calls or not isinstance(agent, dict)
                or agent.get("agent_review_status") != "complete"
                or key in state.get("truth", {})):
            continue
        agent["inspection_trace"] = calls
        updated.append(int(key))
    if not updated:
        raise SystemExit("no eligible completed machine frame trace found")
    if json_digest(state.get("truth", {})) != truth_before:
        raise SystemExit("truth changed during backfill")
    if json_digest(state.get("candidates", {})) != candidates_before:
        raise SystemExit("candidates changed during backfill")
    temporary = destination.with_suffix(".trace-backfill.tmp")
    temporary.write_text(json.dumps(
        root, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8", newline="\n")
    if digest(destination) != expected_destination_sha:
        temporary.unlink(missing_ok=True)
        raise SystemExit("destination changed before commit; refusing backfill")
    temporary.replace(destination)
    print(json.dumps({
        "schema_version": "labelagent.inspection-trace-backfill.v1",
        "updated_frames": updated,
        "updated_count": len(updated),
        "truth_sha256": truth_before,
        "candidates_sha256": candidates_before,
        "destination": str(destination),
        "destination_sha256_after": digest(destination),
        "candidate_only": True,
        "human_truth_changed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
