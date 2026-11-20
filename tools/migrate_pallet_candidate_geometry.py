#!/usr/bin/env python3
"""Safely migrate legacy pallet candidates through the public workbench API.

The command is read-only unless ``--execute`` is supplied.  A dry run emits a
collection binding that must be supplied back for execution.  Every changed
frame is then read back and checked for candidate identity/class preservation,
truth preservation, review invalidation, revision creation, and idempotence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _request_json(base_url: str, path: str, *, query: dict | None = None,
                  body: dict | None = None, timeout: float = 60) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    if query:
        url = f"{url}?{urlencode(query)}"
    data = None if body is None else json.dumps(
        body, ensure_ascii=False).encode("utf-8")
    request = Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET")
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 local URL
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path}: HTTP {error.code}: {detail}") from error


def _frame_items(payload: object, frame: int) -> list[dict]:
    if isinstance(payload, dict):
        payload = payload.get(str(frame), payload.get(frame, []))
    return copy.deepcopy(payload) if isinstance(payload, list) else []


def _frame_run(payload: object, frame: int) -> dict:
    if isinstance(payload, dict) and str(frame) in payload:
        payload = payload[str(frame)]
    return copy.deepcopy(payload) if isinstance(payload, dict) else {}


def _without_geometry(candidates: list[dict]) -> list[dict]:
    output = copy.deepcopy(candidates)
    for candidate in output:
        candidate.pop("points", None)
    return output


def _state_invariants(state: dict) -> dict:
    return {
        key: copy.deepcopy(state.get(key)) for key in (
            "path", "topic", "labels", "auto_labels", "reference_only_labels",
            "reference_frames", "truth", "truth_review_status", "config",
            "config_state_sha256", "label_state_sha256",
        )
    }


def _read_frame(base_url: str, frame: int, scope: dict) -> dict:
    return _request_json(
        base_url, f"/api/frame_state/{frame}", query=scope)


def _frame_binding(frame: int, state: dict) -> dict:
    candidates = _frame_items(state.get("candidates"), frame)
    run = _frame_run(state.get("agent_runs"), frame)
    return {
        "frame": frame,
        "candidate_count": len(candidates),
        "pallet_candidate_count": sum(
            str(item.get("label", "")).casefold() == "pallet"
            for item in candidates),
        "candidate_artifact_sha256": _canonical_sha(candidates),
        "agent_run_sha256": _canonical_sha(run),
    }


def inspect_collection(base_url: str, start_frame: int = 0,
                       end_frame: int | None = None) -> tuple[dict, list[dict]]:
    bootstrap = _request_json(base_url, "/api/bootstrap")
    topic = bootstrap.get("topic")
    path = bootstrap.get("path")
    generation = bootstrap.get("file_activation_generation")
    if not path or not topic or type(generation) is not int:
        raise RuntimeError("workbench has no fully bound active dataset")
    count = int((bootstrap.get("catalog") or {}).get(topic, {}).get("count", 0))
    start_frame = max(0, int(start_frame))
    end_frame = count if end_frame is None else min(count, int(end_frame))
    if start_frame >= end_frame:
        raise ValueError("start-frame must be smaller than end-frame")
    scope = {
        "path": path,
        "topic": topic,
        "file_activation_generation": generation,
    }
    bindings = [
        _frame_binding(frame, _read_frame(base_url, frame, scope))
        for frame in range(start_frame, end_frame)
    ]
    collection = {
        **scope,
        "total_frame_count": count,
        "frame_range": [start_frame, end_frame],
        "frames": bindings,
    }
    return collection, bindings


def migrate_collection(base_url: str, *, start_frame: int = 0,
                       end_frame: int | None = None, execute: bool = False,
                       expected_collection_binding_sha256: str | None = None,
                       verify_idempotence: bool = True) -> dict:
    before_state = _request_json(base_url, "/api/state")
    invariant_before = _state_invariants(before_state)
    collection, bindings = inspect_collection(
        base_url, start_frame=start_frame, end_frame=end_frame)
    collection_sha = _canonical_sha(collection)
    eligible = [row for row in bindings if row["pallet_candidate_count"]]
    report = {
        "schema_version": "labelagent.pallet_candidate_geometry_migration.v1",
        "mode": "execute" if execute else "dry_run",
        "workbench": base_url,
        "scope": {key: collection[key] for key in (
            "path", "topic", "file_activation_generation", "total_frame_count",
            "frame_range")},
        "collection_binding_sha256": collection_sha,
        "frame_count": len(bindings),
        "eligible_frame_count": len(eligible),
        "candidate_count": sum(row["candidate_count"] for row in bindings),
        "pallet_candidate_count": sum(
            row["pallet_candidate_count"] for row in bindings),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "approved_for_deployment": False,
        "results": [],
    }
    if not execute:
        report["gate_passed"] = True
        return report
    if not expected_collection_binding_sha256:
        raise ValueError("--expected-collection-binding-sha256 is required with --execute")
    if expected_collection_binding_sha256 != collection_sha:
        raise RuntimeError(
            "collection binding changed after dry run; refusing migration")

    scope = {key: collection[key] for key in (
        "path", "topic", "file_activation_generation")}
    for original_binding in eligible:
        frame = original_binding["frame"]
        prior = _read_frame(base_url, frame, scope)
        prior_binding = _frame_binding(frame, prior)
        if prior_binding != original_binding:
            raise RuntimeError(f"frame {frame}: binding changed during migration")
        prior_candidates = _frame_items(prior.get("candidates"), frame)
        prior_truth = _frame_items(prior.get("truth"), frame)
        prior_revisions = list((prior.get("candidate_revision_summary") or {}).get(
            str(frame), []))
        response = _request_json(
            base_url, f"/api/candidates/pallet_cleanup/{frame}", body={
                **scope,
                "expected_candidate_artifact_sha256":
                    prior_binding["candidate_artifact_sha256"],
                "expected_agent_run_sha256": prior_binding["agent_run_sha256"],
            })
        if not (response.get("candidate_only") is True
                and response.get("human_truth") is False
                and response.get("approved_for_training") is False
                and response.get("approved_for_deployment") is False):
            raise RuntimeError(f"frame {frame}: server returned unsafe provenance")
        after = _read_frame(base_url, frame, scope)
        after_candidates = _frame_items(after.get("candidates"), frame)
        after_truth = _frame_items(after.get("truth"), frame)
        after_run = _frame_run(after.get("agent_runs"), frame)
        after_revisions = list((after.get("candidate_revision_summary") or {}).get(
            str(frame), []))
        if prior_truth != after_truth:
            raise RuntimeError(f"frame {frame}: human truth changed")
        if len(prior_candidates) != len(after_candidates):
            raise RuntimeError(f"frame {frame}: candidate count changed")
        if _without_geometry(prior_candidates) != _without_geometry(after_candidates):
            raise RuntimeError(f"frame {frame}: candidate identity/class metadata changed")
        changed = bool(response.get("changed"))
        expected_revision_delta = 1 if changed else 0
        if len(after_revisions) - len(prior_revisions) != expected_revision_delta:
            raise RuntimeError(f"frame {frame}: revision ledger delta is invalid")
        if changed:
            if after_run.get("candidate_class_audits") != []:
                raise RuntimeError(f"frame {frame}: stale class audits were not cleared")
            if after_run.get("human_review_required") is not True:
                raise RuntimeError(f"frame {frame}: human review was not re-required")
            if (after_run.get("candidate_only") is not True
                    or after_run.get("human_truth") is not False
                    or after_run.get("approved_for_training") is not False
                    or after_run.get("approved_for_deployment") is not False):
                raise RuntimeError(f"frame {frame}: replacement run provenance is unsafe")
        idempotent = None
        if verify_idempotence:
            after_binding = _frame_binding(frame, after)
            second = _request_json(
                base_url, f"/api/candidates/pallet_cleanup/{frame}", body={
                    **scope,
                    "expected_candidate_artifact_sha256":
                        after_binding["candidate_artifact_sha256"],
                    "expected_agent_run_sha256": after_binding["agent_run_sha256"],
                })
            final = _read_frame(base_url, frame, scope)
            idempotent = (
                second.get("changed") is False
                and _frame_items(final.get("candidates"), frame) == after_candidates
                and len((final.get("candidate_revision_summary") or {}).get(
                    str(frame), [])) == len(after_revisions))
            if not idempotent:
                raise RuntimeError(f"frame {frame}: second cleanup was not idempotent")
        report["results"].append({
            "frame": frame,
            "changed": changed,
            "adjusted_candidate_count": int(
                response.get("adjusted_candidate_count", 0)),
            "changed_pixels": int(response.get("changed_pixels", 0)),
            "candidate_count_preserved": True,
            "candidate_identity_and_class_preserved": True,
            "truth_preserved": True,
            "revision_created": expected_revision_delta == 1,
            "class_audit_invalidated": (not changed
                                        or after_run.get("candidate_class_audits") == []),
            "human_review_required": (not changed
                                      or after_run.get("human_review_required") is True),
            "idempotent": idempotent,
        })

    after_state = _request_json(base_url, "/api/state")
    invariant_after = _state_invariants(after_state)
    invariant_preserved = invariant_before == invariant_after
    if not invariant_preserved:
        raise RuntimeError("collection-level truth/reference/label/config invariants changed")
    report.update({
        "changed_frame_count": sum(row["changed"] for row in report["results"]),
        "adjusted_candidate_count": sum(
            row["adjusted_candidate_count"] for row in report["results"]),
        "changed_pixels": sum(row["changed_pixels"] for row in report["results"]),
        "truth_reference_label_config_invariants_preserved": True,
        "all_candidate_counts_preserved": all(
            row["candidate_count_preserved"] for row in report["results"]),
        "all_candidate_identity_and_class_preserved": all(
            row["candidate_identity_and_class_preserved"]
            for row in report["results"]),
        "all_revisions_recorded": all(
            not row["changed"] or row["revision_created"]
            for row in report["results"]),
        "all_changed_frames_require_review": all(
            not row["changed"] or row["human_review_required"]
            for row in report["results"]),
        "all_idempotent": all(
            row["idempotent"] is not False for row in report["results"]),
        "gate_passed": True,
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-collection-binding-sha256")
    parser.add_argument("--no-idempotence-check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = migrate_collection(
        args.base_url,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        execute=args.execute,
        expected_collection_binding_sha256=args.expected_collection_binding_sha256,
        verify_idempotence=not args.no_idempotence_check,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    summary_keys = (
        "mode", "collection_binding_sha256", "frame_count",
        "eligible_frame_count", "candidate_count", "pallet_candidate_count",
        "changed_frame_count", "adjusted_candidate_count", "changed_pixels",
        "truth_reference_label_config_invariants_preserved",
        "all_candidate_counts_preserved",
        "all_candidate_identity_and_class_preserved", "all_revisions_recorded",
        "all_changed_frames_require_review", "all_idempotent", "gate_passed",
    )
    summary = {key: report.get(key) for key in summary_keys if key in report}
    if args.output:
        summary["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report.get("gate_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
