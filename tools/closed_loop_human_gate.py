#!/usr/bin/env python3
"""Import an explicit workbench human decision into the closed-loop ledger."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.agent.closed_loop import ClosedLoopLedger, digest


ACCEPTED = {"accept_candidate", "correct_candidate", "manual_annotation"}


def read_review_chain(workbench_root):
    folder = Path(workbench_root) / "human_reviews"
    records = []
    previous = None
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        review_id = record.pop("review_id", None)
        if record.get("previous_review_id") != previous or digest(record) != review_id:
            raise RuntimeError(f"human review hash chain is invalid at {path.name}")
        record["review_id"] = review_id
        records.append(record)
        previous = review_id
    return records


def truth_for(state, sensor, frame_id):
    if state.get("topic") == sensor:
        truth = state.get("truth", {})
    else:
        truth = state.get("truths_by_topic", {}).get(sensor, {})
    return truth.get(str(frame_id))


def sync_human_gate(*, bundle_path, workbench_root, ledger_root,
                    output_path, review_id=None):
    bundle_path, workbench_root = Path(bundle_path), Path(workbench_root)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    envelope = bundle["input"]
    ledger = ClosedLoopLedger(ledger_root, envelope)
    ledger.register()
    frame_id, sensor = str(envelope["frame_id"]), envelope["sensor"]
    matching = [item for item in read_review_chain(workbench_root)
                if str(item.get("frame")) == frame_id
                and item.get("topic") == sensor
                and item.get("frame_sha256") == envelope["content_sha256"]]
    if review_id is not None:
        matching = [item for item in matching if item["review_id"] == review_id]
    if not matching:
        raise ValueError("no matching human review receipt for this immutable input")
    review = matching[-1]
    decision = review["decision"]
    common = {"actor": {"type": "human", "id": review["reviewer_id"]},
              "action": decision, "label_layer": "human_truth",
              "confidence": None,
              "payload": {"review_id": review["review_id"],
                          "reviewed_at": review["reviewed_at"],
                          "note": review.get("note", ""),
                          "candidate_sha256": review.get("candidate_sha256"),
                          "truth_sha256": review.get("truth_sha256"),
                          "truth_shape_count": review.get("truth_shape_count")},
              "artifacts": [{"kind": "human_review_receipt",
                             "sha256": review["review_id"]}],
              "metrics": {"human_seconds_per_frame": review["elapsed_seconds"],
                          "human_minutes_per_frame": round(review["elapsed_seconds"] / 60, 6)}}
    if decision == "reject_candidate":
        ledger.append_event("human_rejected", **common)
        truth = None
    elif decision in ACCEPTED:
        state = json.loads((workbench_root / "state.json").read_text(encoding="utf-8"))
        truth = truth_for(state, sensor, frame_id)
        if truth is None or digest(truth) != review.get("truth_sha256"):
            raise ValueError("current workbench truth does not match the accepted review receipt")
        ledger.append_event("human_accepted", **common)
        ledger.append_event("truth_committed", actor={"type": "system", "id": "truth-writer"},
            action="commit_reviewed_truth", label_layer="human_truth",
            payload={"review_id": review["review_id"],
                     "truth_sha256": review["truth_sha256"],
                     "truth_shape_count": len(truth)},
            artifacts=[{"kind": "truth", "sha256": review["truth_sha256"]}])
    else:
        raise ValueError(f"unsupported human decision: {decision}")
    receipt = ledger.receipt()
    output = {"schema_version": bundle["schema_version"], "input": envelope,
              "events": ledger.events(), "receipt": receipt,
              "human_review": review,
              "workbench_readback": {
                  "truth_shape_count": len(truth) if truth is not None else 0,
                  "is_reference_frame": False,
                  "candidate_shape_count": review.get("candidate_shape_count", 0)
                  if decision == "reject_candidate" else 0}}
    output["bundle_sha256"] = digest(output)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2,
                                      sort_keys=True) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--workbench-root", required=True)
    parser.add_argument("--ledger-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--review-id")
    args = parser.parse_args()
    result = sync_human_gate(bundle_path=args.bundle,
        workbench_root=args.workbench_root, ledger_root=args.ledger_root,
        output_path=args.output, review_id=args.review_id)
    print(json.dumps({"input_id": result["receipt"]["input_id"],
                      "last_state": result["receipt"]["last_state"],
                      "human_gate_passed": result["receipt"]["human_gate_passed"],
                      "minimum_closed_loop_complete": result["receipt"]["minimum_closed_loop_complete"],
                      "bundle_sha256": result["bundle_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
