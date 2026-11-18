"""Prepare one explicitly isolated candidate frame for edit-cancel acceptance."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--frame", type=int, required=True)
    args = parser.parse_args()
    state_path = Path(args.state).resolve()
    if state_path.name != "state.json" or "isolated-" not in state_path.parent.name:
        raise ValueError("fixture state must be state.json inside an isolated-* directory")
    production_state = Path("D:/Apps/LabelAgent/work/state.json").resolve()
    if state_path == production_state:
        raise ValueError("production state is protected")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    key = str(args.frame)
    candidates = state.get("candidates", {}).get(key)
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"frame {key} has no candidate list")
    if key in state.get("truth", {}):
        raise ValueError(f"frame {key} already has truth")
    registry = state.get("labels")
    if not isinstance(registry, list) or not registry:
        raise ValueError("label registry is missing")
    draft_labels = []
    for shape in candidates:
        label = shape.get("label")
        if label not in registry:
            raise ValueError(f"candidate label is outside registry: {label!r}")
        if label not in draft_labels:
            draft_labels.append(label)

    prior = copy.deepcopy(state.setdefault("agent_runs", {}).get(key, {}))
    prepared = copy.deepcopy(prior)
    prepared.update({
        "agent_review_status": "pending",
        "agent_review_deferred": True,
        "candidate_class_audit_status": "blocked_on_agent_review",
        "candidate_class_audit_deferred": True,
        "candidate_artifact_sha256": digest(candidates),
        "draft_labels": draft_labels,
        "max_rounds": 6,
        "review_reference_frame": None,
        "review_queue": copy.deepcopy(prior.get("review_queue", [])),
    })
    for name in ("review_reference_signature", "review_description_sha256",
                 "agent_refinement", "candidate_class_audits"):
        prepared.pop(name, None)
    state["agent_runs"][key] = prepared

    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(state_path)
    print(json.dumps({
        "schema_version": "labelagent.refinement-cancel-fixture.v1",
        "state": str(state_path),
        "frame": args.frame,
        "candidate_count": len(candidates),
        "draft_labels": draft_labels,
        "candidate_artifact_sha256": prepared["candidate_artifact_sha256"],
        "candidate_only": True,
        "truth_changed": False,
        "source_run_sha256": digest(prior),
        "prepared_run_sha256": digest(prepared),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
