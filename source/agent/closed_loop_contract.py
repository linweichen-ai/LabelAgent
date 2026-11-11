"""Dependency-free validation for the versioned LabelAgent event protocol.

This validates transport events before they are appended to the lower-level
hash-chained ledger in :mod:`source.agent.closed_loop`. It deliberately uses a
strict current-runtime profile: truth commits require explicit human approval.
"""
from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Mapping


SCHEMA_VERSION = "1.1.0"
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
EVENT_TYPES = {
    "frame.ingested", "quality.checked", "candidate.generated",
    "vlm.reviewed", "agent.action", "unknown.proposed",
    "ontology.reviewed", "human.reviewed", "truth.committed",
    "dataset.versioned", "evaluation.completed", "training.completed",
    "deployment.promoted", "deployment.rolled_back",
}
REQUIRED_TOP_LEVEL = {
    "schema_version", "event_id", "event_type", "occurred_at", "run_id",
    "correlation_id", "idempotency_key", "workflow", "source",
    "producer", "decision", "artifacts",
}
OPTIONAL_TOP_LEVEL = {
    "parent_event_id", "task_context", "slope_context", "unknown_candidate",
    "review", "metrics",
}
ALLOWED_TRANSITIONS = {
    None: {"ingested"},
    "ingested": {"qc_passed", "quarantined"},
    "qc_passed": {"candidates_generated"},
    "quarantined": set(),
    "candidates_generated": {"vlm_reviewed"},
    "vlm_reviewed": {"agent_iteration", "auto_accepted", "review_required",
                     "ontology_review_required"},
    "agent_iteration": {"vlm_reviewed", "review_required"},
    "auto_accepted": {"truth_committed"},
    "review_required": {"human_approved", "human_rejected"},
    "ontology_review_required": {"ontology_versioned", "human_rejected"},
    "ontology_versioned": {"reprocess_required"},
    "reprocess_required": {"candidates_generated"},
    "human_approved": {"truth_committed"},
    "human_rejected": set(),
    "truth_committed": {"dataset_versioned"},
    "dataset_versioned": {"evaluated"},
    "evaluated": {"trained", "deployment_candidate"},
    "trained": {"evaluated"},
    "deployment_candidate": {"promoted"},
    "promoted": {"rolled_back"},
    "rolled_back": set(),
}
EVENT_STATES = {
    "frame.ingested": {"ingested"},
    "quality.checked": {"qc_passed", "quarantined"},
    "candidate.generated": {"candidates_generated"},
    "vlm.reviewed": {"vlm_reviewed"},
    "agent.action": {"agent_iteration"},
    "unknown.proposed": {"ontology_review_required"},
    "ontology.reviewed": {"ontology_versioned", "human_rejected"},
    "human.reviewed": {"human_approved", "human_rejected"},
    "truth.committed": {"truth_committed"},
    "dataset.versioned": {"dataset_versioned"},
    "evaluation.completed": {"evaluated"},
    "training.completed": {"trained"},
    "deployment.promoted": {"promoted"},
    "deployment.rolled_back": {"rolled_back"},
}
PRODUCERS = {"ingestor", "quality_gate", "sam", "vlm", "agent", "human",
             "trainer", "deployer"}
DECISIONS = {"accepted", "rejected", "review_required", "unknown",
             "quarantined", "committed", "versioned", "promoted", "rolled_back"}


def _datetime_with_timezone(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _nonempty(value: object, minimum=1) -> bool:
    return isinstance(value, str) and len(value.strip()) >= minimum


def validate_event(event: Mapping[str, Any]) -> list[str]:
    """Return deterministic safety errors without mutating *event*."""
    errors: list[str] = []
    if not isinstance(event, Mapping):
        return ["event must be an object"]
    missing = sorted(REQUIRED_TOP_LEVEL - set(event))
    if missing:
        errors.append(f"missing top-level fields: {', '.join(missing)}")
    unknown = sorted(set(event) - REQUIRED_TOP_LEVEL - OPTIONAL_TOP_LEVEL)
    if unknown:
        errors.append(f"unknown top-level fields: {', '.join(unknown)}")
    if event.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    event_type = event.get("event_type")
    if event_type not in EVENT_TYPES:
        errors.append("event_type is not supported")
    for name, minimum in (("event_id", 8), ("run_id", 8),
                          ("correlation_id", 8), ("idempotency_key", 16)):
        if not _nonempty(event.get(name), minimum):
            errors.append(f"{name} must contain at least {minimum} characters")
    if not _datetime_with_timezone(event.get("occurred_at")):
        errors.append("occurred_at must be an ISO-8601 datetime with timezone")

    workflow = event.get("workflow")
    if not isinstance(workflow, Mapping):
        errors.append("workflow must be an object")
    else:
        previous, state = workflow.get("previous_state"), workflow.get("state")
        if previous not in ALLOWED_TRANSITIONS:
            errors.append("workflow.previous_state is not supported")
        elif state not in ALLOWED_TRANSITIONS[previous]:
            errors.append(f"invalid workflow transition: {previous} -> {state}")
        if event_type in EVENT_STATES and state not in EVENT_STATES[event_type]:
            errors.append(f"{event_type} is incompatible with workflow.state={state}")
        iteration, maximum = workflow.get("iteration"), workflow.get("max_iterations")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            errors.append("workflow.iteration must be a non-negative integer")
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            errors.append("workflow.max_iterations must be a positive integer")
        if (isinstance(iteration, int) and not isinstance(iteration, bool)
                and isinstance(maximum, int) and not isinstance(maximum, bool)
                and iteration > maximum):
            errors.append("workflow.iteration cannot exceed max_iterations")

    source = event.get("source")
    if not isinstance(source, Mapping):
        errors.append("source must be an object")
    else:
        for name in ("dataset_id", "source_batch", "frame_id", "capture_time",
                     "origin", "modality", "content_sha256"):
            if name not in source:
                errors.append(f"source.{name} is required")
        if not _datetime_with_timezone(source.get("capture_time")):
            errors.append("source.capture_time must be an ISO-8601 datetime with timezone")
        origin = source.get("origin")
        if origin not in {"real", "simulation"}:
            errors.append("source.origin must be real or simulation")
        simulation = source.get("simulation")
        if origin == "simulation":
            if not isinstance(simulation, Mapping):
                errors.append("simulation sources require source.simulation metadata")
            else:
                for name in ("simulator", "scene_config", "random_seed", "sensor_version"):
                    if name not in simulation:
                        errors.append(f"source.simulation.{name} is required")
        if not SHA256_RE.fullmatch(str(source.get("content_sha256", ""))):
            errors.append("source.content_sha256 must be lowercase SHA-256")

    producer = event.get("producer")
    if not isinstance(producer, Mapping):
        errors.append("producer must be an object")
    else:
        if producer.get("component") not in PRODUCERS:
            errors.append("producer.component is not supported")
        if not _nonempty(producer.get("code_commit"), 7):
            errors.append("producer.code_commit must contain at least 7 characters")

    decision = event.get("decision")
    if not isinstance(decision, Mapping):
        errors.append("decision must be an object")
    else:
        if decision.get("status") not in DECISIONS:
            errors.append("decision.status is not supported")
        confidence = decision.get("confidence")
        if confidence is not None and (not _number(confidence) or not 0 <= confidence <= 1):
            errors.append("decision.confidence must be null or between 0 and 1")
        reasons = decision.get("reason_codes")
        if not isinstance(reasons, list) or len(reasons) != len(set(map(str, reasons))):
            errors.append("decision.reason_codes must be a unique array")

    artifacts = event.get("artifacts")
    artifact_ids: set[str] = set()
    if not isinstance(artifacts, list):
        errors.append("artifacts must be an array")
    else:
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, Mapping):
                errors.append(f"artifacts[{index}] must be an object")
                continue
            artifact_id = str(artifact.get("artifact_id", ""))
            if not artifact_id:
                errors.append(f"artifacts[{index}].artifact_id is required")
            elif artifact_id in artifact_ids:
                errors.append(f"duplicate artifact_id: {artifact_id}")
            artifact_ids.add(artifact_id)
            if not SHA256_RE.fullmatch(str(artifact.get("sha256", ""))):
                errors.append(f"artifacts[{index}].sha256 must be lowercase SHA-256")

    tags: set[str] = set()
    task = event.get("task_context")
    if task is not None:
        if not isinstance(task, Mapping) or not isinstance(task.get("hard_case_tags"), list):
            errors.append("task_context.hard_case_tags must be an array")
        else:
            tags = set(map(str, task["hard_case_tags"]))
    if "slope" in tags:
        slope = event.get("slope_context")
        if not isinstance(slope, Mapping):
            errors.append("slope hard cases require slope_context")
        else:
            if slope.get("semantic_label") != "slope_visible":
                errors.append("slope_context.semantic_label must be slope_visible")
            traversability = slope.get("traversability_status")
            evidence = slope.get("geometry_evidence_artifact_ids")
            if traversability in {"traversable", "non_traversable"}:
                if slope.get("geometry_evidence_sufficient") is not True:
                    errors.append("slope traversability decisions require sufficient geometry evidence")
                if not isinstance(evidence, list) or not evidence:
                    errors.append("slope traversability decisions require geometry evidence artifacts")
            if isinstance(evidence, list):
                missing_ids = sorted(set(map(str, evidence)) - artifact_ids)
                if missing_ids:
                    errors.append("slope geometry evidence references missing artifacts: "
                                  + ", ".join(missing_ids))
            if (event_type == "evaluation.completed"
                    and slope.get("human_pixel_truth_available") is not True):
                errors.append("slope evaluation requires human pixel truth")
            if slope.get("approved_for_training") is True:
                if slope.get("human_pixel_truth_available") is not True:
                    errors.append("slope cannot be approved for training without human pixel truth")
                if event_type in {"candidate.generated", "vlm.reviewed", "agent.action",
                                  "unknown.proposed"}:
                    errors.append("machine slope candidate events cannot be approved for training")

    review = event.get("review")
    if event_type == "unknown.proposed":
        unknown_candidate = event.get("unknown_candidate")
        if not isinstance(unknown_candidate, Mapping):
            errors.append("unknown.proposed requires unknown_candidate")
        else:
            if unknown_candidate.get("ontology_mutation_allowed") is not False:
                errors.append("unknown candidates cannot mutate the production ontology")
            evidence = unknown_candidate.get("evidence_artifact_ids")
            if not isinstance(evidence, list) or not evidence:
                errors.append("unknown candidates require evidence artifacts")
            else:
                missing_ids = sorted(set(map(str, evidence)) - artifact_ids)
                if missing_ids:
                    errors.append("unknown evidence references missing artifacts: "
                                  + ", ".join(missing_ids))
        if not isinstance(review, Mapping):
            errors.append("unknown.proposed requires review")
        else:
            if review.get("required") is not True:
                errors.append("unknown candidates require review.required=true")
            if review.get("status") != "pending":
                errors.append("new unknown candidates must start with pending review")
        if isinstance(decision, Mapping) and decision.get("status") != "review_required":
            errors.append("unknown.proposed decision must require review")

    if event_type == "truth.committed":
        previous = workflow.get("previous_state") if isinstance(workflow, Mapping) else None
        if previous != "human_approved":
            errors.append("current runtime truth commits require previous_state=human_approved")
        if not isinstance(review, Mapping):
            errors.append("current runtime truth commits require an approved human review")
        elif not (review.get("required") is True and review.get("status") == "approved"
                  and review.get("reviewer_type") == "human"
                  and _datetime_with_timezone(review.get("reviewed_at"))):
            errors.append("current runtime truth commits require an approved, timestamped human review")
    return errors


def assert_valid_event(event: Mapping[str, Any]) -> None:
    errors = validate_event(event)
    if errors:
        raise ValueError("; ".join(errors))
