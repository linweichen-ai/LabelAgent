"""Offline contracts for VLM label discovery and human ontology promotion.

There is no model, GPU, network, OSS, CVAT, or process-launch code here.  VLM
observations and deterministic clustering can only create proposals.  A new
append-only registry version requires an authenticated human decision, complete
approval evidence, and a fixed regression-set result with real/sim accounting.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re

from .factory_data_contract import digest, validate_label_registry


HASH = re.compile(r"^[0-9a-f]{64}$")
OUTCOMES = {"known_label", "unknown_candidate", "uncertain_conflict"}
SAM_RELATIONS = {"consistent", "conflict", "insufficient_evidence"}
EVENT_TYPES = {
    "vlm.label_observation.recorded",
    "ontology.unknown_cluster.proposed",
    "ontology.registry.version.promoted",
    "ontology.promotion.rejected",
}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes)
                          else canonical_json(value)).hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 4000:
        raise ValueError(f"{name} must be non-empty text")
    return value


def _sha(value, name):
    if not isinstance(value, str) or not HASH.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _time(value, name):
    _text(value, name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include timezone")


def _artifact(value, name):
    if not isinstance(value, dict) or set(value) != {"artifact_id", "uri", "sha256"}:
        raise ValueError(f"{name} artifact is incomplete")
    _sha(value["artifact_id"], f"{name}.artifact_id")
    _text(value["uri"], f"{name}.uri")
    _sha(value["sha256"], f"{name}.sha256")


def _validate_provenance(value):
    if not isinstance(value, dict) or set(value) != {"source_kind", "simulation"}:
        raise ValueError("provenance fields are incomplete")
    if value["source_kind"] == "real":
        if value["simulation"] is not None:
            raise ValueError("real provenance cannot include simulation metadata")
    elif value["source_kind"] == "simulation":
        simulation = value["simulation"]
        if not isinstance(simulation, dict) or set(simulation) != {
            "simulator", "scene", "seed", "version"
        }:
            raise ValueError("simulation provenance is incomplete")
        for field in simulation:
            _text(str(simulation[field]), field)
    else:
        raise ValueError("source_kind must be real or simulation")


def build_label_observation(payload, registry, *, observed_at,
                            confidence_threshold=0.8):
    """Record a supplied structured VLM observation; never call a model."""
    validate_label_registry(registry)
    result = payload["result"]
    outcome = result.get("outcome")
    if outcome not in OUTCOMES:
        raise ValueError("VLM label outcome is invalid")
    confidence = result.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("VLM confidence must be in [0,1]")
    sam_relation = result.get("sam_relation")
    if sam_relation not in SAM_RELATIONS:
        raise ValueError("SAM relation is invalid")
    known_names = {item["name"] for item in registry["labels"]}
    proposed = result.get("proposed_name")
    if outcome == "known_label" and proposed not in known_names:
        raise ValueError("known_label must name a registered label")
    review_required = (
        outcome != "known_label" or confidence < confidence_threshold
        or sam_relation == "conflict"
    )
    body = {
        "schema_version": "labelagent.vlm-label-observation.v1",
        "event_type": "vlm.label_observation.recorded",
        "observed_at": observed_at,
        "identity": payload["identity"],
        "provenance": payload["provenance"],
        "inputs": payload["inputs"],
        "registry_binding": {
            "registry_version": registry["registry_version"],
            "registry_sha256": registry["registry_content_sha256"],
        },
        "result": result,
        "routing": {
            "confidence_threshold": confidence_threshold,
            "human_review_required": review_required,
            "reason_codes": sorted([
                *( ["UNKNOWN_OR_NEW_LABEL"] if outcome == "unknown_candidate" else []),
                *( ["UNCERTAIN_OR_CONFLICT"] if outcome == "uncertain_conflict" else []),
                *( ["LOW_CONFIDENCE"] if confidence < confidence_threshold else []),
                *( ["SAM_ONTOLOGY_CONFLICT"] if sam_relation == "conflict" else []),
            ]),
            "next_state": "human_review_required" if review_required else "known_label_candidate",
        },
        "ontology_mutation_allowed": False,
        "truth_promotion_allowed": False,
        "write_policy": {"mode": "append_only", "overwrite_registry": False},
    }
    body["idempotency_key"] = digest({
        "input_id": body["identity"]["input_id"],
        "candidate_sha256": body["inputs"]["candidate"]["sha256"],
        "model_sha256": body["inputs"]["model"]["sha256"],
        "prompt_sha256": body["inputs"]["prompt"]["sha256"],
        "registry_sha256": body["registry_binding"]["registry_sha256"],
    })
    body["event_id"] = sha(body)
    validate_label_observation(body)
    return body


def validate_label_observation(event):
    required = {
        "schema_version", "event_type", "observed_at", "identity",
        "provenance", "inputs", "registry_binding", "result", "routing",
        "ontology_mutation_allowed", "truth_promotion_allowed",
        "write_policy", "idempotency_key", "event_id",
    }
    if not isinstance(event, dict) or set(event) != required:
        raise ValueError("label observation fields are incomplete or unknown")
    if event["schema_version"] != "labelagent.vlm-label-observation.v1" or event["event_type"] != "vlm.label_observation.recorded":
        raise ValueError("invalid label observation")
    _time(event["observed_at"], "observed_at")
    identity = event["identity"]
    if set(identity) != {"dataset_id", "source_batch", "sequence", "frame_id", "input_id"}:
        raise ValueError("label observation identity is incomplete")
    for field in ("dataset_id", "source_batch", "sequence", "frame_id"):
        _text(identity[field], field)
    _sha(identity["input_id"], "input_id")
    _validate_provenance(event["provenance"])
    inputs = event["inputs"]
    if set(inputs) != {"views", "candidate", "model", "prompt"}:
        raise ValueError("label observation inputs are incomplete")
    views = inputs["views"]
    if not isinstance(views, list) or len(views) < 2:
        raise ValueError("multi-view observation requires at least two views")
    cameras = set()
    for index, view in enumerate(views):
        if set(view) != {"camera", "artifact"}:
            raise ValueError("view fields are incomplete")
        _text(view["camera"], "camera")
        if view["camera"] in cameras:
            raise ValueError("multi-view cameras must be unique")
        cameras.add(view["camera"])
        _artifact(view["artifact"], f"views[{index}]")
    _artifact(inputs["candidate"], "candidate")
    for name in ("model", "prompt"):
        value = inputs[name]
        if set(value) != {"id", "version", "sha256"}:
            raise ValueError(f"{name} binding is incomplete")
        _text(value["id"], f"{name}.id")
        _text(value["version"], f"{name}.version")
        _sha(value["sha256"], f"{name}.sha256")
    binding = event["registry_binding"]
    if set(binding) != {"registry_version", "registry_sha256"}:
        raise ValueError("registry binding is incomplete")
    _text(binding["registry_version"], "registry_version")
    _sha(binding["registry_sha256"], "registry_sha256")
    result = event["result"]
    if set(result) != {
        "outcome", "proposed_name", "aliases", "nearest_known_labels",
        "evidence", "confidence", "multiview_consistency", "sam_relation",
    }:
        raise ValueError("structured VLM result is incomplete")
    if result["outcome"] not in OUTCOMES:
        raise ValueError("structured VLM outcome is invalid")
    _text(result["proposed_name"], "proposed_name")
    if not isinstance(result["aliases"], list) or any(not isinstance(x, str) or not x for x in result["aliases"]):
        raise ValueError("candidate aliases are invalid")
    if not isinstance(result["nearest_known_labels"], list):
        raise ValueError("nearest known labels must be a list")
    for nearest in result["nearest_known_labels"]:
        if set(nearest) != {"label", "similarity", "difference"}:
            raise ValueError("nearest known label evidence is incomplete")
        _text(nearest["label"], "nearest label")
        _text(nearest["difference"], "nearest difference")
        if isinstance(nearest["similarity"], bool) or not isinstance(nearest["similarity"], (int, float)) or not 0 <= nearest["similarity"] <= 1:
            raise ValueError("nearest label similarity is invalid")
    evidence = result["evidence"]
    if set(evidence) != {"summary", "visual_cues", "view_artifact_ids"}:
        raise ValueError("VLM evidence is incomplete")
    _text(evidence["summary"], "evidence summary")
    if not evidence["visual_cues"] or not evidence["view_artifact_ids"]:
        raise ValueError("VLM evidence must cite cues and views")
    known_view_ids = {view["artifact"]["artifact_id"] for view in views}
    if not set(evidence["view_artifact_ids"]) <= known_view_ids:
        raise ValueError("VLM evidence cites unknown views")
    consistency = result["multiview_consistency"]
    if set(consistency) != {"status", "agreement", "supporting_cameras", "conflicting_cameras"}:
        raise ValueError("multi-view consistency fields are incomplete")
    if consistency["status"] not in {"consistent", "inconsistent", "insufficient"}:
        raise ValueError("multi-view consistency status is invalid")
    if isinstance(consistency["agreement"], bool) or not isinstance(consistency["agreement"], (int, float)) or not 0 <= consistency["agreement"] <= 1:
        raise ValueError("multi-view agreement is invalid")
    if not set(consistency["supporting_cameras"] + consistency["conflicting_cameras"]) <= cameras:
        raise ValueError("multi-view consistency cites unknown cameras")
    confidence = result["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("VLM confidence is invalid")
    if result["sam_relation"] not in SAM_RELATIONS:
        raise ValueError("SAM relation is invalid")
    routing = event["routing"]
    if set(routing) != {"confidence_threshold", "human_review_required", "reason_codes", "next_state"}:
        raise ValueError("label observation routing is incomplete")
    expected_review = result["outcome"] != "known_label" or confidence < routing["confidence_threshold"] or result["sam_relation"] == "conflict"
    if routing["human_review_required"] is not expected_review:
        raise ValueError("unsafe label observation routing")
    if expected_review and routing["next_state"] != "human_review_required":
        raise ValueError("unknown/low-confidence/conflict must enter human review")
    if event["ontology_mutation_allowed"] is not False or event["truth_promotion_allowed"] is not False:
        raise ValueError("VLM observations cannot mutate ontology or truth")
    if event["write_policy"] != {"mode": "append_only", "overwrite_registry": False}:
        raise ValueError("label observation write policy is unsafe")
    for field in ("idempotency_key", "event_id"):
        _sha(event[field], field)
    if event["event_id"] != sha({key: value for key, value in event.items() if key != "event_id"}):
        raise ValueError("label observation event hash mismatch")


def build_cluster_proposal(observations, *, cluster_id, config,
                           proposed_at):
    if not observations:
        raise ValueError("unknown cluster requires observations")
    for event in observations:
        validate_label_observation(event)
        if not event["routing"]["human_review_required"]:
            raise ValueError("auto-acceptable known labels cannot enter unknown clustering")
    source_kinds = sorted({event["provenance"]["source_kind"] for event in observations})
    ranked = sorted(observations, key=lambda event: (
        -len(event["result"]["evidence"]["view_artifact_ids"]),
        -event["result"]["confidence"], event["event_id"]))
    representative = ranked[0]
    aliases = sorted({alias for event in observations
                      for alias in [event["result"]["proposed_name"], *event["result"]["aliases"]]})
    body = {
        "schema_version": "labelagent.ontology-cluster-proposal.v1",
        "event_type": "ontology.unknown_cluster.proposed",
        "proposed_at": proposed_at,
        "cluster_id": cluster_id,
        "member_event_ids": sorted(event["event_id"] for event in observations),
        "member_event_sha256": sorted(sha(event) for event in observations),
        "representative": {
            "event_id": representative["event_id"],
            "candidate_sha256": representative["inputs"]["candidate"]["sha256"],
            "selection_policy": "max_view_evidence_then_confidence_then_event_id",
        },
        "candidate_names_and_aliases": aliases,
        "source_kinds": source_kinds,
        "source_partition_event_ids": {
            kind: sorted(event["event_id"] for event in observations
                         if event["provenance"]["source_kind"] == kind)
            for kind in source_kinds
        },
        "clustering": config,
        "status": "proposal_only",
        "human_review_required": True,
        "ontology_mutation_allowed": False,
        "write_policy": {"mode": "append_only", "overwrite_registry": False},
    }
    body["idempotency_key"] = digest({
        "cluster_id": cluster_id,
        "members": body["member_event_ids"],
        "config_sha256": config["config_sha256"],
    })
    body["event_id"] = sha(body)
    validate_cluster_proposal(body)
    return body


def validate_cluster_proposal(event):
    required = {
        "schema_version", "event_type", "proposed_at", "cluster_id",
        "member_event_ids", "member_event_sha256", "representative",
        "candidate_names_and_aliases", "source_kinds", "source_partition_event_ids",
        "clustering", "status",
        "human_review_required", "ontology_mutation_allowed", "write_policy",
        "idempotency_key", "event_id",
    }
    if not isinstance(event, dict) or set(event) != required:
        raise ValueError("cluster proposal fields are incomplete or unknown")
    if event["schema_version"] != "labelagent.ontology-cluster-proposal.v1" or event["event_type"] != "ontology.unknown_cluster.proposed":
        raise ValueError("invalid cluster proposal")
    _time(event["proposed_at"], "proposed_at")
    _text(event["cluster_id"], "cluster_id")
    if not event["member_event_ids"] or len(event["member_event_ids"]) != len(event["member_event_sha256"]):
        raise ValueError("cluster members are incomplete")
    for value in event["member_event_ids"] + event["member_event_sha256"]:
        _sha(value, "cluster member hash")
    representative = event["representative"]
    if set(representative) != {"event_id", "candidate_sha256", "selection_policy"}:
        raise ValueError("representative sample binding is incomplete")
    if representative["event_id"] not in event["member_event_ids"]:
        raise ValueError("representative must belong to cluster")
    _sha(representative["candidate_sha256"], "representative candidate")
    if representative["selection_policy"] != "max_view_evidence_then_confidence_then_event_id":
        raise ValueError("representative selection policy is invalid")
    if not event["candidate_names_and_aliases"]:
        raise ValueError("cluster proposal needs candidate names")
    if not set(event["source_kinds"]) <= {"real", "simulation"}:
        raise ValueError("cluster source_kind is invalid")
    partitions = event["source_partition_event_ids"]
    if set(partitions) != set(event["source_kinds"]):
        raise ValueError("cluster source partitions are incomplete")
    flattened = [value for values in partitions.values() for value in values]
    if sorted(flattened) != sorted(event["member_event_ids"]) or len(flattened) != len(set(flattened)):
        raise ValueError("cluster source partitions must exactly partition members")
    clustering = event["clustering"]
    if set(clustering) != {"algorithm", "version", "config_sha256"}:
        raise ValueError("clustering binding is incomplete")
    _text(clustering["algorithm"], "clustering algorithm")
    _text(clustering["version"], "clustering version")
    _sha(clustering["config_sha256"], "clustering config")
    if event["status"] != "proposal_only" or event["human_review_required"] is not True or event["ontology_mutation_allowed"] is not False:
        raise ValueError("clustering can only create a human-review proposal")
    if event["write_policy"] != {"mode": "append_only", "overwrite_registry": False}:
        raise ValueError("cluster write policy is unsafe")
    for field in ("idempotency_key", "event_id"):
        _sha(event[field], field)
    if event["event_id"] != sha({key: value for key, value in event.items() if key != "event_id"}):
        raise ValueError("cluster proposal event hash mismatch")


def _promotion_blockers(parent_registry, proposal, request):
    blockers = []
    try:
        validate_label_registry(parent_registry)
        validate_cluster_proposal(proposal)
    except ValueError:
        return ["PARENT_REGISTRY_OR_PROPOSAL_INVALID"]
    if request.get("proposal_event_id") != proposal["event_id"]:
        blockers.append("PROPOSAL_BINDING_MISMATCH")
    human = request.get("human_decision", {})
    if human.get("decision") != "approved_create_new_label" or human.get("identity_assurance") != "authenticated":
        blockers.append("AUTHENTICATED_HUMAN_APPROVAL_REQUIRED")
    for field in ("canonical_name", "definition", "reviewer_id", "approval_id", "receipt_uri"):
        try:
            _text(human.get(field), field)
        except ValueError:
            blockers.append("HUMAN_NAMING_OR_APPROVAL_EVIDENCE_MISSING")
            break
    try:
        _time(human.get("reviewed_at"), "reviewed_at")
        _sha(human.get("receipt_sha256"), "receipt_sha256")
    except ValueError:
        blockers.append("HUMAN_NAMING_OR_APPROVAL_EVIDENCE_MISSING")
    class_id = human.get("class_id")
    if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 1:
        blockers.append("CLASS_ID_INVALID")
    existing_ids = {item["class_id"] for item in parent_registry["labels"]}
    existing_names = {item["name"] for item in parent_registry["labels"]}
    if class_id in existing_ids or human.get("canonical_name") in existing_names:
        blockers.append("CLASS_ID_OR_NAME_ALREADY_REGISTERED")
    regression = request.get("regression_validation", {})
    required_regression = {
        "fixed_set_id", "manifest_uri", "manifest_sha256", "config_sha256",
        "real_metrics", "simulation_metrics", "truth_versions",
        "all_required_gates_passed", "validated_at",
    }
    if set(regression) != required_regression:
        blockers.append("FIXED_REGRESSION_VALIDATION_INCOMPLETE")
    else:
        try:
            _text(regression["fixed_set_id"], "fixed_set_id")
            _text(regression["manifest_uri"], "manifest_uri")
            _sha(regression["manifest_sha256"], "manifest_sha256")
            _sha(regression["config_sha256"], "config_sha256")
            _time(regression["validated_at"], "validated_at")
        except ValueError:
            blockers.append("FIXED_REGRESSION_VALIDATION_INCOMPLETE")
        if regression.get("all_required_gates_passed") is not True:
            blockers.append("FIXED_REGRESSION_GATES_NOT_PASSED")
        for key, code in (("real_metrics", "REAL_METRICS_MISSING"),
                          ("simulation_metrics", "SIMULATION_METRICS_MISSING")):
            metrics = regression.get(key, {})
            if set(metrics) != {"sample_count", "precision", "recall", "false_positive_rate"}:
                blockers.append(code)
                continue
            if isinstance(metrics["sample_count"], bool) or not isinstance(metrics["sample_count"], int) or metrics["sample_count"] < 0:
                blockers.append(code)
            for field in ("precision", "recall", "false_positive_rate"):
                if isinstance(metrics[field], bool) or not isinstance(metrics[field], (int, float)) or not 0 <= metrics[field] <= 1:
                    blockers.append(code)
        truth_versions = regression.get("truth_versions", {})
        if (set(truth_versions) != {"real", "simulation"}
                or truth_versions.get("real") == truth_versions.get("simulation")
                or not all(isinstance(value, str) and value for value in truth_versions.values())):
            blockers.append("REAL_SIM_TRUTH_VERSIONS_NOT_SEPARATE")
    version = request.get("version", {})
    if set(version) != {"registry_version", "parent_registry_version", "parent_registry_sha256"}:
        blockers.append("REGISTRY_VERSION_LINEAGE_MISSING")
    else:
        if version["parent_registry_version"] != parent_registry["registry_version"] or version["parent_registry_sha256"] != parent_registry["registry_content_sha256"]:
            blockers.append("PARENT_REGISTRY_BINDING_MISMATCH")
        if version["registry_version"] == parent_registry["registry_version"]:
            blockers.append("REGISTRY_VERSION_MUST_ADVANCE")
    return sorted(set(blockers))


def evaluate_ontology_promotion(parent_registry, proposal, request, *, occurred_at):
    blockers = _promotion_blockers(parent_registry, proposal, request)
    common = {
        "occurred_at": occurred_at,
        "proposal_event_id": proposal.get("event_id"),
        "parent_registry_version": parent_registry.get("registry_version"),
        "parent_registry_sha256": parent_registry.get("registry_content_sha256"),
    }
    if blockers:
        body = {
            "schema_version": "labelagent.ontology-promotion-rejection.v1",
            "event_type": "ontology.promotion.rejected", **common,
            "status": "rejected_fail_closed", "blockers": blockers,
            "registry_version_created": False, "old_registry_modified": False,
            "replay_plan_created": False, "next_state": "human_review_required",
        }
        body["idempotency_key"] = digest({**common, "blockers": blockers,
                                           "requested_version": request.get("version", {}).get("registry_version")})
        body["event_id"] = sha(body)
        validate_promotion_event(body)
        return body

    human = request["human_decision"]
    labels = [dict(item) for item in parent_registry["labels"]]
    labels.append({"class_id": human["class_id"], "name": human["canonical_name"],
                   "training_label": True})
    registry_body = {
        "schema_version": "labelagent.label-registry.v1",
        "registry_version": request["version"]["registry_version"],
        "append_only": True, "labels": labels,
        "training_semantics": parent_registry["training_semantics"],
        "downstream_mapping": parent_registry["downstream_mapping"],
        "unknown_policy": parent_registry["unknown_policy"],
    }
    registry_body["registry_content_sha256"] = digest(registry_body)
    validate_label_registry(registry_body)
    replay = {
        "schema_version": "labelagent.ontology-replay-plan.v1",
        "plan_id": digest({"proposal": proposal["event_id"],
                            "registry": registry_body["registry_content_sha256"]}),
        "source_registry_version": parent_registry["registry_version"],
        "target_registry_version": registry_body["registry_version"],
        "affected_candidate_event_ids": proposal["member_event_ids"],
        "source_partitions": proposal["source_partition_event_ids"],
        "execution_status": "planned_not_executed",
        "automatic_truth_promotion": False,
        "write_policy": {"mode": "append_only_new_outputs", "overwrite_history": False},
    }
    body = {
        "schema_version": "labelagent.ontology-registry-promotion.v1",
        "event_type": "ontology.registry.version.promoted", **common,
        "proposal": {"cluster_id": proposal["cluster_id"],
                     "representative_event_id": proposal["representative"]["event_id"]},
        "human_decision": human,
        "regression_validation": request["regression_validation"],
        "new_registry": registry_body,
        "replay_plan": replay,
        "write_policy": {"mode": "append_only_new_registry_version",
                         "overwrite_parent_registry": False},
    }
    body["idempotency_key"] = digest({
        "proposal": proposal["event_id"],
        "approval": human["approval_id"],
        "registry": registry_body["registry_content_sha256"],
    })
    body["event_id"] = sha(body)
    validate_promotion_event(body)
    return body


def validate_replay_plan(plan):
    if not isinstance(plan, dict) or set(plan) != {
        "schema_version", "plan_id", "source_registry_version",
        "target_registry_version", "affected_candidate_event_ids",
        "source_partitions", "execution_status", "automatic_truth_promotion",
        "write_policy",
    }:
        raise ValueError("ontology replay plan fields are incomplete")
    if plan["schema_version"] != "labelagent.ontology-replay-plan.v1":
        raise ValueError("invalid ontology replay plan")
    _sha(plan["plan_id"], "plan_id")
    if not plan["affected_candidate_event_ids"]:
        raise ValueError("ontology replay plan is empty")
    for value in plan["affected_candidate_event_ids"]:
        _sha(value, "affected event id")
    if set(plan["source_partitions"]) - {"real", "simulation"}:
        raise ValueError("replay source partitions are invalid")
    flattened = [value for values in plan["source_partitions"].values()
                 for value in values]
    if (sorted(flattened) != sorted(plan["affected_candidate_event_ids"])
            or len(flattened) != len(set(flattened))):
        raise ValueError("replay partitions must exactly partition affected candidates")
    if plan["execution_status"] != "planned_not_executed" or plan["automatic_truth_promotion"] is not False:
        raise ValueError("ontology replay must remain unexecuted and candidate-only")
    if plan["write_policy"] != {"mode": "append_only_new_outputs", "overwrite_history": False}:
        raise ValueError("ontology replay write policy is unsafe")


def validate_promotion_event(event):
    event_type = event.get("event_type")
    if event_type == "ontology.promotion.rejected":
        required = {
            "schema_version", "event_type", "occurred_at", "proposal_event_id",
            "parent_registry_version", "parent_registry_sha256", "status",
            "blockers", "registry_version_created", "old_registry_modified",
            "replay_plan_created", "next_state", "idempotency_key", "event_id",
        }
        if set(event) != required or event["schema_version"] != "labelagent.ontology-promotion-rejection.v1":
            raise ValueError("invalid ontology promotion rejection")
        if event["status"] != "rejected_fail_closed" or not event["blockers"]:
            raise ValueError("ontology rejection requires blockers")
        if any(event[field] is not False for field in (
                "registry_version_created", "old_registry_modified", "replay_plan_created")):
            raise ValueError("rejected ontology promotion cannot mutate state")
    elif event_type == "ontology.registry.version.promoted":
        required = {
            "schema_version", "event_type", "occurred_at", "proposal_event_id",
            "parent_registry_version", "parent_registry_sha256", "proposal",
            "human_decision", "regression_validation", "new_registry",
            "replay_plan", "write_policy", "idempotency_key", "event_id",
        }
        if set(event) != required or event["schema_version"] != "labelagent.ontology-registry-promotion.v1":
            raise ValueError("invalid ontology promotion event")
        validate_label_registry(event["new_registry"])
        validate_replay_plan(event["replay_plan"])
        human = event["human_decision"]
        if human.get("decision") != "approved_create_new_label" or human.get("identity_assurance") != "authenticated":
            raise ValueError("ontology promotion requires authenticated human approval")
        if set(human) != {"decision", "canonical_name", "definition", "class_id",
                          "reviewer_id", "identity_assurance", "reviewed_at",
                          "approval_id", "receipt_uri", "receipt_sha256"}:
            raise ValueError("ontology human decision fields are incomplete")
        for field in ("canonical_name", "definition", "reviewer_id",
                      "approval_id", "receipt_uri"):
            _text(human[field], field)
        _time(human["reviewed_at"], "reviewed_at")
        _sha(human["receipt_sha256"], "receipt_sha256")
        if isinstance(human["class_id"], bool) or not isinstance(human["class_id"], int) or human["class_id"] < 1:
            raise ValueError("ontology class_id is invalid")
        if event["write_policy"] != {"mode": "append_only_new_registry_version", "overwrite_parent_registry": False}:
            raise ValueError("ontology promotion cannot overwrite parent registry")
        if event["new_registry"]["registry_version"] == event["parent_registry_version"]:
            raise ValueError("ontology registry version did not advance")
        regression = event["regression_validation"]
        if set(regression) != {"fixed_set_id", "manifest_uri", "manifest_sha256",
                               "config_sha256", "real_metrics", "simulation_metrics",
                               "truth_versions", "all_required_gates_passed",
                               "validated_at"}:
            raise ValueError("ontology regression fields are incomplete")
        if regression.get("all_required_gates_passed") is not True or set(regression.get("truth_versions", {})) != {"real", "simulation"}:
            raise ValueError("ontology promotion regression gate is incomplete")
        if regression["truth_versions"]["real"] == regression["truth_versions"]["simulation"]:
            raise ValueError("real/sim truth versions must be separate")
        for field in ("fixed_set_id", "manifest_uri"):
            _text(regression[field], field)
        for field in ("manifest_sha256", "config_sha256"):
            _sha(regression[field], field)
        _time(regression["validated_at"], "validated_at")
        for name in ("real_metrics", "simulation_metrics"):
            metrics = regression[name]
            if set(metrics) != {"sample_count", "precision", "recall", "false_positive_rate"}:
                raise ValueError("real/sim metrics are incomplete")
            if isinstance(metrics["sample_count"], bool) or not isinstance(metrics["sample_count"], int) or metrics["sample_count"] < 0:
                raise ValueError("metric sample_count is invalid")
            if any(isinstance(metrics[field], bool) or not isinstance(metrics[field], (int, float)) or not 0 <= metrics[field] <= 1
                   for field in ("precision", "recall", "false_positive_rate")):
                raise ValueError("real/sim metric value is invalid")
    else:
        raise ValueError("unsupported ontology promotion event")
    _time(event["occurred_at"], "occurred_at")
    for field in ("proposal_event_id", "parent_registry_sha256", "idempotency_key", "event_id"):
        _sha(event[field], field)
    if event["event_id"] != sha({key: value for key, value in event.items() if key != "event_id"}):
        raise ValueError("ontology promotion event hash mismatch")


def validate_domain_event(event):
    if event.get("event_type") == "vlm.label_observation.recorded":
        return validate_label_observation(event)
    if event.get("event_type") == "ontology.unknown_cluster.proposed":
        return validate_cluster_proposal(event)
    return validate_promotion_event(event)


def _record(event, sequence, previous):
    record = {"sequence": sequence, "previous_record_sha256": previous,
              "event_type": event["event_type"],
              "idempotency_key": event["idempotency_key"],
              "event_sha256": digest(event), "event": event}
    record["record_sha256"] = digest(record)
    return record


def build_ontology_ledger(events, *, ledger_id, generated_at, sample_kind):
    records, previous, registry_versions = [], None, set()
    for index, event in enumerate(events, 1):
        validate_domain_event(event)
        if event["event_type"] == "ontology.registry.version.promoted":
            version = event["new_registry"]["registry_version"]
            if version in registry_versions:
                raise ValueError("ontology registry versions must be unique")
            registry_versions.add(version)
        record = _record(event, index, previous)
        records.append(record)
        previous = record["record_sha256"]
    body = {
        "schema_version": "labelagent.ontology-audit-ledger.v1",
        "ledger_id": ledger_id, "generated_at": generated_at,
        "sample_kind": sample_kind, "append_only": True,
        "record_count": len(records), "records": records,
        "head_record_sha256": previous,
    }
    body["snapshot_sha256"] = digest(body)
    validate_ontology_ledger(body)
    return body


def validate_ontology_ledger(snapshot):
    if snapshot.get("schema_version") != "labelagent.ontology-audit-ledger.v1" or snapshot.get("append_only") is not True:
        raise ValueError("invalid ontology audit ledger")
    _time(snapshot.get("generated_at"), "generated_at")
    records = snapshot.get("records")
    if not isinstance(records, list) or snapshot.get("record_count") != len(records):
        raise ValueError("ontology ledger record count mismatch")
    previous, versions = None, set()
    for index, record in enumerate(records, 1):
        if record.get("sequence") != index or record.get("previous_record_sha256") != previous:
            raise ValueError("ontology ledger hash chain mismatch")
        event = record.get("event")
        validate_domain_event(event)
        if record.get("event_sha256") != digest(event):
            raise ValueError("ontology ledger event hash mismatch")
        if record.get("record_sha256") != digest({key: value for key, value in record.items() if key != "record_sha256"}):
            raise ValueError("ontology ledger record hash mismatch")
        if event["event_type"] == "ontology.registry.version.promoted":
            version = event["new_registry"]["registry_version"]
            if version in versions:
                raise ValueError("duplicate ontology registry version")
            versions.add(version)
        previous = record["record_sha256"]
    if snapshot.get("head_record_sha256") != previous:
        raise ValueError("ontology ledger head mismatch")
    if snapshot.get("snapshot_sha256") != digest({key: value for key, value in snapshot.items() if key != "snapshot_sha256"}):
        raise ValueError("ontology ledger snapshot hash mismatch")


class AppendOnlyOntologyLedger:
    def __init__(self, root, ledger_id):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger_id = _text(ledger_id, "ledger_id")

    def records(self):
        values = [json.loads(path.read_text(encoding="utf-8")) for path in
                  sorted(self.root.glob("[0-9][0-9][0-9][0-9][0-9][0-9]-*.json"))]
        snapshot = build_ontology_ledger(
            [item["event"] for item in values], ledger_id=self.ledger_id,
            generated_at="1970-01-01T00:00:00Z", sample_kind="runtime_readback")
        if values != snapshot["records"]:
            raise ValueError("stored ontology ledger differs from canonical chain")
        return values

    def append(self, event):
        validate_domain_event(event)
        records = self.records()
        event_sha = digest(event)
        for record in records:
            if record["idempotency_key"] == event["idempotency_key"]:
                if record["event_sha256"] != event_sha:
                    raise ValueError("ontology idempotency replay changed event")
                return record, True
        if event["event_type"] == "ontology.registry.version.promoted":
            version = event["new_registry"]["registry_version"]
            if any(item["event"].get("new_registry", {}).get("registry_version") == version
                   for item in records):
                raise ValueError("ontology registry version already exists")
        previous = records[-1]["record_sha256"] if records else None
        record = _record(event, len(records) + 1, previous)
        path = self.root / f"{record['sequence']:06d}-{record['record_sha256']}.json"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
        except Exception:
            path.unlink(missing_ok=True)
            raise
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError("ontology ledger readback mismatch")
        return record, False
