"""CPU-only, event-driven factory LabelAgent orchestration contract.

This module validates and replays immutable events.  It never calls a model,
GPU, remote service, CVAT, SSH, OSS, or a training launcher.
"""
from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re


SCHEMA_VERSION = "labelagent.factory-orchestration-event.v1"
LEDGER_VERSION = "labelagent.factory-orchestration-ledger.v1"
HASH = re.compile(r"^[0-9a-f]{64}$")
SPLITS = {"train", "val", "test"}
SOURCE_KINDS = {"real", "simulation"}
SAMPLE_KINDS = {"production", "synthetic_contract_fixture_not_production"}
CLASSIFICATIONS = {"not_applicable", "known_label", "unknown_candidate",
                   "uncertain_conflict"}
AGENT_ACTIONS = {"not_applicable", "split", "merge", "resegment",
                 "reclassify", "drop", "no_op"}
BUSINESS_STATES = {
    "ingested", "qc_passed", "quarantined", "candidate_planned",
    "vlm_reviewed", "agent_corrected", "vlm_rechecked",
    "human_review_required", "candidate_accepted", "truth_promoted",
    "evaluation_truth", "training_eligible", "trained", "evaluated",
    "deployment_candidate", "baseline_retained", "deployed", "rolled_back",
}
TRANSITIONS = {
    None: {"ingested"},
    "ingested": {"qc_passed", "quarantined"},
    "qc_passed": {"candidate_planned"},
    "candidate_planned": {"vlm_reviewed"},
    "vlm_reviewed": {"agent_corrected"},
    "agent_corrected": {"vlm_rechecked"},
    "vlm_rechecked": {"human_review_required", "candidate_accepted"},
    "human_review_required": {"candidate_accepted", "evaluation_truth"},
    "candidate_accepted": {"truth_promoted", "evaluation_truth"},
    "truth_promoted": {"training_eligible"},
    "training_eligible": {"trained"},
    "trained": {"evaluated"},
    "evaluated": {"deployment_candidate", "baseline_retained"},
    "deployment_candidate": {"deployed"},
    "deployed": {"rolled_back"},
    "quarantined": set(),
    "evaluation_truth": set(),
    "baseline_retained": set(),
    "rolled_back": set(),
}
EVENT_TYPES = {
    "state.transition", "operation.failed", "operation.retry_released",
    "queue.backpressure_applied", "queue.backpressure_released",
}
OPERATIONAL_STATUS = {"ready", "retry_wait", "backpressured", "dead_letter",
                      "quarantined"}
CONTRACT_TYPES_BY_STATE = {
    "ingested": {"factory_ingress"},
    "qc_passed": {"factory_qc"},
    "quarantined": {"factory_qc"},
    "candidate_planned": {"candidate_batch_plan"},
    "vlm_reviewed": {"vlm_label_observation"},
    "agent_corrected": {"closed_loop_agent_action"},
    "vlm_rechecked": {"closed_loop_vlm_recheck"},
    "human_review_required": {"human_review_queue", "cvat_offline_export"},
    "candidate_accepted": {"human_review_receipt", "cvat_offline_return"},
    "truth_promoted": {"immutable_truth_promotion"},
    "evaluation_truth": {"cvat_evaluation_truth"},
    "training_eligible": {"sam3_stage0_dry_run_plan"},
    "trained": {"sam3_stage0_model_card"},
    "evaluated": {"sam3_stage0_deployment_decision"},
    "deployment_candidate": {"sam3_stage0_deployment_decision"},
    "baseline_retained": {"sam3_stage0_deployment_decision"},
    "deployed": {"deployment_receipt"},
    "rolled_back": {"rollback_receipt"},
}
METRICS = {"mask_iou", "boundary_f1", "object_precision", "object_recall",
           "false_positives"}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes)
                          else canonical_json(value)).hexdigest()


def _sha(value, name):
    if not isinstance(value, str) or HASH.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _time(value, name):
    _text(value, name)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include timezone")
    return value


def stable_stream_id(identity):
    required = {"dataset_id", "source_batch", "sequence", "frame_id",
                "camera", "split"}
    if set(identity) != required:
        raise ValueError("identity fields are incomplete or contain extras")
    for key in required - {"split"}:
        _text(identity[key], f"identity.{key}")
    if identity["split"] not in SPLITS:
        raise ValueError("identity.split is invalid")
    return digest(identity)


def stable_batch_key(identity):
    return digest({key: identity[key]
                   for key in ("dataset_id", "source_batch", "split")})


def artifact(kind, uri, sha256):
    return {"kind": _text(kind, "artifact.kind"),
            "uri": _text(uri, "artifact.uri"),
            "sha256": _sha(sha256, "artifact.sha256")}


def contract_ref(contract_type, schema_path, artifact_uri, artifact_sha256,
                 *, source_event_id=None):
    result = {
        "contract_type": _text(contract_type, "contract_type"),
        "schema_path": _text(schema_path, "schema_path"),
        "artifact_uri": _text(artifact_uri, "artifact_uri"),
        "artifact_sha256": _sha(artifact_sha256, "artifact_sha256"),
        "source_event_id": source_event_id,
    }
    if source_event_id is not None:
        _sha(source_event_id, "source_event_id")
    return result


def _validate_metrics_view(value, name):
    if not isinstance(value, dict) or set(value) != METRICS:
        raise ValueError(f"{name} must contain the complete metric set")
    for metric in METRICS - {"false_positives"}:
        number = value[metric]
        if isinstance(number, bool) or not isinstance(number, (int, float)) \
                or not 0 <= number <= 1:
            raise ValueError(f"{name}.{metric} is invalid")
    false_positives = value["false_positives"]
    if isinstance(false_positives, bool) or not isinstance(false_positives, int) \
            or false_positives < 0:
        raise ValueError(f"{name}.false_positives is invalid")


def _validate_evaluation(value):
    if value is None:
        return
    required = {"fixed_holdout_manifest_sha256", "overall", "per_class",
                "source_real", "source_simulation", "overall_guard_passed",
                "slope_guard_passed", "deployment_gate_passed"}
    if set(value) != required:
        raise ValueError("evaluation fields are incomplete or contain extras")
    _sha(value["fixed_holdout_manifest_sha256"],
         "evaluation.fixed_holdout_manifest_sha256")
    _validate_metrics_view(value["overall"], "evaluation.overall")
    if set(value["per_class"]) != {"floor", "slope"}:
        raise ValueError("evaluation must report floor and slope separately")
    _validate_metrics_view(value["per_class"]["floor"],
                           "evaluation.per_class.floor")
    _validate_metrics_view(value["per_class"]["slope"],
                           "evaluation.per_class.slope")
    _validate_metrics_view(value["source_real"], "evaluation.source_real")
    _validate_metrics_view(value["source_simulation"],
                           "evaluation.source_simulation")
    for field in ("overall_guard_passed", "slope_guard_passed",
                  "deployment_gate_passed"):
        if not isinstance(value[field], bool):
            raise ValueError(f"evaluation.{field} must be boolean")
    expected = value["overall_guard_passed"] and value["slope_guard_passed"]
    if value["deployment_gate_passed"] is not expected:
        raise ValueError("deployment gate must equal overall AND slope guards")


def _validate_provenance(value):
    required = {"source_kind", "simulation", "sample_kind", "execution_mode",
                "external_effects_allowed"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("provenance fields are incomplete or contain extras")
    if value["source_kind"] not in SOURCE_KINDS:
        raise ValueError("provenance.source_kind is invalid")
    if value["sample_kind"] not in SAMPLE_KINDS:
        raise ValueError("provenance.sample_kind is invalid")
    if value["execution_mode"] not in {"static_replay", "live_contract"}:
        raise ValueError("provenance.execution_mode is invalid")
    if not isinstance(value["external_effects_allowed"], bool):
        raise ValueError("provenance.external_effects_allowed must be boolean")
    simulation = value["simulation"]
    if value["source_kind"] == "real" and simulation is not None:
        raise ValueError("real provenance cannot contain simulation metadata")
    if value["source_kind"] == "simulation":
        fields = {"simulator", "scene", "random_seed", "version"}
        if not isinstance(simulation, dict) or set(simulation) != fields:
            raise ValueError("simulation provenance is incomplete")
    if value["execution_mode"] == "static_replay" \
            and value["external_effects_allowed"] is not False:
        raise ValueError("static replay cannot allow external effects")


def _validate_human_gate(value):
    required = {"required", "status", "review_event_id", "cvat_task_version",
                "receipt_sha256", "candidate_sha256", "source_image_sha256"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("human_gate fields are incomplete or contain extras")
    if not isinstance(value["required"], bool):
        raise ValueError("human_gate.required must be boolean")
    if value["status"] not in {"not_required", "pending", "approved", "rejected"}:
        raise ValueError("human_gate.status is invalid")
    for field in ("review_event_id", "cvat_task_version", "receipt_sha256",
                  "candidate_sha256", "source_image_sha256"):
        if value[field] is not None:
            (_sha(value[field], f"human_gate.{field}")
             if field != "cvat_task_version"
             else _text(value[field], f"human_gate.{field}"))
    if value["status"] == "approved":
        for field in ("review_event_id", "receipt_sha256", "candidate_sha256",
                      "source_image_sha256"):
            if value[field] is None:
                raise ValueError(f"approved human gate requires {field}")


def _event_body(event):
    return {key: value for key, value in event.items() if key != "event_id"}


def validate_event(event):
    required = {
        "schema_version", "event_id", "event_key", "event_type", "occurred_at",
        "trace_id", "stream_id", "batch_key", "previous_event_id",
        "state_version", "previous_state", "state", "operational_status",
        "identity", "bindings", "retry", "queue", "human_gate", "provenance",
        "label_policy",
        "classification", "agent_action", "reason_codes", "failure_reason",
        "artifacts", "contract_refs", "evaluation", "truth_version",
        "training_run", "effect",
    }
    if not isinstance(event, dict) or set(event) != required:
        raise ValueError("orchestration event fields are incomplete or contain extras")
    if event["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported orchestration event schema")
    _sha(event["event_id"], "event_id")
    _sha(event["event_key"], "event_key")
    if event["event_type"] not in EVENT_TYPES:
        raise ValueError("event_type is invalid")
    _time(event["occurred_at"], "occurred_at")
    _text(event["trace_id"], "trace_id")
    if stable_stream_id(event["identity"]) != event["stream_id"]:
        raise ValueError("stream_id does not match identity")
    if stable_batch_key(event["identity"]) != event["batch_key"]:
        raise ValueError("batch_key does not match identity")
    if event["previous_event_id"] is not None:
        _sha(event["previous_event_id"], "previous_event_id")
    if not isinstance(event["state_version"], int) or event["state_version"] < 1:
        raise ValueError("state_version must be a positive integer")
    if event["previous_state"] is not None \
            and event["previous_state"] not in BUSINESS_STATES:
        raise ValueError("previous_state is invalid")
    if event["state"] not in BUSINESS_STATES:
        raise ValueError("state is invalid")
    if event["operational_status"] not in OPERATIONAL_STATUS:
        raise ValueError("operational_status is invalid")
    bindings = event["bindings"]
    if set(bindings) != {"input_sha256", "output_sha256", "config_sha256",
                         "model_sha256", "prompt_sha256"}:
        raise ValueError("bindings are incomplete or contain extras")
    for key, value in bindings.items():
        if key in {"input_sha256", "output_sha256"}:
            _sha(value, f"bindings.{key}")
        elif value is not None:
            _sha(value, f"bindings.{key}")
    _validate_human_gate(event["human_gate"])
    _validate_provenance(event["provenance"])
    label_policy = event["label_policy"]
    if label_policy != {
            "semantic_labels": ["floor", "slope"],
            "preserve_separate_training_labels": True,
            "downstream_target": "traversable_area",
            "downstream_operation": "union",
            "downstream_source_labels": ["floor", "slope"],
            "slope_default": "passable",
            "policy_basis": "user_defined_project_semantics"}:
        raise ValueError("floor/slope two-layer label policy is invalid")
    if event["classification"] not in CLASSIFICATIONS:
        raise ValueError("classification is invalid")
    if event["agent_action"] not in AGENT_ACTIONS:
        raise ValueError("agent_action is invalid")
    if not isinstance(event["reason_codes"], list) \
            or len(event["reason_codes"]) != len(set(event["reason_codes"])):
        raise ValueError("reason_codes must be a unique list")
    for reason in event["reason_codes"]:
        _text(reason, "reason_code")
    if event["failure_reason"] is not None:
        _text(event["failure_reason"], "failure_reason")
    retry = event["retry"]
    if set(retry) != {"attempt", "max_attempts", "retryable"} \
            or not isinstance(retry["attempt"], int) or retry["attempt"] < 0 \
            or not isinstance(retry["max_attempts"], int) \
            or retry["max_attempts"] < 0 \
            or not isinstance(retry["retryable"], bool):
        raise ValueError("retry contract is invalid")
    queue = event["queue"]
    if set(queue) != {"name", "depth", "high_watermark", "route"} \
            or queue["route"] not in {"active", "backpressure", "retry",
                                      "dead_letter", "quarantine"} \
            or not isinstance(queue["depth"], int) or queue["depth"] < 0 \
            or not isinstance(queue["high_watermark"], int) \
            or queue["high_watermark"] < 1:
        raise ValueError("queue contract is invalid")
    for item in event["artifacts"]:
        if set(item) != {"kind", "uri", "sha256"}:
            raise ValueError("artifact fields are invalid")
        artifact(item["kind"], item["uri"], item["sha256"])
    if not event["contract_refs"]:
        raise ValueError("at least one contract reference is required")
    for item in event["contract_refs"]:
        if set(item) != {"contract_type", "schema_path", "artifact_uri",
                         "artifact_sha256", "source_event_id"}:
            raise ValueError("contract reference fields are invalid")
        contract_ref(**item)
    if event["event_type"] == "state.transition":
        expected = CONTRACT_TYPES_BY_STATE[event["state"]]
        actual = {item["contract_type"] for item in event["contract_refs"]}
        if not actual.intersection(expected):
            raise ValueError(f"state {event['state']} lacks required contract reference")
    _validate_evaluation(event["evaluation"])
    if event["truth_version"] is not None:
        if set(event["truth_version"]) != {"version_id", "manifest_sha256",
                                           "split", "immutable"}:
            raise ValueError("truth_version fields are invalid")
        _text(event["truth_version"]["version_id"], "truth_version.version_id")
        _sha(event["truth_version"]["manifest_sha256"],
             "truth_version.manifest_sha256")
        if event["truth_version"]["split"] not in SPLITS \
                or event["truth_version"]["immutable"] is not True:
            raise ValueError("truth_version must be immutable and split-bound")
    if event["training_run"] is not None:
        fields = {"run_id", "package_sha256", "checkpoint_sha256",
                  "simulated_static_replay"}
        if set(event["training_run"]) != fields:
            raise ValueError("training_run fields are invalid")
        _text(event["training_run"]["run_id"], "training_run.run_id")
        _sha(event["training_run"]["package_sha256"],
             "training_run.package_sha256")
        if event["training_run"]["checkpoint_sha256"] is not None:
            _sha(event["training_run"]["checkpoint_sha256"],
                 "training_run.checkpoint_sha256")
        if not isinstance(event["training_run"]["simulated_static_replay"], bool):
            raise ValueError("training_run.simulated_static_replay must be boolean")
    effect = event["effect"]
    if set(effect) != {"mode", "external_call_count", "production_mutation"} \
            or effect["mode"] not in {"none", "simulated", "result_reference"} \
            or not isinstance(effect["external_call_count"], int) \
            or effect["external_call_count"] < 0 \
            or not isinstance(effect["production_mutation"], bool):
        raise ValueError("effect contract is invalid")
    if event["provenance"]["execution_mode"] == "static_replay":
        if effect["external_call_count"] != 0 \
                or effect["production_mutation"] is not False:
            raise ValueError("static replay must have zero external calls/mutations")
    expected_key = digest({
        "stream_id": event["stream_id"], "event_type": event["event_type"],
        "state_version": event["state_version"],
        "previous_state": event["previous_state"], "state": event["state"],
        "input_sha256": bindings["input_sha256"],
        "output_sha256": bindings["output_sha256"],
    })
    if event["event_key"] != expected_key:
        raise ValueError("event_key does not match stable event inputs")
    if event["event_id"] != digest(_event_body(event)):
        raise ValueError("event_id does not match immutable event body")
    return True


def build_event(*, event_type, occurred_at, trace_id, identity, state_version,
                previous_state, state, previous_event_id, operational_status="ready",
                bindings=None, retry=None, queue=None, human_gate=None,
                provenance=None, classification="not_applicable",
                label_policy=None,
                agent_action="not_applicable", reason_codes=None,
                failure_reason=None, artifacts=None, contract_refs=None,
                evaluation=None, truth_version=None, training_run=None,
                effect=None):
    stream_id = stable_stream_id(identity)
    bindings = bindings or {
        "input_sha256": digest({"stream": stream_id, "version": state_version,
                                 "direction": "input"}),
        "output_sha256": digest({"stream": stream_id, "version": state_version,
                                  "direction": "output"}),
        "config_sha256": None, "model_sha256": None, "prompt_sha256": None,
    }
    retry = retry or {"attempt": 0, "max_attempts": 2, "retryable": False}
    queue = queue or {"name": "factory_labelagent", "depth": 0,
                      "high_watermark": 32, "route": "active"}
    human_gate = human_gate or {
        "required": False, "status": "not_required", "review_event_id": None,
        "cvat_task_version": None, "receipt_sha256": None,
        "candidate_sha256": None, "source_image_sha256": None,
    }
    provenance = provenance or {
        "source_kind": "real", "simulation": None,
        "sample_kind": "synthetic_contract_fixture_not_production",
        "execution_mode": "static_replay", "external_effects_allowed": False,
    }
    effect = effect or {"mode": "none", "external_call_count": 0,
                        "production_mutation": False}
    label_policy = label_policy or {
        "semantic_labels": ["floor", "slope"],
        "preserve_separate_training_labels": True,
        "downstream_target": "traversable_area",
        "downstream_operation": "union",
        "downstream_source_labels": ["floor", "slope"],
        "slope_default": "passable",
        "policy_basis": "user_defined_project_semantics",
    }
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_id": "0" * 64,
        "event_key": digest({
            "stream_id": stream_id, "event_type": event_type,
            "state_version": state_version, "previous_state": previous_state,
            "state": state, "input_sha256": bindings["input_sha256"],
            "output_sha256": bindings["output_sha256"],
        }),
        "event_type": event_type, "occurred_at": occurred_at,
        "trace_id": trace_id, "stream_id": stream_id,
        "batch_key": stable_batch_key(identity),
        "previous_event_id": previous_event_id, "state_version": state_version,
        "previous_state": previous_state, "state": state,
        "operational_status": operational_status, "identity": copy.deepcopy(identity),
        "bindings": copy.deepcopy(bindings), "retry": copy.deepcopy(retry),
        "queue": copy.deepcopy(queue), "human_gate": copy.deepcopy(human_gate),
        "provenance": copy.deepcopy(provenance),
        "label_policy": copy.deepcopy(label_policy),
        "classification": classification, "agent_action": agent_action,
        "reason_codes": list(reason_codes or []), "failure_reason": failure_reason,
        "artifacts": copy.deepcopy(artifacts or []),
        "contract_refs": copy.deepcopy(contract_refs or []),
        "evaluation": copy.deepcopy(evaluation),
        "truth_version": copy.deepcopy(truth_version),
        "training_run": copy.deepcopy(training_run), "effect": copy.deepcopy(effect),
    }
    event["event_id"] = digest(_event_body(event))
    validate_event(event)
    return event


class FactoryOrchestrationLedger:
    """Append-only event ledger with idempotency, replay, and fail-closed gates."""

    def __init__(self, *, ledger_id, max_inflight=32, retry_budget=2):
        self.ledger_id = _text(ledger_id, "ledger_id")
        if max_inflight < 1 or retry_budget < 0:
            raise ValueError("queue/retry policy is invalid")
        self.max_inflight = max_inflight
        self.retry_budget = retry_budget
        self.records = []
        self._by_key = {}
        self._heads = {}

    def _validate_transition(self, event, head):
        previous_state = None if head is None else head["state"]
        previous_event_id = None if head is None else head["event_id"]
        previous_version = 0 if head is None else head["state_version"]
        if event["previous_state"] != previous_state \
                or event["previous_event_id"] != previous_event_id:
            raise ValueError("out-of-order event or stale previous state")
        if event["state_version"] != previous_version + 1:
            raise ValueError("state_version must increment exactly once")
        if head is not None:
            if event["identity"]["split"] != head["split"]:
                raise ValueError("cross-split transition is forbidden")
            if event["provenance"]["source_kind"] != head["source_kind"]:
                raise ValueError("source provenance cannot change within a stream")
            if head["operational_status"] == "dead_letter":
                raise ValueError("dead-letter stream cannot advance")
        if event["event_type"] == "state.transition":
            if event["state"] not in TRANSITIONS[previous_state]:
                raise ValueError(f"illegal state transition {previous_state}->{event['state']}")
            self._validate_state_gate(event, head)
        else:
            if head is None or event["state"] != previous_state:
                raise ValueError("operational event must retain existing business state")
            self._validate_operational_event(event, head)

    def _validate_state_gate(self, event, head):
        state = event["state"]
        split = event["identity"]["split"]
        gate = event["human_gate"]
        if state == "quarantined":
            if event["operational_status"] != "quarantined" \
                    or event["queue"]["route"] != "quarantine" \
                    or not event["failure_reason"]:
                raise ValueError("QC failure must enter explicit quarantine")
        if state == "qc_passed" and event["failure_reason"] is not None:
            raise ValueError("qc_passed cannot carry a failure reason")
        if state in {"human_review_required", "candidate_accepted",
                     "truth_promoted", "evaluation_truth"} and not gate["required"]:
            raise ValueError(f"{state} requires an explicit human gate")
        if state == "human_review_required" and gate["status"] != "pending":
            raise ValueError("human review route must remain pending")
        if state in {"candidate_accepted", "truth_promoted", "evaluation_truth"} \
                and gate["status"] != "approved":
            raise ValueError(f"{state} requires approved human review")
        if state == "candidate_accepted" and head is not None \
                and head["classification"] == "unknown_candidate" \
                and "ontology_human_approved" not in event["reason_codes"]:
            raise ValueError("unknown candidate requires ontology human approval")
        if state == "truth_promoted":
            if split == "test":
                raise ValueError("test split can only create evaluation truth")
            if event["truth_version"] is None \
                    or event["truth_version"]["split"] != split:
                raise ValueError("truth promotion requires same-split immutable version")
            if head is None or event["human_gate"] != head["human_gate"]:
                raise ValueError("truth promotion must retain exact accepted human gate")
        if state == "evaluation_truth":
            if split != "test" or event["truth_version"] is None \
                    or event["truth_version"]["split"] != "test":
                raise ValueError("evaluation truth is restricted to test split")
        if state == "training_eligible":
            if split != "train" or head is None or head["state"] != "truth_promoted":
                raise ValueError("only train truth can become training eligible")
            if event["provenance"]["sample_kind"] != "production" \
                    and event["effect"]["mode"] != "simulated":
                raise ValueError("synthetic fixtures cannot authorize real training")
            if event["truth_version"] is None \
                    or event["truth_version"] != head["truth_version"]:
                raise ValueError("training eligibility must bind exact truth version")
        if state == "trained" and event["training_run"] is None:
            raise ValueError("trained state requires a bound training run")
        if state == "trained" and (head is None
                                    or event["truth_version"] != head["truth_version"]):
            raise ValueError("training result must retain exact truth version")
        if state == "evaluated" and event["evaluation"] is None:
            raise ValueError("evaluated state requires full fixed-holdout metrics")
        if state == "evaluated" and (head is None
                                      or event["training_run"] != head["training_run"]
                                      or event["truth_version"] != head["truth_version"]):
            raise ValueError("evaluation must bind exact training run and truth version")
        if state == "deployment_candidate":
            if head is None or head["evaluation"] is None \
                    or head["evaluation"]["deployment_gate_passed"] is not True:
                raise ValueError("deployment candidate requires passed overall+slope gate")
            if event["evaluation"] != head["evaluation"] \
                    or event["training_run"] != head["training_run"]:
                raise ValueError("deployment candidate must bind exact evaluation/run")
        if state == "baseline_retained":
            if head is None or head["evaluation"] is None \
                    or head["evaluation"]["deployment_gate_passed"] is not False:
                raise ValueError("baseline retention requires a failed deployment gate")
            if event["evaluation"] != head["evaluation"] \
                    or event["training_run"] != head["training_run"]:
                raise ValueError("baseline decision must bind exact evaluation/run")
        if state in {"deployed", "rolled_back"} \
                and event["provenance"]["execution_mode"] == "static_replay" \
                and event["effect"]["mode"] != "simulated":
            raise ValueError("static deploy/rollback must be explicitly simulated")

    def _validate_operational_event(self, event, head):
        event_type = event["event_type"]
        if event_type == "operation.failed":
            if not event["failure_reason"] or event["retry"]["attempt"] \
                    != head["retry_attempt"] + 1:
                raise ValueError("failure must increment retry attempt with a reason")
            if event["retry"]["max_attempts"] != self.retry_budget:
                raise ValueError("retry budget does not match ledger policy")
            should_retry = event["retry"]["retryable"] \
                and event["retry"]["attempt"] <= self.retry_budget
            expected_status = "retry_wait" if should_retry else "dead_letter"
            expected_route = "retry" if should_retry else "dead_letter"
            if event["operational_status"] != expected_status \
                    or event["queue"]["route"] != expected_route:
                raise ValueError("retry/dead-letter route violates retry budget")
        elif event_type == "operation.retry_released":
            if head["operational_status"] != "retry_wait" \
                    or event["operational_status"] != "ready" \
                    or event["retry"]["attempt"] != head["retry_attempt"]:
                raise ValueError("only retry_wait can be released without reset")
        elif event_type == "queue.backpressure_applied":
            if event["queue"]["high_watermark"] != self.max_inflight \
                    or event["queue"]["depth"] < event["queue"]["high_watermark"] \
                    or event["operational_status"] != "backpressured" \
                    or event["queue"]["route"] != "backpressure":
                raise ValueError("backpressure requires high-watermark saturation")
        elif event_type == "queue.backpressure_released":
            if event["queue"]["high_watermark"] != self.max_inflight \
                    or head["operational_status"] != "backpressured" \
                    or event["queue"]["depth"] >= event["queue"]["high_watermark"] \
                    or event["operational_status"] != "ready":
                raise ValueError("backpressure release requires capacity")

    def append(self, event):
        validate_event(event)
        existing = self._by_key.get(event["event_key"])
        if existing is not None:
            if existing["event"] != event:
                raise ValueError("idempotency key collision with immutable mismatch")
            return copy.deepcopy(existing), True
        head = self._heads.get(event["stream_id"])
        self._validate_transition(event, head)
        previous_record_sha256 = (None if not self.records
                                  else self.records[-1]["record_sha256"])
        record = {
            "sequence": len(self.records) + 1,
            "event": copy.deepcopy(event),
            "event_sha256": digest(event),
            "previous_record_sha256": previous_record_sha256,
        }
        record["record_sha256"] = digest(record)
        self.records.append(record)
        self._by_key[event["event_key"]] = record
        retry_attempt = (event["retry"]["attempt"]
                         if event["event_type"] == "operation.failed"
                         else (0 if head is None else head["retry_attempt"]))
        self._heads[event["stream_id"]] = {
            "event_id": event["event_id"], "state": event["state"],
            "state_version": event["state_version"],
            "operational_status": event["operational_status"],
            "retry_attempt": retry_attempt, "split": event["identity"]["split"],
            "source_kind": event["provenance"]["source_kind"],
            "classification": event["classification"],
            "evaluation": copy.deepcopy(event["evaluation"]),
            "human_gate": copy.deepcopy(event["human_gate"]),
            "truth_version": copy.deepcopy(event["truth_version"]),
            "training_run": copy.deepcopy(event["training_run"]),
            "trace_id": event["trace_id"],
        }
        return copy.deepcopy(record), False

    def snapshot(self, *, generated_at):
        _time(generated_at, "generated_at")
        traces = {}
        for record in self.records:
            event = record["event"]
            trace = traces.setdefault(event["trace_id"], {
                "stream_id": event["stream_id"], "event_count": 0,
                "final_state": None, "final_operational_status": None,
            })
            trace["event_count"] += 1
            trace["final_state"] = event["state"]
            trace["final_operational_status"] = event["operational_status"]
        accounting = {kind: {"streams": 0, "events": 0}
                      for kind in ("real", "simulation")}
        for head in self._heads.values():
            accounting[head["source_kind"]]["streams"] += 1
        for record in self.records:
            accounting[record["event"]["provenance"]["source_kind"]]["events"] += 1
        queues = {
            "quarantine": sum(head["operational_status"] == "quarantined"
                              for head in self._heads.values()),
            "dead_letter": sum(head["operational_status"] == "dead_letter"
                               for head in self._heads.values()),
            "backpressured": sum(head["operational_status"] == "backpressured"
                                 for head in self._heads.values()),
            "retry_wait": sum(head["operational_status"] == "retry_wait"
                              for head in self._heads.values()),
        }
        result = {
            "schema_version": LEDGER_VERSION, "ledger_id": self.ledger_id,
            "generated_at": generated_at,
            "append_only": True, "overwrite_old_state": False,
            "silent_skip": False,
            "policy": {"max_inflight": self.max_inflight,
                       "retry_budget": self.retry_budget},
            "records": copy.deepcopy(self.records), "traces": traces,
            "queues": queues, "source_accounting": accounting,
            "external_call_counts": {"sam": 0, "vlm": 0, "agent": 0,
                                     "cvat": 0, "gpu": 0, "ssh": 0,
                                     "oss": 0, "training": 0,
                                     "deployment": 0},
        }
        result["ledger_sha256"] = digest(result)
        return result

    @classmethod
    def from_snapshot(cls, snapshot):
        validate_ledger_snapshot(snapshot)
        ledger = cls(ledger_id=snapshot["ledger_id"],
                     max_inflight=snapshot["policy"]["max_inflight"],
                     retry_budget=snapshot["policy"]["retry_budget"])
        for expected in snapshot["records"]:
            actual, replay = ledger.append(expected["event"])
            if replay or actual != expected:
                raise ValueError("ledger replay record mismatch")
        if ledger.snapshot(generated_at=snapshot["generated_at"]) != snapshot:
            raise ValueError("ledger snapshot metadata does not match replayed records")
        return ledger


def validate_ledger_snapshot(snapshot):
    required = {"schema_version", "ledger_id", "generated_at", "append_only",
                "overwrite_old_state", "silent_skip", "policy", "records",
                "traces", "queues", "source_accounting", "external_call_counts",
                "ledger_sha256"}
    if not isinstance(snapshot, dict) or set(snapshot) != required:
        raise ValueError("ledger snapshot fields are incomplete or contain extras")
    if snapshot["schema_version"] != LEDGER_VERSION \
            or snapshot["append_only"] is not True \
            or snapshot["overwrite_old_state"] is not False \
            or snapshot["silent_skip"] is not False:
        raise ValueError("ledger immutability policy is invalid")
    _time(snapshot["generated_at"], "generated_at")
    if snapshot["ledger_sha256"] != digest({
            key: value for key, value in snapshot.items()
            if key != "ledger_sha256"}):
        raise ValueError("ledger SHA-256 mismatch")
    previous = None
    for index, record in enumerate(snapshot["records"], 1):
        if set(record) != {"sequence", "event", "event_sha256",
                           "previous_record_sha256", "record_sha256"}:
            raise ValueError("ledger record fields are invalid")
        if record["sequence"] != index \
                or record["previous_record_sha256"] != previous:
            raise ValueError("ledger record chain is broken")
        validate_event(record["event"])
        if record["event_sha256"] != digest(record["event"]):
            raise ValueError("ledger event SHA-256 mismatch")
        expected = digest({key: value for key, value in record.items()
                           if key != "record_sha256"})
        if record["record_sha256"] != expected:
            raise ValueError("ledger record SHA-256 mismatch")
        previous = record["record_sha256"]
    expected_calls = {"sam", "vlm", "agent", "cvat", "gpu", "ssh", "oss",
                      "training", "deployment"}
    if set(snapshot["external_call_counts"]) != expected_calls \
            or any(snapshot["external_call_counts"].values()):
        raise ValueError("static orchestration ledger must record zero external calls")
    return True


def write_snapshot_exclusive(path, snapshot):
    """Write a new immutable snapshot; existing paths are never overwritten."""
    validate_ledger_snapshot(snapshot)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(snapshot, ensure_ascii=False, indent=2,
                                sort_keys=True) + "\n")
    if json.loads(path.read_text(encoding="utf-8")) != snapshot:
        raise RuntimeError("ledger snapshot readback mismatch")
