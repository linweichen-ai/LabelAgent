"""Append-only evidence ledger for the LabelAgent annotation closed loop."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

from .downstream_mapping import TRAVERSABLE_AREA_MAPPING


SCHEMA_VERSION = "labelagent.closed-loop.v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_KINDS = {"real", "sim"}
LABEL_LAYERS = {"preannotation", "agent_candidate", "human_truth"}
STATES = {
    "received", "quality_checked", "quality_failed", "sam_candidate_ready",
    "vlm_reviewed", "agent_actioned", "vlm_reverified",
    "open_set_candidate", "ontology_review_required", "ontology_approved",
    "ontology_rejected", "human_review_required", "human_accepted",
    "human_rejected", "candidate_rejected", "remediation_required",
    "candidate_final", "truth_committed", "version_committed",
    "evaluation_completed",
}
TRANSITIONS = {
    None: {"received"},
    "received": {"quality_checked", "human_review_required", "quality_failed"},
    "quality_checked": {"sam_candidate_ready", "human_review_required", "quality_failed"},
    "sam_candidate_ready": {"vlm_reviewed", "candidate_final", "quality_failed"},
    "vlm_reviewed": {"agent_actioned", "open_set_candidate",
                     "human_review_required", "candidate_final", "quality_failed"},
    "agent_actioned": {"vlm_reverified", "quality_failed"},
    "vlm_reverified": {"agent_actioned", "open_set_candidate",
                       "human_review_required", "candidate_final", "quality_failed"},
    "open_set_candidate": {"ontology_review_required"},
    "ontology_review_required": {"ontology_approved", "ontology_rejected"},
    "ontology_approved": {"remediation_required"},
    "ontology_rejected": {"remediation_required"},
    "quality_failed": {"human_review_required", "remediation_required"},
    "human_review_required": {"human_accepted", "human_rejected", "candidate_rejected"},
    "human_accepted": {"quality_checked", "candidate_final", "truth_committed"},
    "human_rejected": {"remediation_required"},
    "candidate_rejected": {"remediation_required"},
    "remediation_required": {"quality_checked", "human_review_required"},
    "candidate_final": {"human_review_required"},
    "truth_committed": {"version_committed"},
    "version_committed": {"evaluation_completed"},
    "evaluation_completed": set(),
}
HUMAN_ONLY = {"human_accepted", "human_rejected", "candidate_rejected",
              "ontology_approved", "ontology_rejected"}
MODEL_ACTORS = {"sam", "vlm", "agent"}
TRAVERSABILITY = {"passable", "non_passable"}
SLOPE_POLICY_DEFAULT = "passable"
SLOPE_POLICY_BASIS = "user_defined_project_semantics"
SLOPE_VISIBLE_STATUS = {"candidate_supported", "human_confirmed", "human_rejected", "unknown"}
SLOPE_HARD_SET_STRATA = {"unassigned", "near", "mid", "far", "weak_texture",
                         "shadow", "backlight", "occlusion",
                         "slope_foot_boundary", "slope_top_boundary"}
DATASET_SPLITS = {"train", "val", "test"}
ARTIFACT_FIELDS = {"artifact_id", "kind", "uri", "sha256", "bytes", "media_type"}
MODEL_FIELDS = {"name", "version", "weights_sha256", "config_sha256", "code_commit"}
PROMPT_FIELDS = {"version", "content_sha256"}
METRIC_FIELDS = {"end_to_end_latency_ms", "human_seconds", "human_minutes",
                 "human_seconds_per_frame", "human_minutes_per_frame",
                 "single_frame_cost", "estimated_cost", "agent_rounds",
                 "slope_iou", "slope_dice", "slope_boundary_f1",
                 "slope_boundary_tolerance_px", "slope_pixel_precision",
                 "slope_pixel_recall", "slope_instance_precision",
                 "slope_instance_recall", "slope_to_floor_confusion_pixels",
                 "floor_to_slope_confusion_pixels",
                 "slope_to_step_curb_confusion_pixels",
                 "step_curb_to_slope_confusion_pixels",
                 "slope_false_traversable_count"}
SLOPE_RATIO_METRICS = {"slope_iou", "slope_dice", "slope_boundary_f1",
                       "slope_pixel_precision", "slope_pixel_recall",
                       "slope_instance_precision", "slope_instance_recall"}
SLOPE_INTEGER_METRICS = {"slope_to_floor_confusion_pixels",
                         "floor_to_slope_confusion_pixels",
                         "slope_to_step_curb_confusion_pixels",
                         "step_curb_to_slope_confusion_pixels",
                         "slope_false_traversable_count"}
SLOPE_EVALUATION_METRICS = SLOPE_RATIO_METRICS | SLOPE_INTEGER_METRICS | {
    "slope_boundary_tolerance_px"}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical_json(value)).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _text(value, name, limit=300):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be a non-empty string no longer than {limit}")
    return value.strip()


def validate_input(envelope):
    if not isinstance(envelope, dict):
        raise ValueError("input envelope must be an object")
    required = ("dataset_id", "source_batch", "frame_id", "sensor",
                "captured_at", "source_kind", "content_sha256")
    unknown = set(envelope) - set(required) - {"source_uri", "timestamp_source"}
    if unknown:
        raise ValueError(f"unknown input fields: {sorted(unknown)}")
    normalized = {name: _text(envelope.get(name), name) for name in required[:-2]}
    normalized["source_kind"] = envelope.get("source_kind")
    normalized["content_sha256"] = envelope.get("content_sha256")
    if normalized["source_kind"] not in SOURCE_KINDS:
        raise ValueError("source_kind must be real or sim")
    if not isinstance(normalized["content_sha256"], str) or not HEX64.fullmatch(normalized["content_sha256"]):
        raise ValueError("content_sha256 must be a lowercase SHA-256 hex digest")
    try:
        timestamp = datetime.fromisoformat(normalized["captured_at"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("captured_at must be an ISO-8601 timestamp") from error
    if timestamp.tzinfo is None:
        raise ValueError("captured_at must include a timezone")
    for optional in ("source_uri", "timestamp_source"):
        if optional in envelope:
            normalized[optional] = _text(envelope[optional], optional, 2000)
    return normalized


def input_id(envelope):
    envelope = validate_input(envelope)
    identity = {key: envelope[key] for key in
                ("dataset_id", "source_batch", "frame_id", "sensor",
                 "captured_at", "source_kind")}
    return digest(identity)


def _exclusive_json(path, value):
    path = Path(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
    except Exception:
        path.unlink(missing_ok=True)
        raise


def validate_open_set(payload):
    candidate = payload.get("open_set") if isinstance(payload, dict) else None
    if not isinstance(candidate, dict):
        raise ValueError("open_set_candidate requires payload.open_set")
    required = {"status", "visual_evidence", "candidate_name",
                "nearest_existing_class", "difference", "confidence"}
    if set(candidate) - required - {"ontology_write"} or not required <= set(candidate):
        raise ValueError("open_set payload fields are incomplete or unknown")
    if candidate["status"] not in {"unknown", "new_label_candidate"}:
        raise ValueError("open-set status must remain unknown or new_label_candidate")
    if not isinstance(candidate["visual_evidence"], list) or not candidate["visual_evidence"]:
        raise ValueError("open-set visual evidence is required")
    if not all(isinstance(item, str) and item.strip() for item in candidate["visual_evidence"]):
        raise ValueError("open-set visual evidence entries must be non-empty strings")
    for name in ("candidate_name", "nearest_existing_class", "difference"):
        _text(candidate[name], f"open_set.{name}", 500)
    confidence = candidate["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("open-set confidence must be between 0 and 1")
    if candidate.get("ontology_write") is not False:
        raise ValueError("open-set candidates cannot write the production ontology")


def validate_artifacts(artifacts):
    if not isinstance(artifacts, list):
        raise ValueError("artifacts must be a list")
    identifiers = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise ValueError(f"artifacts[{index}] must be an object")
        unknown = set(artifact) - ARTIFACT_FIELDS
        if unknown:
            raise ValueError(f"artifacts[{index}] has unknown fields: {sorted(unknown)}")
        _text(artifact.get("kind"), f"artifacts[{index}].kind")
        sha256 = artifact.get("sha256")
        if not isinstance(sha256, str) or not HEX64.fullmatch(sha256):
            raise ValueError(f"artifacts[{index}].sha256 must be a lowercase SHA-256 digest")
        artifact_id = artifact.get("artifact_id")
        if artifact_id is not None:
            artifact_id = _text(artifact_id, f"artifacts[{index}].artifact_id")
            if artifact_id in identifiers:
                raise ValueError(f"duplicate artifact_id: {artifact_id}")
            identifiers.add(artifact_id)
        if "uri" in artifact:
            _text(artifact["uri"], f"artifacts[{index}].uri", 2000)
        if "bytes" in artifact and (isinstance(artifact["bytes"], bool) or
                not isinstance(artifact["bytes"], int) or artifact["bytes"] < 0):
            raise ValueError(f"artifacts[{index}].bytes must be a non-negative integer")
    return identifiers


def validate_model_prompt(actor_type, model, prompt):
    if actor_type in MODEL_ACTORS and not isinstance(model, dict):
        raise ValueError("model-driven events require model metadata")
    if model is not None:
        if not isinstance(model, dict) or set(model) - MODEL_FIELDS:
            raise ValueError("model metadata contains unknown fields")
        _text(model.get("name"), "model.name")
        _text(model.get("version"), "model.version", 1000)
        for field in ("weights_sha256", "config_sha256"):
            if field in model and (not isinstance(model[field], str) or
                    not HEX64.fullmatch(model[field])):
                raise ValueError(f"model.{field} must be a lowercase SHA-256 digest")
        if "code_commit" in model:
            _text(model["code_commit"], "model.code_commit")
    if actor_type in MODEL_ACTORS and not isinstance(prompt, dict):
        raise ValueError("model-driven events require prompt metadata")
    if prompt is not None:
        if not isinstance(prompt, dict) or set(prompt) - PROMPT_FIELDS:
            raise ValueError("prompt metadata contains unknown fields")
        _text(prompt.get("version"), "prompt.version")
        if "content_sha256" in prompt and (not isinstance(prompt["content_sha256"], str) or
                not HEX64.fullmatch(prompt["content_sha256"])):
            raise ValueError("prompt.content_sha256 must be a lowercase SHA-256 digest")


def validate_metrics(metrics):
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be an object")
    unknown = set(metrics) - METRIC_FIELDS
    if unknown:
        raise ValueError(f"metrics contains unknown fields: {sorted(unknown)}")
    for name, value in metrics.items():
        if value is None:
            continue
        if name == "agent_rounds" or name in SLOPE_INTEGER_METRICS:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"metrics.{name} must be a non-negative integer or null")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"metrics.{name} must be a non-negative number or null")
        if name in SLOPE_RATIO_METRICS and value > 1:
            raise ValueError(f"metrics.{name} must be between 0 and 1")


def validate_quality(state, payload):
    if state != "quality_checked":
        return
    quality = payload.get("quality_checked") if isinstance(payload, dict) else None
    if not isinstance(quality, dict) or quality.get("passed") is not True:
        raise ValueError("quality_checked requires payload.quality_checked.passed=true")
    if set(quality) != {"passed", "checks"}:
        raise ValueError("payload.quality_checked fields are incomplete or unknown")
    checks = quality.get("checks")
    if not isinstance(checks, list) or not checks or not all(
            isinstance(item, str) and item.strip() for item in checks):
        raise ValueError("quality_checked requires a non-empty checks list")


def validate_slope(state, payload, artifacts, metrics, actor_type=None):
    if not isinstance(payload, dict):
        return
    task = payload.get("task_context")
    if task is not None and not isinstance(task, dict):
        raise ValueError("payload.task_context must be an object")
    tags = task.get("hard_case_tags", []) if isinstance(task, dict) else []
    if not isinstance(tags, list) or not all(isinstance(item, str) for item in tags):
        raise ValueError("task_context.hard_case_tags must be a string array")
    if "slope_context" in payload and "slope" not in tags:
        raise ValueError("payload.slope_context requires the slope hard-case tag")
    if "slope" not in tags:
        return
    slope = payload.get("slope_context")
    if not isinstance(slope, dict):
        raise ValueError("slope hard cases require payload.slope_context")
    required = {"semantic_label", "visible_status", "human_pixel_truth_available",
                "approved_for_training", "traversability_status",
                "business_mapping", "downstream_mapping", "policy_default",
                "policy_basis", "policy_override",
                "candidate_sha256", "evidence_provenance", "hard_set_strata"}
    if set(slope) != required:
        raise ValueError("slope_context fields are incomplete")
    if slope["semantic_label"] != "slope_visible":
        raise ValueError("slope_context.semantic_label must be slope_visible")
    if slope["visible_status"] not in SLOPE_VISIBLE_STATUS:
        raise ValueError("slope_context.visible_status is invalid")
    if not isinstance(slope["human_pixel_truth_available"], bool):
        raise ValueError("slope_context.human_pixel_truth_available must be boolean")
    if not isinstance(slope["approved_for_training"], bool):
        raise ValueError("slope_context.approved_for_training must be boolean")
    status = slope["traversability_status"]
    if status not in TRAVERSABILITY:
        raise ValueError("slope_context.traversability_status is invalid")
    if slope["business_mapping"] != "traversable_area":
        raise ValueError("slope_context.business_mapping must be traversable_area")
    if slope["downstream_mapping"] != TRAVERSABLE_AREA_MAPPING:
        raise ValueError("slope_context.downstream_mapping must preserve floor/slope and union them downstream")
    if slope["policy_default"] != SLOPE_POLICY_DEFAULT:
        raise ValueError("slope_context.policy_default must be passable")
    if slope["policy_basis"] != SLOPE_POLICY_BASIS:
        raise ValueError("slope_context.policy_basis must be user_defined_project_semantics")
    override = slope["policy_override"]
    override_fields = {"status", "rule_id", "reason", "decided_by",
                       "evidence_artifact_ids"}
    if not isinstance(override, dict) or set(override) != override_fields:
        raise ValueError("slope_context.policy_override fields are incomplete")
    override_status = override["status"]
    if override_status not in {"none", "non_passable"}:
        raise ValueError("slope_context.policy_override.status is invalid")
    evidence_ids = override["evidence_artifact_ids"]
    if (not isinstance(evidence_ids, list)
            or not all(isinstance(item, str) and item.strip() for item in evidence_ids)
            or len(evidence_ids) != len(set(evidence_ids))):
        raise ValueError("slope policy override evidence IDs must be a unique string array")
    if override_status == "none":
        if (status != "passable" or override["rule_id"] is not None
                or override["reason"] is not None or override["decided_by"] is not None
                or evidence_ids):
            raise ValueError("default slope policy must remain passable without override metadata")
    else:
        if status != "non_passable":
            raise ValueError("non-passable slope override must set traversability_status=non_passable")
        _text(override["rule_id"], "slope policy override rule_id")
        _text(override["reason"], "slope policy override reason")
        if override["decided_by"] not in {"human", "system_policy"}:
            raise ValueError("slope policy override must be decided by human or system_policy")
        if actor_type in MODEL_ACTORS:
            raise ValueError("models cannot create a non-passable slope policy override")
    candidate_sha = slope["candidate_sha256"]
    if candidate_sha is not None and not HEX64.fullmatch(candidate_sha):
        raise ValueError("slope_context.candidate_sha256 must be lowercase SHA-256 or null")
    provenance = slope["evidence_provenance"]
    if not isinstance(provenance, list) or not provenance:
        raise ValueError("slope_context.evidence_provenance must be a non-empty array")
    for item in provenance:
        if not isinstance(item, dict) or set(item) != {"kind", "source", "sha256"}:
            raise ValueError("slope evidence provenance fields are incomplete or unknown")
        _text(item.get("kind"), "slope evidence provenance kind")
        _text(item.get("source"), "slope evidence provenance source")
        if not HEX64.fullmatch(str(item.get("sha256", ""))):
            raise ValueError("slope evidence provenance SHA-256 is invalid")
    strata = slope["hard_set_strata"]
    if (not isinstance(strata, list) or not strata
            or len(strata) != len(set(strata))
            or not set(strata) <= SLOPE_HARD_SET_STRATA):
        raise ValueError("slope_context.hard_set_strata is invalid")
    by_id = {artifact.get("artifact_id"): artifact for artifact in artifacts
             if isinstance(artifact, dict) and artifact.get("artifact_id")}
    missing = sorted(set(evidence_ids) - set(by_id))
    if missing:
        raise ValueError("slope policy override references missing artifacts: " + ", ".join(missing))
    if slope["approved_for_training"] and not slope["human_pixel_truth_available"]:
        raise ValueError("slope cannot be approved for training without human pixel truth")
    if state == "evaluation_completed" and not slope["human_pixel_truth_available"]:
        raise ValueError("slope evaluation requires human pixel truth")
    if state == "evaluation_completed":
        missing_metrics = sorted(SLOPE_EVALUATION_METRICS - set(metrics))
        if missing_metrics or any(metrics.get(name) is None for name in SLOPE_EVALUATION_METRICS):
            raise ValueError("slope evaluation requires complete slope-specific metrics: "
                             + ", ".join(missing_metrics or sorted(SLOPE_EVALUATION_METRICS)))
        if "unassigned" in strata:
            raise ValueError("slope evaluation requires assigned hard-set strata")


def validate_version(state, payload, artifacts, source_kind):
    if state != "version_committed":
        return
    version = payload.get("dataset_version") if isinstance(payload, dict) else None
    if not isinstance(version, dict):
        raise ValueError("version_committed requires payload.dataset_version")
    required = {"version_id", "parent_version", "split", "split_group_id",
                "ontology_version", "source_kind", "truth_sha256", "manifest_sha256"}
    if set(version) != required:
        raise ValueError("payload.dataset_version fields are incomplete or unknown")
    for field in ("version_id", "split_group_id", "ontology_version"):
        _text(version.get(field), f"dataset_version.{field}")
    parent = version.get("parent_version")
    if parent is not None:
        _text(parent, "dataset_version.parent_version")
        if parent == version["version_id"]:
            raise ValueError("dataset_version.parent_version must differ from version_id")
    if version.get("split") not in DATASET_SPLITS:
        raise ValueError("dataset_version.split must be train, val, or test")
    if version.get("source_kind") != source_kind:
        raise ValueError("dataset_version.source_kind must match the immutable input")
    for field in ("truth_sha256", "manifest_sha256"):
        value = version.get(field)
        if not isinstance(value, str) or not HEX64.fullmatch(value):
            raise ValueError(f"dataset_version.{field} must be a lowercase SHA-256 digest")
    if not any(item.get("kind") == "dataset_manifest" and
               item.get("sha256") == version["manifest_sha256"] for item in artifacts):
        raise ValueError("dataset version manifest SHA-256 must reference an event artifact")


def accepted_candidate(history, acceptance_index):
    """Return the exact candidate_final event bound to one human acceptance."""
    acceptance = history[acceptance_index]
    payload = acceptance.get("payload", {})
    event_id = payload.get("accepted_event_id")
    artifact_sha = payload.get("accepted_artifact_sha256")
    if not isinstance(event_id, str) or not HEX64.fullmatch(event_id):
        raise ValueError("human acceptance requires accepted_event_id")
    if not isinstance(artifact_sha, str) or not HEX64.fullmatch(artifact_sha):
        raise ValueError("human acceptance requires accepted_artifact_sha256")
    candidates = [(index, event) for index, event in enumerate(history[:acceptance_index])
                  if event.get("state") == "candidate_final"]
    if not candidates or candidates[-1][1].get("event_id") != event_id:
        raise ValueError("human acceptance must bind the latest candidate_final event")
    candidate_index, candidate = candidates[-1]
    if not any(item.get("kind") == "candidate_final" and
               item.get("sha256") == artifact_sha
               for item in candidate.get("artifacts", [])):
        raise ValueError("accepted candidate artifact SHA-256 does not match candidate_final")
    return candidate_index, candidate, artifact_sha


def linked_truth(history, truth_index):
    """Validate immediate human acceptance and exact candidate/truth linkage."""
    if truth_index < 1 or history[truth_index - 1].get("state") != "human_accepted":
        raise ValueError("truth commit requires an immediately preceding human acceptance")
    _, candidate, accepted_sha = accepted_candidate(history, truth_index - 1)
    truth = history[truth_index]
    truth_sha = truth.get("payload", {}).get("truth_sha256")
    if truth_sha != accepted_sha or not any(
            item.get("kind") == "truth" and item.get("sha256") == accepted_sha
            for item in truth.get("artifacts", [])):
        raise ValueError("truth artifact must exactly match the human-accepted candidate artifact")
    return candidate


class ClosedLoopLedger:
    def __init__(self, root, envelope):
        self.root = Path(root).resolve()
        self.envelope = validate_input(envelope)
        self.input_id = input_id(self.envelope)
        self.run = self.root / "runs" / self.input_id
        self.events_dir = self.run / "events"
        self.receipts_dir = self.run / "receipts"

    def _committed_versions(self):
        records = []
        runs = self.root / "runs"
        if not runs.exists():
            return records
        for run in runs.iterdir():
            source = run / "input.json"
            if not source.is_file():
                continue
            envelope = json.loads(source.read_text(encoding="utf-8"))
            if envelope.get("dataset_id") != self.envelope["dataset_id"]:
                continue
            for path in sorted((run / "events").glob("*.json")):
                event = json.loads(path.read_text(encoding="utf-8"))
                event_id = event.pop("event_id", None)
                if digest(event) != event_id:
                    raise RuntimeError(f"version registry event hash is invalid at {path}")
                if event.get("state") == "version_committed":
                    version = event.get("payload", {}).get("dataset_version")
                    if isinstance(version, dict):
                        records.append(version)
        return records

    def _validate_version_lineage(self, version):
        records = self._committed_versions()
        parent = version["parent_version"]
        if parent is not None and not any(item.get("version_id") == parent
                                          for item in records):
            raise ValueError("dataset_version.parent_version is not present in the local immutable registry")
        for item in records:
            if (item.get("split_group_id") == version["split_group_id"] and
                    item.get("split") != version["split"]):
                raise ValueError("train/val/test isolation violation for split_group_id")
            if item.get("version_id") == version["version_id"]:
                for field in ("parent_version", "ontology_version", "source_kind"):
                    if item.get(field) != version[field]:
                        raise ValueError(f"dataset version {field} conflicts with the immutable registry")

    def register(self):
        (self.root / "runs").mkdir(parents=True, exist_ok=True)
        try:
            self.run.mkdir()
            created = True
        except FileExistsError:
            created = False
        self.events_dir.mkdir(exist_ok=True)
        self.receipts_dir.mkdir(exist_ok=True)
        source = self.run / "input.json"
        if created:
            _exclusive_json(source, {"schema_version": SCHEMA_VERSION,
                                     "input_id": self.input_id, **self.envelope})
        else:
            if not source.is_file():
                raise RuntimeError("existing run is missing immutable input.json")
            previous = json.loads(source.read_text(encoding="utf-8"))
            if previous.get("content_sha256") != self.envelope["content_sha256"]:
                raise ValueError("idempotency conflict: the same frame identity has different content")
            comparable = {key: previous.get(key) for key in self.envelope}
            if comparable != self.envelope:
                raise ValueError("idempotency conflict: immutable input metadata changed")
        if not self.events():
            self.append_event("received", actor={"type": "system", "id": "ingest"},
                              action="register_input", label_layer="preannotation",
                              confidence=None, payload={"idempotent_replay": not created})
        return {"input_id": self.input_id, "created": created,
                "idempotent_replay": not created}

    def events(self):
        result = []
        previous = None
        for path in sorted(self.events_dir.glob("*.json")):
            event = json.loads(path.read_text(encoding="utf-8"))
            event_id = event.pop("event_id", None)
            if event.get("previous_event_id") != previous or digest(event) != event_id:
                raise RuntimeError(f"event hash chain is invalid at {path.name}")
            event["event_id"] = event_id
            result.append(event)
            previous = event_id
        return result

    def append_event(self, state, *, actor, action, label_layer,
                     confidence=None, model=None, prompt=None, payload=None,
                     artifacts=None, metrics=None):
        if state not in STATES:
            raise ValueError("unknown closed-loop state")
        if label_layer not in LABEL_LAYERS:
            raise ValueError("unknown label layer")
        if not isinstance(actor, dict) or actor.get("type") not in {"system", "sam", "vlm", "agent", "human", "quality"}:
            raise ValueError("actor.type is invalid")
        if set(actor) != {"type", "id"}:
            raise ValueError("actor must contain only type and id")
        _text(actor.get("id"), "actor.id")
        _text(action, "action")
        if state in HUMAN_ONLY and actor["type"] != "human":
            raise ValueError(f"{state} requires a human actor")
        if confidence is not None and (isinstance(confidence, bool) or
                not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1):
            raise ValueError("confidence must be null or between 0 and 1")
        payload, artifacts, metrics = payload or {}, artifacts or [], metrics or {}
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        validate_artifacts(artifacts)
        validate_metrics(metrics)
        validate_model_prompt(actor["type"], model, prompt)
        validate_quality(state, payload)
        validate_slope(state, payload, artifacts, metrics, actor["type"])
        validate_version(state, payload, artifacts, self.envelope["source_kind"])
        if state == "open_set_candidate":
            if label_layer == "human_truth":
                raise ValueError("open-set output cannot be human truth")
            validate_open_set(payload)
        history = self.events()
        immutable = {"state": state, "actor": actor, "action": action,
                     "label_layer": label_layer, "confidence": confidence,
                     "model": model, "prompt": prompt, "payload": payload,
                     "artifacts": artifacts, "metrics": metrics}
        event_key = digest(immutable)
        duplicate = next((item for item in history if item.get("event_key") == event_key), None)
        if duplicate:
            if any(duplicate.get(key) != value for key, value in immutable.items()):
                raise RuntimeError("stored event_key does not match immutable event fields")
            return {**duplicate, "idempotent_replay": True}
        current = history[-1]["state"] if history else None
        if state not in TRANSITIONS[current]:
            raise ValueError(f"invalid transition: {current} -> {state}")
        if state == "human_accepted":
            has_candidate = any(item["state"] == "candidate_final" for item in history)
            has_link = bool({"accepted_event_id", "accepted_artifact_sha256"} & set(payload))
            if has_candidate or has_link:
                accepted_candidate(history + [{"state": state, "payload": payload}],
                                   len(history))
        if state == "truth_committed":
            if label_layer != "human_truth":
                raise ValueError("truth commit requires the human_truth layer")
            linked_truth(history + [{"state": state, "payload": payload,
                                     "artifacts": artifacts}], len(history))
        if state == "version_committed":
            self._validate_version_lineage(payload["dataset_version"])
        if state == "ontology_approved":
            if not any(item["state"] == "open_set_candidate" for item in history):
                raise ValueError("ontology approval requires an open-set candidate")
            for field in ("ontology_version", "parent_ontology_version"):
                _text(payload.get(field), field)
        if state == "ontology_rejected" and not any(
                item["state"] == "open_set_candidate" for item in history):
            raise ValueError("ontology rejection requires an open-set candidate")
        body = {"schema_version": SCHEMA_VERSION, "input_id": self.input_id,
                "sequence": len(history) + 1, "occurred_at": utc_now(),
                "previous_event_id": history[-1]["event_id"] if history else None,
                "event_key": event_key, "state": state, "actor": actor,
                "action": action, "label_layer": label_layer,
                "confidence": confidence, "model": model, "prompt": prompt,
                "payload": payload, "artifacts": artifacts, "metrics": metrics}
        event_id = digest(body)
        event = {**body, "event_id": event_id}
        _exclusive_json(self.events_dir / f"{body['sequence']:06d}-{event_id}.json", event)
        return event

    def receipt(self):
        history = self.events()
        states = [item["state"] for item in history]
        linked = []
        for index, event in enumerate(history):
            if event["state"] != "truth_committed":
                continue
            try:
                candidate = linked_truth(history, index)
            except ValueError:
                continue
            linked.append((index, candidate))
        accepted = bool(linked)
        quality_observed = "quality_checked" in states
        agent_path = False
        if linked:
            truth_index, candidate = linked[-1]
            candidate_index = next(index for index, event in enumerate(history)
                                   if event["event_id"] == candidate["event_id"])
            cursor = -1
            ordered = True
            for required in ("quality_checked", "sam_candidate_ready", "vlm_reviewed",
                             "agent_actioned", "vlm_reverified"):
                position = next((index for index in range(cursor + 1, candidate_index)
                                 if history[index]["state"] == required), None)
                if position is None:
                    ordered = False
                    break
                cursor = position
            agent_path = ordered
        minimum_complete = accepted and quality_observed and agent_path
        version_event = None
        if linked:
            truth_index, _ = linked[-1]
            if truth_index + 1 < len(history) and history[truth_index + 1]["state"] == "version_committed":
                candidate_version = history[truth_index + 1]
                try:
                    validate_version("version_committed", candidate_version["payload"],
                                     candidate_version["artifacts"],
                                     self.envelope["source_kind"])
                    self._validate_version_lineage(candidate_version["payload"]["dataset_version"])
                except ValueError:
                    candidate_version = None
                version_event = candidate_version
        version = version_event.get("payload", {}).get("dataset_version", {}) if version_event else {}
        receipt = {"schema_version": SCHEMA_VERSION, "input_id": self.input_id,
                   "content_sha256": self.envelope["content_sha256"],
                   "event_count": len(history), "last_state": states[-1] if states else None,
                   "last_event_id": history[-1]["event_id"] if history else None,
                   "human_gate_passed": accepted,
                   "quality_gate_observed": quality_observed,
                   "full_agent_path_observed": agent_path,
                   "minimum_closed_loop_complete": minimum_complete,
                   "training_ready": version_event is not None,
                   "dataset_version": version.get("version_id"),
                   "dataset_split": version.get("split"),
                   "layers_seen": sorted({item["label_layer"] for item in history}),
                   "as_of_event_time": history[-1]["occurred_at"] if history else None}
        receipt_hash = digest(receipt)
        receipt["receipt_sha256"] = receipt_hash
        for existing_path in sorted(self.receipts_dir.glob("*.json")):
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            if existing.get("last_event_id") == receipt["last_event_id"]:
                if existing != receipt:
                    raise RuntimeError("receipt conflict for the same last_event_id")
                return existing
        name = f"{len(history):06d}-{receipt_hash}.json"
        path = self.receipts_dir / name
        if not path.exists():
            _exclusive_json(path, receipt)
        return receipt

