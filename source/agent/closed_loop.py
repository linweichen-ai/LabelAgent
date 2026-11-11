"""Append-only evidence ledger for the LabelAgent annotation closed loop."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re


SCHEMA_VERSION = "labelagent.closed-loop.v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_KINDS = {"real", "sim"}
LABEL_LAYERS = {"preannotation", "agent_candidate", "human_truth"}
STATES = {
    "received", "sam_candidate_ready", "vlm_reviewed", "agent_actioned",
    "vlm_reverified", "open_set_candidate", "human_review_required",
    "human_accepted", "human_rejected", "candidate_final",
    "truth_committed", "ontology_approved", "ontology_rejected",
    "quality_failed", "version_committed",
}
TRANSITIONS = {
    None: {"received"},
    "received": {"sam_candidate_ready", "human_review_required", "quality_failed"},
    "sam_candidate_ready": {"vlm_reviewed", "quality_failed"},
    "vlm_reviewed": {"agent_actioned", "open_set_candidate",
                     "human_review_required", "candidate_final", "quality_failed"},
    "agent_actioned": {"vlm_reverified", "quality_failed"},
    "vlm_reverified": {"agent_actioned", "open_set_candidate",
                       "human_review_required", "candidate_final", "quality_failed"},
    "open_set_candidate": {"human_review_required"},
    "quality_failed": {"human_review_required"},
    "human_review_required": {"human_accepted", "human_rejected"},
    "human_accepted": {"sam_candidate_ready", "candidate_final", "truth_committed", "ontology_approved"},
    "human_rejected": set(),
    "candidate_final": {"human_review_required", "truth_committed"},
    "truth_committed": {"version_committed"},
    "ontology_approved": {"candidate_final", "truth_committed"},
    "ontology_rejected": set(),
    "version_committed": set(),
}
HUMAN_ONLY = {"human_accepted", "human_rejected", "ontology_approved", "ontology_rejected"}


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


class ClosedLoopLedger:
    def __init__(self, root, envelope):
        self.root = Path(root).resolve()
        self.envelope = validate_input(envelope)
        self.input_id = input_id(self.envelope)
        self.run = self.root / "runs" / self.input_id
        self.events_dir = self.run / "events"
        self.receipts_dir = self.run / "receipts"

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
        _text(actor.get("id"), "actor.id")
        _text(action, "action")
        if state in HUMAN_ONLY and actor["type"] != "human":
            raise ValueError(f"{state} requires a human actor")
        if confidence is not None and (isinstance(confidence, bool) or
                not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1):
            raise ValueError("confidence must be null or between 0 and 1")
        payload, artifacts, metrics = payload or {}, artifacts or [], metrics or {}
        if state == "open_set_candidate":
            if label_layer == "human_truth":
                raise ValueError("open-set output cannot be human truth")
            validate_open_set(payload)
        history = self.events()
        event_key = digest({"state": state, "actor": actor, "action": action,
                            "label_layer": label_layer, "confidence": confidence,
                            "model": model, "prompt": prompt, "payload": payload,
                            "artifacts": artifacts, "metrics": metrics})
        duplicate = next((item for item in history if item.get("event_key") == event_key), None)
        if duplicate:
            return {**duplicate, "idempotent_replay": True}
        current = history[-1]["state"] if history else None
        if state not in TRANSITIONS[current]:
            raise ValueError(f"invalid transition: {current} -> {state}")
        if state == "truth_committed":
            if label_layer != "human_truth" or not any(item["state"] == "human_accepted" for item in history):
                raise ValueError("truth commit requires a prior human acceptance and human_truth layer")
        if state == "ontology_approved":
            if not any(item["state"] == "open_set_candidate" for item in history):
                raise ValueError("ontology approval requires an open-set candidate")
            for field in ("ontology_version", "parent_ontology_version"):
                _text(payload.get(field), field)
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
        accepted = "human_accepted" in states
        agent_path = all(required in states for required in
                         ("sam_candidate_ready", "vlm_reviewed",
                          "agent_actioned", "vlm_reverified"))
        minimum_complete = accepted and agent_path and states[-1] in {
            "candidate_final", "truth_committed", "version_committed"}
        receipt = {"schema_version": SCHEMA_VERSION, "input_id": self.input_id,
                   "content_sha256": self.envelope["content_sha256"],
                   "event_count": len(history), "last_state": states[-1] if states else None,
                   "last_event_id": history[-1]["event_id"] if history else None,
                   "human_gate_passed": accepted,
                   "full_agent_path_observed": agent_path,
                   "minimum_closed_loop_complete": minimum_complete,
                   "training_ready": bool(states and states[-1] == "version_committed"),
                   "layers_seen": sorted({item["label_layer"] for item in history}),
                   "as_of_event_time": history[-1]["occurred_at"] if history else None}
        receipt_hash = digest(receipt)
        receipt["receipt_sha256"] = receipt_hash
        name = f"{len(history):06d}-{receipt_hash}.json"
        path = self.receipts_dir / name
        if not path.exists():
            _exclusive_json(path, receipt)
        return receipt

