"""Register one workbench frame and emit immutable closed-loop snapshots."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.agent.closed_loop import (ClosedLoopLedger, SCHEMA_VERSION,
                                      canonical_json, digest)
from source.agent.downstream_mapping import mapping_contract


ACCEPTED_REVIEWS = {"accept_candidate", "correct_candidate", "manual_annotation"}


def request(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=60) as response:
        return response.read(), response.headers.get_content_type()


def request_json(url):
    data, _ = request(url)
    return json.loads(data)


def captured_time(path, topic, frame):
    # Keep MCAP optional for pure state-machine/schema tests.
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory
    from source.agent.sensors import message_time

    factory = DecoderFactory()
    with Path(path).open("rb") as stream:
        messages = make_reader(stream).iter_messages(topics=[topic])
        for index, (schema, channel, message) in enumerate(messages):
            if index != frame:
                continue
            decoder = factory.decoder_for(channel.message_encoding, schema)
            decoded = decoder(message.data) if decoder else message.data
            timestamp = message_time(decoded, message)
            source = "capture" if timestamp else "log_time"
            timestamp = timestamp or message.log_time
            value = datetime.fromtimestamp(timestamp / 1_000_000_000,
                                           timezone.utc).isoformat().replace("+00:00", "Z")
            return value, source
    raise ValueError("frame was not found in the selected MCAP topic")


def encoded_json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) +
            "\n").encode("utf-8")


def write_bytes_once(path, encoded):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"immutable artifact already exists with different content: {path}")
        return False
    path.write_bytes(encoded)
    return True


def write_snapshot(path, value):
    """Write the first snapshot at path and later states to stable siblings."""
    path = Path(path)
    encoded = encoded_json(value)
    if not path.exists() or path.read_bytes() == encoded:
        write_bytes_once(path, encoded)
        return path
    last_event = value["receipt"]["last_event_id"]
    snapshot = path.with_name(f"{path.stem}-{last_event[:16]}{path.suffix}")
    write_bytes_once(snapshot, encoded)
    return snapshot


def _frame_layer(state, topic, singular, plural):
    if state.get("topic") == topic:
        return state.get(singular, {})
    return state.get(plural, {}).get(topic, {})


def _matching_review(reviews, *, frame, topic, frame_sha256, truth_sha256):
    matching = []
    for review in reviews or []:
        if (review.get("decision") not in ACCEPTED_REVIEWS or
                int(review.get("frame", -1)) != frame or
                review.get("topic") != topic or
                review.get("frame_sha256") != frame_sha256 or
                review.get("truth_sha256") != truth_sha256):
            continue
        review_id = review.get("review_id")
        body = {key: value for key, value in review.items() if key != "review_id"}
        if not isinstance(review_id, str) or digest(body) != review_id:
            raise ValueError("human review receipt hash is invalid")
        matching.append(review)
    return matching[-1] if matching else None


def _candidate_evidence(state, topic, frame, review):
    candidates = _frame_layer(state, topic, "candidates", "candidates_by_topic")
    shapes = candidates.get(str(frame))
    if shapes is not None:
        return digest(shapes), len(shapes), "workbench_machine_candidate"
    if review and review.get("candidate_sha256"):
        return (review["candidate_sha256"],
                int(review.get("candidate_shape_count") or 0),
                "human_review_receipt")
    return None, 0, None


def _contains_slope(shapes):
    return bool(shapes) and any(isinstance(item, dict) and
                                item.get("label") in {"slope", "slope_visible"}
                                for item in shapes)


def _slope_context(*, human_truth, approved, candidate_sha, provenance_source):
    return {"task_context": {"hard_case_tags": ["slope"]},
            "slope_context": {"semantic_label": "slope_visible",
                "visible_status": "human_confirmed" if human_truth else "candidate_supported",
                "human_pixel_truth_available": human_truth,
                "approved_for_training": approved,
                "business_mapping": "traversable_area",
                "downstream_mapping": mapping_contract(),
                "traversability_status": "passable",
                "policy_default": "passable",
                "policy_basis": "user_defined_project_semantics",
                "policy_override": {"status": "none", "rule_id": None,
                    "reason": None, "decided_by": None,
                    "evidence_artifact_ids": []},
                "candidate_sha256": candidate_sha,
                "evidence_provenance": [{"kind": "candidate_hash",
                    "source": provenance_source, "sha256": candidate_sha}],
                "hard_set_strata": ["unassigned"]}}


def _history_contains_slope(history):
    return any("slope" in event.get("payload", {}).get("task_context", {}).get(
        "hard_case_tags", []) for event in history)


def _existing_candidate_event(history):
    return next((event for event in reversed(history)
                 if event["state"] == "candidate_final"), None)


def _existing_review_id(history):
    for event in reversed(history):
        review_id = event.get("payload", {}).get("review_id")
        if review_id:
            return review_id
    return None


def _stable_registration(ledger):
    received = ledger.events()[0]
    replay = bool(received.get("payload", {}).get("idempotent_replay", False))
    return {"input_id": ledger.input_id, "created": not replay,
            "idempotent_replay": replay}


def process_vertical(args, *, state, image, media_type, timestamp,
                     timestamp_source, reviews=None):
    """Advance one immutable frame as far as current workbench evidence allows."""
    if not state.get("loaded") or args.topic not in state.get("topics", {}):
        raise ValueError("the requested dataset/topic is not loaded")
    topic_count = state["topics"][args.topic]["count"]
    if not 0 <= args.frame < topic_count:
        raise ValueError("frame is outside the selected topic")
    if not image or not isinstance(media_type, str) or not media_type.startswith("image/"):
        raise ValueError("workbench frame must be non-empty image content")
    image_hash = hashlib.sha256(image).hexdigest()
    envelope = {"dataset_id": args.dataset_id, "source_batch": args.source_batch,
                "frame_id": str(args.frame), "sensor": args.topic,
                "captured_at": timestamp, "source_kind": args.source_kind,
                "content_sha256": image_hash, "source_uri": state["path"],
                "timestamp_source": timestamp_source}
    ledger = ClosedLoopLedger(args.ledger_root, envelope)
    ledger.register()

    truth_frames = _frame_layer(state, args.topic, "truth", "truths_by_topic")
    truth = truth_frames.get(str(args.frame))
    truth_sha = digest(truth) if truth is not None else None
    references = state.get("reference_frames", {}).get(args.topic, [])
    is_reference = args.frame in references
    review = _matching_review(reviews, frame=args.frame, topic=args.topic,
                              frame_sha256=image_hash, truth_sha256=truth_sha)
    candidate_sha, candidate_count, candidate_source = _candidate_evidence(
        state, args.topic, args.frame, review)
    candidate_shapes = _frame_layer(state, args.topic, "candidates",
                                    "candidates_by_topic").get(str(args.frame))
    slope_case = _contains_slope(candidate_shapes) or _contains_slope(truth)

    current = ledger.events()[-1]["state"]
    slope_case = slope_case or _history_contains_slope(ledger.events())
    review_id = review["review_id"] if review else _existing_review_id(ledger.events())
    if current == "received":
        ledger.append_event("quality_checked",
            actor={"type": "quality", "id": "vertical-input-qc"},
            action="validate_frame_input", label_layer="preannotation",
            payload={"quality_checked": {"passed": True,
                "checks": ["nonempty_image_bytes", "image_media_type",
                           "topic_membership", "frame_bounds", "content_hash"]}},
            artifacts=[{"kind": "input_frame", "sha256": image_hash,
                        "bytes": len(image), "media_type": media_type}])
        current = "quality_checked"

    if current == "quality_checked" and candidate_sha:
        candidate_payload = {"candidate_ingestion": {
            "source": candidate_source, "candidate_sha256": candidate_sha,
            "shape_count": candidate_count, "saved_as_truth": False}}
        if slope_case:
            candidate_payload.update(_slope_context(human_truth=False, approved=False,
                candidate_sha=candidate_sha, provenance_source=candidate_source))
        ledger.append_event("sam_candidate_ready",
            actor={"type": "sam", "id": "workbench-candidate-ingest"},
            action="ingest_existing_machine_candidate", label_layer="preannotation",
            model={"name": args.sam_model_name, "version": args.sam_model_version},
            prompt={"version": args.sam_prompt_version},
            payload=candidate_payload,
            artifacts=[{"kind": "machine_candidate", "sha256": candidate_sha}])
        current = "sam_candidate_ready"

    if current == "sam_candidate_ready":
        frozen_payload = {"candidate_sha256": candidate_sha,
                          "source": "workbench_machine_candidate"}
        if slope_case:
            frozen_payload.update(_slope_context(human_truth=False, approved=False,
                candidate_sha=candidate_sha, provenance_source="candidate_final"))
        ledger.append_event("candidate_final",
            actor={"type": "system", "id": "vertical-candidate-freezer"},
            action="freeze_machine_candidate_for_review", label_layer="agent_candidate",
            payload=frozen_payload,
            artifacts=[{"kind": "candidate_final", "sha256": candidate_sha}])
        current = "candidate_final"

    if current in {"quality_checked", "candidate_final"}:
        reasons = []
        if truth is None:
            reasons.append("missing_human_truth")
        if not is_reference:
            reasons.append("not_a_reference_frame")
        if review is None:
            reasons.append("missing_human_review_receipt")
        ledger.append_event("human_review_required",
            actor={"type": "system", "id": "vertical-frame-gate"},
            action="route_candidate_to_human_gate", label_layer="preannotation",
            payload={"reasons": reasons, "routing": {"novel": False,
                "low_confidence": False, "model_conflict": False,
                "quality_failure": False, "missing_human_gate": bool(reasons)},
                "production_truth_write": False})
        current = "human_review_required"

    if current == "human_review_required" and truth is not None and is_reference and review:
        slope_case = slope_case or _history_contains_slope(ledger.events())
        candidate_event = _existing_candidate_event(ledger.events())
        recorded_candidate = (candidate_event.get("payload", {}).get("candidate_sha256")
                              if candidate_event else None)
        if candidate_event is None or recorded_candidate != review.get("candidate_sha256"):
            raise ValueError("human review candidate hash does not match the recorded machine candidate")
        if truth_sha != recorded_candidate:
            raise ValueError("human truth must exactly match candidate_final before acceptance")
        accepted_payload = {"review_id": review["review_id"],
                            "reviewed_at": review["reviewed_at"],
                            "truth_sha256": truth_sha,
                            "candidate_sha256": review.get("candidate_sha256"),
                            "accepted_event_id": candidate_event["event_id"],
                            "accepted_artifact_sha256": recorded_candidate}
        if slope_case:
            accepted_payload.update(_slope_context(human_truth=True, approved=False,
                candidate_sha=recorded_candidate, provenance_source="human_review_receipt"))
        if (not review.get("reviewer_declared_id")
                or review.get("identity_assurance") != "self_declared"):
            raise ValueError("human review identity must be explicitly self-declared")
        ledger.append_event("human_accepted",
            actor={"type": "human", "id": review["reviewer_declared_id"]},
            action=review["decision"], label_layer="human_truth",
            payload=accepted_payload,
            artifacts=[{"kind": "human_review_receipt", "sha256": review["review_id"]}],
            metrics={"human_seconds_per_frame": review["elapsed_seconds"],
                     "human_minutes_per_frame": round(review["elapsed_seconds"] / 60, 6)})
        current = "human_accepted"
        review_id = review["review_id"]

    if current == "human_accepted":
        if truth is None or not is_reference or not review_id:
            raise ValueError("accepted human gate cannot advance without current truth, reference, and review")
        truth_payload = {"review_id": review_id, "truth_sha256": truth_sha,
                         "truth_shape_count": len(truth)}
        if slope_case:
            truth_payload.update(_slope_context(human_truth=True, approved=False,
                candidate_sha=truth_sha, provenance_source="human_truth"))
        ledger.append_event("truth_committed",
            actor={"type": "system", "id": "truth-writer"},
            action="commit_reviewed_truth", label_layer="human_truth",
            payload=truth_payload,
            artifacts=[{"kind": "truth", "sha256": truth_sha}])
        current = "truth_committed"

    if current == "truth_committed":
        if truth is None or not is_reference or not review_id:
            raise ValueError("truth commit cannot be versioned without current truth, reference, and review")
        manifest = {"schema_version": SCHEMA_VERSION, "input_id": ledger.input_id,
                    "version_id": args.version_id,
                    "parent_version": args.parent_version,
                    "dataset_split": args.dataset_split,
                    "split_group_id": args.split_group_id,
                    "ontology_version": args.ontology_version,
                    "source_kind": args.source_kind, "truth_sha256": truth_sha,
                    "review_id": review_id}
        manifest_bytes = encoded_json(manifest)
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        # Keep the immutable content hash in the event while avoiding legacy
        # Windows path-length failures in deeply nested ledger roots.
        manifest_path = ledger.run / "artifacts" / f"manifest-{manifest_sha[:16]}.json"
        write_bytes_once(manifest_path, manifest_bytes)
        version_payload = {"dataset_version": {"version_id": args.version_id,
            "parent_version": args.parent_version, "split": args.dataset_split,
            "split_group_id": args.split_group_id, "ontology_version": args.ontology_version,
            "source_kind": args.source_kind, "truth_sha256": truth_sha,
            "manifest_sha256": manifest_sha}}
        if slope_case:
            version_payload.update(_slope_context(human_truth=True, approved=True,
                candidate_sha=truth_sha, provenance_source="dataset_version"))
        ledger.append_event("version_committed",
            actor={"type": "system", "id": "dataset-versioner"},
            action="commit_immutable_dataset_version", label_layer="human_truth",
            payload=version_payload,
            artifacts=[{"kind": "dataset_manifest", "sha256": manifest_sha,
                        "bytes": len(manifest_bytes), "media_type": "application/json",
                        "uri": str(manifest_path)}])

    receipt = ledger.receipt()
    candidates = _frame_layer(state, args.topic, "candidates", "candidates_by_topic")
    bundle = {"schema_version": SCHEMA_VERSION, "input": envelope,
              "registration": _stable_registration(ledger),
              "events": ledger.events(), "receipt": receipt,
              "workbench_readback": {"truth_shape_count": len(truth) if truth else 0,
                  "is_reference_frame": is_reference,
                  "candidate_shape_count": len(candidates.get(str(args.frame), []))}}
    if review is not None:
        bundle["human_review"] = review
    bundle["bundle_sha256"] = hashlib.sha256(canonical_json(bundle)).hexdigest()
    output_path = write_snapshot(args.output, bundle)
    return bundle, output_path


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--workbench", default="http://127.0.0.1:8767")
    value.add_argument("--dataset-id", required=True)
    value.add_argument("--source-batch", required=True)
    value.add_argument("--frame", type=int, required=True)
    value.add_argument("--topic", required=True)
    value.add_argument("--source-kind", choices=["real", "sim"], required=True)
    value.add_argument("--ledger-root", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--sam-model-name", default="sam3")
    value.add_argument("--sam-model-version", required=True)
    value.add_argument("--sam-prompt-version", required=True)
    value.add_argument("--version-id", required=True)
    value.add_argument("--parent-version")
    value.add_argument("--dataset-split", choices=["train", "val", "test"], required=True)
    value.add_argument("--split-group-id", required=True,
                       help="sequence/entity group kept wholly inside one split")
    value.add_argument("--ontology-version", required=True)
    return value


def main():
    args = parser().parse_args()
    base = args.workbench.rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("workbench must be a loopback HTTP URL")
    state = request_json(base + "/api/state")
    image, media_type = request(base + f"/api/frame/{args.frame}?" +
                                urllib.parse.urlencode({"topic": args.topic}))
    timestamp, timestamp_source = captured_time(state["path"], args.topic, args.frame)
    review_data = request_json(base + "/api/human_reviews?" +
                               urllib.parse.urlencode({"frame": args.frame,
                                                       "topic": args.topic}))
    bundle, output_path = process_vertical(args, state=state, image=image,
        media_type=media_type, timestamp=timestamp, timestamp_source=timestamp_source,
        reviews=review_data.get("reviews", []))
    print(json.dumps({"output": str(output_path), "bundle": bundle},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
