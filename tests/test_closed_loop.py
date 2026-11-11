import json
from pathlib import Path
import tempfile
import unittest

from source.agent.closed_loop import ClosedLoopLedger, STATES, validate_input, validate_slope
from source.agent.downstream_mapping import mapping_contract


def envelope(content="a" * 64):
    return {"dataset_id": "sfl-cpd15y", "source_batch": "batch-a-000",
            "frame_id": "0", "sensor": "/capture/camera_0/image",
            "captured_at": "2026-09-20T10:00:00.000000Z", "source_kind": "real",
            "content_sha256": content, "source_uri": "/private/example.mcap",
            "timestamp_source": "capture"}


def actor(kind, identifier):
    return {"type": kind, "id": identifier}


def model(name):
    return {"name": name, "version": "test-v1", "code_commit": "1234567"}


def prompt(version="test-prompt-v1"):
    return {"version": version, "content_sha256": "c" * 64}


def quality(ledger, action="check_input"):
    return ledger.append_event("quality_checked", actor=actor("quality", "qc"),
        action=action, label_layer="preannotation",
        payload={"quality_checked": {"passed": True,
                                     "checks": ["decode", "dimensions", "content_hash"]}})


def accept_final_candidate(ledger, candidate_sha="d" * 64, reviewer="reviewer-a"):
    candidate = ledger.append_event("candidate_final",
        actor=actor("system", "candidate-freezer"), action="freeze_candidate",
        label_layer="agent_candidate",
        artifacts=[{"kind": "candidate_final", "sha256": candidate_sha}])
    ledger.append_event("human_review_required", actor=actor("system", "gate"),
        action="route_final_candidate", label_layer="agent_candidate")
    acceptance = ledger.append_event("human_accepted", actor=actor("human", reviewer),
        action="accept_candidate", label_layer="human_truth",
        payload={"accepted_event_id": candidate["event_id"],
                 "accepted_artifact_sha256": candidate_sha,
                 "truth_sha256": candidate_sha})
    return candidate, acceptance


def commit_truth(ledger, truth_sha="d" * 64):
    return ledger.append_event("truth_committed", actor=actor("system", "truth-writer"),
        action="commit_immutable_truth", label_layer="human_truth",
        payload={"truth_sha256": truth_sha},
        artifacts=[{"kind": "truth", "sha256": truth_sha}])


class ClosedLoopTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.ledger = ClosedLoopLedger(self.temporary.name, envelope())

    def test_input_requires_timezone_source_and_hash(self):
        self.assertEqual(validate_input(envelope())["source_kind"], "real")
        for change in ({"captured_at": "2026-09-20T10:00:00"},
                       {"source_kind": "mixed"}, {"content_sha256": "bad"}):
            with self.assertRaises(ValueError):
                validate_input(envelope() | change)

    def test_duplicate_input_and_event_are_idempotent(self):
        first = self.ledger.register()
        self.assertTrue(first["created"])
        self.assertEqual(len(self.ledger.events()), 1)
        second = ClosedLoopLedger(self.temporary.name, envelope())
        self.assertTrue(second.register()["idempotent_replay"])
        self.assertEqual(len(second.events()), 1)
        kwargs = dict(actor=actor("system", "gate"), action="route",
                      label_layer="preannotation", payload={"reason": "missing truth"})
        original = second.append_event("human_review_required", **kwargs)
        replay = second.append_event("human_review_required", **kwargs)
        self.assertEqual(original["event_id"], replay["event_id"])
        self.assertTrue(replay["idempotent_replay"])

    def test_replay_after_state_advances_returns_original_event(self):
        self.ledger.register()
        quality(self.ledger)
        kwargs = dict(actor=actor("sam", "sam3"), action="segment",
                      label_layer="preannotation", model=model("sam3"),
                      prompt=prompt("slope-point-v1"))
        original = self.ledger.append_event("sam_candidate_ready", **kwargs)
        self.ledger.append_event("vlm_reviewed", actor=actor("vlm", "qwen"),
            action="review", label_layer="agent_candidate", model=model("qwen"),
            prompt=prompt("review-v1"))
        replay = self.ledger.append_event("sam_candidate_ready", **kwargs)
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(replay["event_id"], original["event_id"])
        self.assertEqual(len(self.ledger.events()), 4)

    def test_receipt_is_stable_for_unchanged_history(self):
        self.ledger.register()
        first = self.ledger.receipt()
        second = self.ledger.receipt()
        self.assertEqual(first, second)
        self.assertEqual(len(list(self.ledger.receipts_dir.glob("*.json"))), 1)

    def test_same_identity_with_changed_content_is_rejected(self):
        self.ledger.register()
        with self.assertRaisesRegex(ValueError, "different content"):
            ClosedLoopLedger(self.temporary.name, envelope("b" * 64)).register()

    def test_open_set_cannot_write_ontology_or_truth(self):
        self.ledger.register()
        quality(self.ledger)
        self.ledger.append_event("sam_candidate_ready", actor=actor("sam", "sam3"),
            action="segment", label_layer="preannotation", model=model("sam3"),
            prompt=prompt("segment-v1"))
        self.ledger.append_event("vlm_reviewed", actor=actor("vlm", "qwen3-vl-8b"),
            action="classify_and_check", label_layer="agent_candidate",
            model=model("qwen3-vl-8b"), prompt=prompt("review-v1"))
        candidate = {"open_set": {"status": "new_label_candidate",
            "visual_evidence": ["distinct fork geometry"], "candidate_name": "fork-extension",
            "nearest_existing_class": "forklift", "difference": "separate attachment",
            "confidence": 0.72, "ontology_write": False}}
        self.ledger.append_event("open_set_candidate", actor=actor("vlm", "qwen3-vl-8b"),
            action="propose_unknown", label_layer="agent_candidate", payload=candidate,
            model=model("qwen3-vl-8b"), prompt=prompt("open-set-v1"))
        with self.assertRaisesRegex(ValueError, "cannot write"):
            bad = json.loads(json.dumps(candidate))
            bad["open_set"]["ontology_write"] = True
            ClosedLoopLedger(Path(self.temporary.name) / "bad-open-set", envelope()).append_event(
                "open_set_candidate", actor=actor("vlm", "qwen"), action="bad",
                label_layer="agent_candidate", payload=bad, model=model("qwen"),
                prompt=prompt("open-set-v1"))

    def test_truth_requires_human_gate_and_full_path_for_completion(self):
        self.ledger.register()
        self.ledger.append_event("human_review_required", actor=actor("system", "gate"),
            action="request_reference", label_layer="preannotation")
        receipt = self.ledger.receipt()
        self.assertFalse(receipt["human_gate_passed"])
        self.assertFalse(receipt["minimum_closed_loop_complete"])
        self.ledger.append_event("human_accepted", actor=actor("human", "reviewer-a"),
            action="approve_reference", label_layer="human_truth")
        reference_receipt = self.ledger.receipt()
        self.assertFalse(reference_receipt["human_gate_passed"])
        self.assertFalse(reference_receipt["minimum_closed_loop_complete"])
        quality(self.ledger)
        self.ledger.append_event("sam_candidate_ready", actor=actor("sam", "sam3"),
            action="segment", label_layer="preannotation", model=model("sam3"),
            prompt=prompt("segment-v1"))
        self.ledger.append_event("vlm_reviewed", actor=actor("vlm", "qwen"),
            action="classify_and_check", label_layer="agent_candidate",
            model=model("qwen"), prompt=prompt("review-v1"))
        self.ledger.append_event("agent_actioned", actor=actor("agent", "labelagent"),
            action="audit_or_edit", label_layer="agent_candidate",
            model=model("labelagent"), prompt=prompt("agent-v1"))
        self.ledger.append_event("vlm_reverified", actor=actor("vlm", "qwen"),
            action="recheck", label_layer="agent_candidate", model=model("qwen"),
            prompt=prompt("reverify-v1"))
        accept_final_candidate(self.ledger)
        commit_truth(self.ledger)
        receipt = self.ledger.receipt()
        self.assertTrue(receipt["human_gate_passed"])
        self.assertTrue(receipt["quality_gate_observed"])
        self.assertTrue(receipt["full_agent_path_observed"])
        self.assertTrue(receipt["minimum_closed_loop_complete"])
        self.assertFalse(receipt["training_ready"])

    def test_quality_model_prompt_artifact_and_version_gates(self):
        self.ledger.register()
        with self.assertRaisesRegex(ValueError, "invalid transition"):
            self.ledger.append_event("sam_candidate_ready", actor=actor("sam", "sam3"),
                action="segment", label_layer="preannotation", model=model("sam3"),
                prompt=prompt())
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            self.ledger.append_event("quality_checked", actor=actor("quality", "qc"),
                action="check", label_layer="preannotation",
                payload={"quality_checked": {"passed": True, "checks": ["decode"]}},
                artifacts=[{"kind": "report", "sha256": "BAD"}])
        quality(self.ledger)
        with self.assertRaisesRegex(ValueError, "model metadata"):
            self.ledger.append_event("sam_candidate_ready", actor=actor("sam", "sam3"),
                action="segment", label_layer="preannotation")

    def test_explicit_ontology_and_candidate_remediation_paths(self):
        self.ledger.register()
        quality(self.ledger)
        self.ledger.append_event("sam_candidate_ready", actor=actor("sam", "sam3"),
            action="segment", label_layer="preannotation", model=model("sam3"), prompt=prompt())
        self.ledger.append_event("vlm_reviewed", actor=actor("vlm", "qwen"),
            action="review", label_layer="agent_candidate", model=model("qwen"), prompt=prompt())
        open_set = {"open_set": {"status": "new_label_candidate",
            "visual_evidence": ["novel geometry"], "candidate_name": "candidate-x",
            "nearest_existing_class": "slope", "difference": "different edge profile",
            "confidence": 0.6, "ontology_write": False}}
        self.ledger.append_event("open_set_candidate", actor=actor("vlm", "qwen"),
            action="propose", label_layer="agent_candidate", payload=open_set,
            model=model("qwen"), prompt=prompt("open-v1"))
        self.ledger.append_event("ontology_review_required", actor=actor("system", "ontology-gate"),
            action="route", label_layer="agent_candidate")
        self.ledger.append_event("ontology_rejected", actor=actor("human", "reviewer-a"),
            action="reject_ontology", label_layer="agent_candidate")
        self.ledger.append_event("remediation_required", actor=actor("system", "remediation"),
            action="map_to_existing", label_layer="agent_candidate")
        quality(self.ledger, "recheck_after_ontology_remediation")
        self.assertEqual(self.ledger.events()[-1]["state"], "quality_checked")

        other = ClosedLoopLedger(self.temporary.name,
                                 envelope("d" * 64) | {"frame_id": "1"})
        other.register()
        other.append_event("human_review_required", actor=actor("system", "gate"),
            action="route", label_layer="agent_candidate")
        other.append_event("candidate_rejected", actor=actor("human", "reviewer-b"),
            action="reject_candidate", label_layer="human_truth")
        other.append_event("remediation_required", actor=actor("system", "remediation"),
            action="correct_candidate", label_layer="agent_candidate")
        quality(other, "recheck_after_candidate_remediation")
        self.assertEqual(other.events()[-1]["state"], "quality_checked")

    def test_dataset_version_requires_split_hashes_and_manifest_artifact(self):
        self.ledger.register()
        quality(self.ledger)
        self.ledger.append_event("sam_candidate_ready", actor=actor("sam", "sam3"),
            action="segment", label_layer="preannotation", model=model("sam3"),
            prompt=prompt())
        accept_final_candidate(self.ledger, "e" * 64)
        commit_truth(self.ledger, "e" * 64)
        with self.assertRaisesRegex(ValueError, "payload.dataset_version"):
            self.ledger.append_event("version_committed", actor=actor("system", "versioner"),
                action="version", label_layer="human_truth")
        manifest = "f" * 64
        version_payload = {"dataset_version": {"version_id": "slope-truth-v1",
            "parent_version": None, "split": "train", "split_group_id": "sequence-000",
            "ontology_version": "seven-class-v1", "source_kind": "real",
            "truth_sha256": "e" * 64, "manifest_sha256": manifest}}
        missing_split = json.loads(json.dumps(version_payload))
        missing_split["dataset_version"].pop("split")
        with self.assertRaisesRegex(ValueError, "fields are incomplete"):
            self.ledger.append_event("version_committed", actor=actor("system", "versioner"),
                action="version_missing_split", label_layer="human_truth",
                payload=missing_split,
                artifacts=[{"kind": "dataset_manifest", "sha256": manifest}])
        with self.assertRaisesRegex(ValueError, "manifest SHA-256"):
            self.ledger.append_event("version_committed", actor=actor("system", "versioner"),
                action="version", label_layer="human_truth", payload=version_payload,
                artifacts=[{"kind": "report", "sha256": manifest}])
        unknown_parent = json.loads(json.dumps(version_payload))
        unknown_parent["dataset_version"]["parent_version"] = "missing-parent-v0"
        with self.assertRaisesRegex(ValueError, "parent_version is not present"):
            self.ledger.append_event("version_committed", actor=actor("system", "versioner"),
                action="version_unknown_parent", label_layer="human_truth",
                payload=unknown_parent,
                artifacts=[{"kind": "dataset_manifest", "sha256": manifest}])
        self.ledger.append_event("version_committed", actor=actor("system", "versioner"),
            action="version", label_layer="human_truth",
            payload=version_payload,
            artifacts=[{"kind": "dataset_manifest", "sha256": manifest}])
        receipt = self.ledger.receipt()
        self.assertTrue(receipt["training_ready"])
        self.assertEqual(receipt["dataset_version"], "slope-truth-v1")
        self.assertEqual(receipt["dataset_split"], "train")

    def test_reference_acceptance_cannot_authorize_a_later_candidate_truth(self):
        self.ledger.register()
        self.ledger.append_event("human_review_required", actor=actor("system", "reference-gate"),
            action="request_reference", label_layer="preannotation")
        self.ledger.append_event("human_accepted", actor=actor("human", "reviewer-a"),
            action="approve_reference", label_layer="human_truth")
        quality(self.ledger)
        self.ledger.append_event("sam_candidate_ready", actor=actor("sam", "sam3"),
            action="segment", label_layer="preannotation", model=model("sam3"), prompt=prompt())
        candidate = self.ledger.append_event("candidate_final",
            actor=actor("system", "candidate-freezer"), action="freeze_candidate",
            label_layer="agent_candidate",
            artifacts=[{"kind": "candidate_final", "sha256": "9" * 64}])
        with self.assertRaisesRegex(ValueError, "invalid transition"):
            commit_truth(self.ledger, "9" * 64)
        self.ledger.append_event("human_review_required", actor=actor("system", "final-gate"),
            action="route_final", label_layer="agent_candidate")
        with self.assertRaisesRegex(ValueError, "accepted_event_id"):
            self.ledger.append_event("human_accepted", actor=actor("human", "reviewer-a"),
                action="accept_candidate", label_layer="human_truth")
        linked = self.ledger.append_event("human_accepted", actor=actor("human", "reviewer-a"),
            action="accept_candidate", label_layer="human_truth",
            payload={"accepted_event_id": candidate["event_id"],
                     "accepted_artifact_sha256": "9" * 64})
        self.assertEqual(linked["state"], "human_accepted")
        commit_truth(self.ledger, "9" * 64)
        self.assertTrue(self.ledger.receipt()["human_gate_passed"])

    def test_split_group_cannot_leak_between_train_and_test(self):
        def ready(frame, content, split, group):
            ledger = ClosedLoopLedger(self.temporary.name,
                envelope(content) | {"frame_id": frame})
            ledger.register()
            quality(ledger)
            ledger.append_event("sam_candidate_ready", actor=actor("sam", "sam3"),
                action="segment", label_layer="preannotation", model=model("sam3"),
                prompt=prompt())
            accept_final_candidate(ledger, content)
            commit_truth(ledger, content)
            manifest = ("a" if split == "train" else "b") * 64
            payload = {"dataset_version": {"version_id": "dataset-v1",
                "parent_version": None, "split": split, "split_group_id": group,
                "ontology_version": "seven-class-v1", "source_kind": "real",
                "truth_sha256": content, "manifest_sha256": manifest}}
            return ledger, payload, manifest

        train, payload, manifest = ready("10", "1" * 64, "train", "entity-ramp-01")
        train.append_event("version_committed", actor=actor("system", "versioner"),
            action="version", label_layer="human_truth", payload=payload,
            artifacts=[{"kind": "dataset_manifest", "sha256": manifest}])
        test, payload, manifest = ready("11", "2" * 64, "test", "entity-ramp-01")
        with self.assertRaisesRegex(ValueError, "isolation violation"):
            test.append_event("version_committed", actor=actor("system", "versioner"),
                action="version", label_layer="human_truth", payload=payload,
                artifacts=[{"kind": "dataset_manifest", "sha256": manifest}])

    def test_slope_default_passable_policy_and_explicit_override(self):
        self.ledger.register()
        quality(self.ledger)
        slope = {"task_context": {"hard_case_tags": ["slope"]},
                 "slope_context": {"semantic_label": "slope_visible",
                    "visible_status": "candidate_supported",
                    "human_pixel_truth_available": False, "approved_for_training": False,
                    "business_mapping": "traversable_area",
                    "downstream_mapping": mapping_contract(),
                    "traversability_status": "passable",
                    "policy_default": "passable",
                    "policy_basis": "user_defined_project_semantics",
                    "policy_override": {"status": "none", "rule_id": None,
                        "reason": None, "decided_by": None,
                        "evidence_artifact_ids": []},
                    "candidate_sha256": "d" * 64,
                    "evidence_provenance": [{"kind": "candidate_hash",
                        "source": "sam3", "sha256": "d" * 64}],
                    "hard_set_strata": ["unassigned"]}}
        untagged = json.loads(json.dumps(slope))
        untagged["task_context"]["hard_case_tags"] = []
        with self.assertRaisesRegex(ValueError, "slope hard-case tag"):
            self.ledger.append_event("sam_candidate_ready", actor=actor("sam", "sam3"),
                action="segment", label_layer="preannotation", payload=untagged,
                model=model("sam3"), prompt=prompt("slope-v1"))
        event = self.ledger.append_event("sam_candidate_ready", actor=actor("sam", "sam3"),
            action="segment", label_layer="preannotation", payload=slope,
            model=model("sam3"), prompt=prompt("slope-v1"))
        self.assertEqual(event["payload"]["slope_context"]["traversability_status"],
                         "passable")
        invalid = json.loads(json.dumps(slope))
        invalid["slope_context"]["traversability_status"] = "unknown"
        with self.assertRaisesRegex(ValueError, "traversability_status"):
            validate_slope("sam_candidate_ready", invalid, [], {})
        merged = json.loads(json.dumps(slope))
        merged["slope_context"]["downstream_mapping"]["preserve_source_labels"] = False
        with self.assertRaisesRegex(ValueError, "preserve floor/slope"):
            validate_slope("sam_candidate_ready", merged, [], {})

        override = json.loads(json.dumps(slope))
        override["slope_context"].update(traversability_status="non_passable",
            policy_override={"status": "non_passable", "rule_id": "SITE-RAMP-001",
                "reason": "explicit site exclusion", "decided_by": "system_policy",
                "evidence_artifact_ids": []})
        validate_slope("sam_candidate_ready", override, [], {}, "system")
        with self.assertRaisesRegex(ValueError, "models cannot create"):
            validate_slope("sam_candidate_ready", override, [], {}, "vlm")

        evaluation = {"task_context": {"hard_case_tags": ["slope"]},
                      "slope_context": {"semantic_label": "slope_visible",
                        "visible_status": "candidate_supported",
                        "human_pixel_truth_available": False, "approved_for_training": False,
                        "business_mapping": "traversable_area",
                        "downstream_mapping": mapping_contract(),
                        "traversability_status": "passable",
                        "policy_default": "passable",
                        "policy_basis": "user_defined_project_semantics",
                        "policy_override": {"status": "none", "rule_id": None,
                            "reason": None, "decided_by": None,
                            "evidence_artifact_ids": []},
                        "candidate_sha256": "d" * 64,
                        "evidence_provenance": [{"kind": "candidate_hash",
                            "source": "sam3", "sha256": "d" * 64}],
                        "hard_set_strata": ["unassigned"]}}
        with self.assertRaisesRegex(ValueError, "human pixel truth"):
            self.ledger.append_event("evaluation_completed", actor=actor("quality", "evaluator"),
                action="evaluate", label_layer="agent_candidate", payload=evaluation)

    def test_slope_evaluation_requires_hard_set_and_complete_single_class_metrics(self):
        slope = {"task_context": {"hard_case_tags": ["slope"]},
                 "slope_context": {"semantic_label": "slope_visible",
                    "visible_status": "human_confirmed",
                    "human_pixel_truth_available": True, "approved_for_training": False,
                    "business_mapping": "traversable_area",
                    "downstream_mapping": mapping_contract(),
                    "traversability_status": "passable",
                    "policy_default": "passable",
                    "policy_basis": "user_defined_project_semantics",
                    "policy_override": {"status": "none", "rule_id": None,
                        "reason": None, "decided_by": None,
                        "evidence_artifact_ids": []},
                    "candidate_sha256": "d" * 64,
                    "evidence_provenance": [{"kind": "human_pixel_truth",
                        "source": "review-receipt", "sha256": "e" * 64}],
                    "hard_set_strata": ["unassigned"]}}
        with self.assertRaisesRegex(ValueError, "complete slope-specific metrics"):
            validate_slope("evaluation_completed", slope, [], {"slope_iou": .8})
        metrics = {"slope_iou": .8, "slope_dice": .88, "slope_boundary_f1": .7,
            "slope_boundary_tolerance_px": 3, "slope_pixel_precision": .9,
            "slope_pixel_recall": .85, "slope_instance_precision": .75,
            "slope_instance_recall": .8, "slope_to_floor_confusion_pixels": 12,
            "floor_to_slope_confusion_pixels": 8,
            "slope_to_step_curb_confusion_pixels": 4,
            "step_curb_to_slope_confusion_pixels": 3,
            "slope_false_traversable_count": 0}
        with self.assertRaisesRegex(ValueError, "assigned hard-set strata"):
            validate_slope("evaluation_completed", slope, [], metrics)
        slope["slope_context"]["hard_set_strata"] = ["far", "backlight",
                                                        "slope_top_boundary"]
        validate_slope("evaluation_completed", slope, [], metrics)

    def test_nonhuman_cannot_approve_and_hash_chain_detects_tampering(self):
        self.ledger.register()
        self.ledger.append_event("human_review_required", actor=actor("system", "gate"),
            action="route", label_layer="agent_candidate")
        with self.assertRaisesRegex(ValueError, "human actor"):
            self.ledger.append_event("human_accepted", actor=actor("agent", "bad"),
                action="approve", label_layer="human_truth")
        event_path = sorted((self.ledger.run / "events").glob("*.json"))[-1]
        event = json.loads(event_path.read_text(encoding="utf-8"))
        event["payload"] = {"tampered": True}
        event_path.write_text(json.dumps(event), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "hash chain"):
            self.ledger.events()

    def test_machine_schema_is_valid_json(self):
        schema = Path(__file__).resolve().parents[1] / "schemas/labelagent-closed-loop.schema.json"
        parsed = json.loads(schema.read_text(encoding="utf-8"))
        self.assertEqual(parsed["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertFalse(parsed["$defs"]["event"]["additionalProperties"])
        self.assertFalse(parsed["$defs"]["receipt"]["additionalProperties"])
        self.assertEqual(set(parsed["$defs"]["event"]["properties"]["state"]["enum"]),
                         STATES)
        self.assertTrue({"quality_gate_observed", "dataset_version", "dataset_split"} <=
                        set(parsed["$defs"]["receipt"]["required"]))


if __name__ == "__main__":
    unittest.main()
