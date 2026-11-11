import copy
import json
from pathlib import Path
import tempfile
import unittest

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from source.agent.factory_orchestrator import (
    BUSINESS_STATES,
    FactoryOrchestrationLedger,
    _event_body,
    build_event,
    contract_ref,
    digest,
    stable_stream_id,
    validate_event,
    validate_ledger_snapshot,
    write_snapshot_exclusive,
)
from tools.build_factory_orchestration_static_acceptance import (
    GENERATED_AT,
    TraceBuilder,
    build_event_set,
    build_static_acceptance,
    common_candidate_path,
    gate,
    identity,
    provenance,
)


ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "docs/test-results/factory-labelagent-orchestration-ledger-static-20261111.json"
REPORT_PATH = ROOT / "docs/test-results/factory-labelagent-orchestration-replay-report-20261111.json"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def schema_errors(value, schema_name):
    event_schema = load(ROOT / "schemas/factory-labelagent-orchestration-event.schema.json")
    registry = Registry().with_resource(
        event_schema["$id"], Resource.from_contents(event_schema))
    schema = load(ROOT / "schemas" / schema_name)
    return list(Draft202012Validator(
        schema, registry=registry, format_checker=FormatChecker()
    ).iter_errors(value))


def first_event(frame="UNIT_FRAME", split="train"):
    identity_value = identity(frame, split)
    output = digest((frame + ":ingested").encode())
    return build_event(
        event_type="state.transition", occurred_at="2026-11-11T22:00:00+08:00",
        trace_id="unit_trace_" + frame, identity=identity_value,
        state_version=1, previous_state=None, state="ingested",
        previous_event_id=None,
        contract_refs=[contract_ref(
            "factory_ingress", "schemas/factory-data-ingress-event.schema.json",
            "example://unit/ingress.json", output)],
        provenance=provenance("real"),
    )


def next_event(previous, state, contract_type, *, split=None,
               human_gate=None, classification=None, reason_codes=None,
               truth_version=None, evaluation=None, training_run=None,
               effect_mode="none"):
    identity_value = copy.deepcopy(previous["identity"])
    if split is not None:
        identity_value["split"] = split
    output = digest({"unit": previous["event_id"], "state": state})
    return build_event(
        event_type="state.transition", occurred_at="2026-11-11T22:00:01+08:00",
        trace_id=previous["trace_id"], identity=identity_value,
        state_version=previous["state_version"] + 1,
        previous_state=previous["state"], state=state,
        previous_event_id=previous["event_id"],
        human_gate=human_gate or previous["human_gate"],
        provenance=previous["provenance"],
        classification=(previous["classification"] if classification is None
                        else classification),
        reason_codes=reason_codes,
        truth_version=truth_version, evaluation=evaluation,
        training_run=training_run,
        contract_refs=[contract_ref(
            contract_type, "schemas/unit-static.schema.json",
            f"example://unit/{state}.json", output)],
        effect={"mode": effect_mode, "external_call_count": 0,
                "production_mutation": False},
    )


def operational(previous, event_type, status, route, *, attempt=0,
                retryable=False, depth=0, failure_reason=None):
    output = digest({"op": event_type, "previous": previous["event_id"],
                     "attempt": attempt})
    return build_event(
        event_type=event_type, occurred_at="2026-11-11T22:01:00+08:00",
        trace_id=previous["trace_id"], identity=previous["identity"],
        state_version=previous["state_version"] + 1,
        previous_state=previous["state"], state=previous["state"],
        previous_event_id=previous["event_id"], operational_status=status,
        retry={"attempt": attempt, "max_attempts": 2,
               "retryable": retryable},
        queue={"name": "factory_labelagent", "depth": depth,
               "high_watermark": 2, "route": route},
        human_gate=previous["human_gate"], provenance=previous["provenance"],
        classification=previous["classification"],
        failure_reason=failure_reason,
        contract_refs=[contract_ref(
            "factory_ingress", "schemas/factory-data-ingress-event.schema.json",
            "example://unit/operation.json", output)],
    )


class FactoryOrchestratorTests(unittest.TestCase):
    def test_exact_static_ledger_and_report_are_schema_and_runtime_valid(self):
        ledger = load(LEDGER_PATH)
        report = load(REPORT_PATH)
        self.assertEqual(schema_errors(
            ledger, "factory-labelagent-orchestration-ledger.schema.json"), [])
        self.assertEqual(schema_errors(
            report, "factory-labelagent-orchestration-replay-report.schema.json"), [])
        self.assertTrue(validate_ledger_snapshot(ledger))
        for record in ledger["records"]:
            self.assertTrue(validate_event(record["event"]))

    def test_static_builder_is_byte_semantically_deterministic(self):
        outputs = build_static_acceptance()
        self.assertEqual(outputs[LEDGER_PATH], load(LEDGER_PATH))
        self.assertEqual(outputs[REPORT_PATH], load(REPORT_PATH))

    def test_four_required_traces_have_expected_final_states(self):
        ledger = load(LEDGER_PATH)
        self.assertEqual({key: value["final_state"]
                          for key, value in ledger["traces"].items()}, {
            "known_label_waiting_human": "human_review_required",
            "unknown_routes_human": "human_review_required",
            "qc_failure_quarantined": "quarantined",
            "slope_regression_baseline_retained": "baseline_retained",
        })

    def test_known_candidate_without_human_truth_stops_at_review(self):
        ledger = load(LEDGER_PATH)
        events = [record["event"] for record in ledger["records"]
                  if record["event"]["trace_id"] == "known_label_waiting_human"]
        self.assertEqual(events[-1]["state"], "human_review_required")
        self.assertEqual(events[-1]["human_gate"]["status"], "pending")
        states = {event["state"] for event in events}
        self.assertNotIn("truth_promoted", states)
        self.assertNotIn("training_eligible", states)

    def test_unknown_candidate_routes_to_human_and_never_mutates_ontology(self):
        ledger = load(LEDGER_PATH)
        events = [record["event"] for record in ledger["records"]
                  if record["event"]["trace_id"] == "unknown_routes_human"]
        self.assertEqual(events[-1]["state"], "human_review_required")
        self.assertEqual(events[-1]["classification"], "unknown_candidate")
        self.assertIn("ontology_review_required", events[-1]["reason_codes"])

    def test_qc_failure_is_not_silently_skipped_and_is_quarantined(self):
        ledger = load(LEDGER_PATH)
        events = [record["event"] for record in ledger["records"]
                  if record["event"]["trace_id"] == "qc_failure_quarantined"]
        self.assertEqual(events[-1]["operational_status"], "quarantined")
        self.assertEqual(events[-1]["queue"]["route"], "quarantine")
        self.assertTrue(events[-1]["failure_reason"])
        self.assertFalse(ledger["silent_skip"])

    def test_slope_regression_retains_baseline(self):
        ledger = load(LEDGER_PATH)
        events = [record["event"] for record in ledger["records"]
                  if record["event"]["trace_id"] ==
                  "slope_regression_baseline_retained"]
        evaluated = next(event for event in events if event["state"] == "evaluated")
        self.assertTrue(evaluated["evaluation"]["overall_guard_passed"])
        self.assertFalse(evaluated["evaluation"]["slope_guard_passed"])
        self.assertFalse(evaluated["evaluation"]["deployment_gate_passed"])
        self.assertEqual(events[-1]["state"], "baseline_retained")
        self.assertNotIn("deployment_candidate", {event["state"] for event in events})

    def test_static_effect_boundaries_are_zero_and_non_production(self):
        ledger = load(LEDGER_PATH)
        self.assertTrue(all(value == 0
                            for value in ledger["external_call_counts"].values()))
        for record in ledger["records"]:
            event = record["event"]
            self.assertEqual(event["provenance"]["execution_mode"],
                             "static_replay")
            self.assertEqual(event["provenance"]["sample_kind"],
                             "synthetic_contract_fixture_not_production")
            self.assertEqual(event["effect"]["external_call_count"], 0)
            self.assertFalse(event["effect"]["production_mutation"])

    def test_real_and_simulation_accounting_is_separate(self):
        accounting = load(LEDGER_PATH)["source_accounting"]
        self.assertEqual(accounting["real"], {"streams": 3, "events": 22})
        self.assertEqual(accounting["simulation"], {"streams": 1, "events": 7})

    def test_floor_slope_are_preserved_and_consumed_as_union(self):
        for record in load(LEDGER_PATH)["records"]:
            policy = record["event"]["label_policy"]
            self.assertEqual(policy["semantic_labels"], ["floor", "slope"])
            self.assertTrue(policy["preserve_separate_training_labels"])
            self.assertEqual(policy["downstream_target"], "traversable_area")
            self.assertEqual(policy["downstream_source_labels"],
                             ["floor", "slope"])
            self.assertEqual(policy["slope_default"], "passable")

    def test_stable_stream_and_event_keys_are_deterministic(self):
        value = first_event()
        self.assertEqual(value, first_event())
        self.assertEqual(value["stream_id"], stable_stream_id(value["identity"]))

    def test_identical_event_replay_is_idempotent_without_append(self):
        ledger = FactoryOrchestrationLedger(ledger_id="unit")
        event = first_event()
        first, replay = ledger.append(event)
        self.assertFalse(replay)
        second, replay = ledger.append(copy.deepcopy(event))
        self.assertTrue(replay)
        self.assertEqual(first, second)
        self.assertEqual(len(ledger.records), 1)

    def test_old_identical_event_replay_remains_idempotent_after_state_advances(self):
        ledger = FactoryOrchestrationLedger(ledger_id="unit")
        first = first_event()
        ledger.append(first)
        ledger.append(next_event(first, "qc_passed", "factory_qc"))
        _, replay = ledger.append(copy.deepcopy(first))
        self.assertTrue(replay)
        self.assertEqual(len(ledger.records), 2)

    def test_same_idempotency_key_with_different_immutable_body_is_rejected(self):
        ledger = FactoryOrchestrationLedger(ledger_id="unit")
        event = first_event()
        ledger.append(event)
        conflict = copy.deepcopy(event)
        conflict["reason_codes"] = ["mutated_replay"]
        conflict["event_id"] = digest(_event_body(conflict))
        with self.assertRaisesRegex(ValueError, "idempotency key collision"):
            ledger.append(conflict)

    def test_illegal_state_jump_is_rejected(self):
        ledger = FactoryOrchestrationLedger(ledger_id="unit")
        first = first_event()
        ledger.append(first)
        jump = next_event(first, "candidate_planned", "candidate_batch_plan")
        with self.assertRaisesRegex(ValueError, "illegal state transition"):
            ledger.append(jump)

    def test_out_of_order_or_stale_previous_event_is_rejected(self):
        ledger = FactoryOrchestrationLedger(ledger_id="unit")
        first = first_event()
        ledger.append(first)
        qc = next_event(first, "qc_passed", "factory_qc")
        ledger.append(qc)
        stale = next_event(first, "qc_passed", "factory_qc")
        stale["bindings"]["output_sha256"] = digest(b"different-stale-output")
        stale["event_key"] = digest({
            "stream_id": stale["stream_id"], "event_type": stale["event_type"],
            "state_version": stale["state_version"],
            "previous_state": stale["previous_state"], "state": stale["state"],
            "input_sha256": stale["bindings"]["input_sha256"],
            "output_sha256": stale["bindings"]["output_sha256"],
        })
        stale["event_id"] = digest(_event_body(stale))
        self.assertEqual(stale["previous_event_id"], first["event_id"])
        self.assertNotEqual(stale["event_id"], qc["event_id"])
        self.assertNotEqual(stale["bindings"]["output_sha256"],
                            qc["bindings"]["output_sha256"])
        with self.assertRaisesRegex(ValueError, "out-of-order"):
            ledger.append(stale)

    def test_cross_split_mutation_is_rejected(self):
        ledger = FactoryOrchestrationLedger(ledger_id="unit")
        first = first_event()
        ledger.append(first)
        changed = next_event(first, "qc_passed", "factory_qc", split="test")
        with self.assertRaises(ValueError):
            ledger.append(changed)

    def test_candidate_acceptance_requires_approved_human_gate(self):
        events = [event for event in build_event_set()
                  if event["trace_id"] == "known_label_waiting_human"]
        ledger = FactoryOrchestrationLedger(ledger_id="unit")
        for event in events[:-1]:
            ledger.append(event)
        rechecked = events[-2]
        candidate = next_event(
            rechecked, "candidate_accepted", "human_review_receipt",
            human_gate=gate("pending"), classification="known_label")
        with self.assertRaisesRegex(ValueError, "approved human review"):
            ledger.append(candidate)

    def test_unknown_acceptance_requires_ontology_human_approval(self):
        events = [event for event in build_event_set()
                  if event["trace_id"] == "unknown_routes_human"]
        ledger = FactoryOrchestrationLedger(ledger_id="unit")
        for event in events:
            ledger.append(event)
        pending = events[-1]
        approved = gate("approved", digest(b"candidate"), digest(b"image"))
        candidate = next_event(
            pending, "candidate_accepted", "human_review_receipt",
            human_gate=approved, classification="unknown_candidate")
        with self.assertRaisesRegex(ValueError, "ontology human approval"):
            ledger.append(candidate)

    def test_retry_budget_routes_third_failure_to_dead_letter(self):
        ledger = FactoryOrchestrationLedger(ledger_id="unit", retry_budget=2)
        current = first_event()
        ledger.append(current)
        for attempt in (1, 2):
            failed = operational(current, "operation.failed", "retry_wait",
                                 "retry", attempt=attempt, retryable=True,
                                 failure_reason="temporary_failure")
            ledger.append(failed)
            released = operational(failed, "operation.retry_released", "ready",
                                   "active", attempt=attempt)
            ledger.append(released)
            current = released
        dead = operational(current, "operation.failed", "dead_letter",
                           "dead_letter", attempt=3, retryable=True,
                           failure_reason="retry_budget_exhausted")
        ledger.append(dead)
        self.assertEqual(ledger.snapshot(generated_at=GENERATED_AT)["queues"][
            "dead_letter"], 1)

    def test_backpressure_applies_and_releases_without_state_change(self):
        ledger = FactoryOrchestrationLedger(ledger_id="unit", max_inflight=2)
        first = first_event()
        ledger.append(first)
        held = operational(first, "queue.backpressure_applied", "backpressured",
                           "backpressure", depth=2)
        ledger.append(held)
        released = operational(held, "queue.backpressure_released", "ready",
                               "active", depth=1)
        ledger.append(released)
        self.assertEqual(released["state"], "ingested")

    def test_checkpoint_recovery_reproduces_exact_chain(self):
        events = build_event_set()
        full = FactoryOrchestrationLedger(ledger_id="recover")
        for event in events:
            full.append(event)
        prefix = FactoryOrchestrationLedger(ledger_id="recover")
        for event in events[:11]:
            prefix.append(event)
        restored = FactoryOrchestrationLedger.from_snapshot(
            prefix.snapshot(generated_at=GENERATED_AT))
        for event in events[11:]:
            restored.append(event)
        self.assertEqual(restored.snapshot(generated_at=GENERATED_AT),
                         full.snapshot(generated_at=GENERATED_AT))

    def test_snapshot_metadata_tamper_is_rejected_during_resume(self):
        snapshot = load(LEDGER_PATH)
        snapshot["queues"]["quarantine"] = 0
        snapshot["ledger_sha256"] = digest({key: value for key, value in snapshot.items()
                                             if key != "ledger_sha256"})
        with self.assertRaisesRegex(ValueError, "metadata"):
            FactoryOrchestrationLedger.from_snapshot(snapshot)

    def test_test_split_can_only_end_as_evaluation_truth(self):
        builder = TraceBuilder("test_evaluation_truth", identity("TEST_FRAME", "test"),
                               provenance("real"), 100)
        common_candidate_path(builder, "known_label")
        builder.transition("human_review_required", classification="known_label",
                           human_gate=gate("pending"))
        approved = gate("approved", builder.candidate_sha,
                        builder.source_image_sha)
        builder.transition("candidate_accepted", classification="known_label",
                           human_gate=approved)
        truth = {"version_id": "STATIC_EVAL_TRUTH", "manifest_sha256": digest(b"eval"),
                 "split": "test", "immutable": True}
        builder.transition("evaluation_truth", classification="known_label",
                           human_gate=approved, truth_version=truth)
        ledger = FactoryOrchestrationLedger(ledger_id="eval-truth")
        for event in builder.events:
            ledger.append(event)
        self.assertEqual(builder.events[-1]["state"], "evaluation_truth")
        self.assertNotIn("training_eligible",
                         {event["state"] for event in builder.events})

    def test_passed_evaluation_can_deploy_then_roll_back_in_static_replay(self):
        events = [event for event in build_event_set()
                  if event["trace_id"] == "slope_regression_baseline_retained"]
        prefix = [event for event in events if event["state"] != "baseline_retained"]
        evaluated = prefix[-1]
        passing = copy.deepcopy(evaluated["evaluation"])
        passing["slope_guard_passed"] = True
        passing["deployment_gate_passed"] = True
        evaluated["evaluation"] = passing
        evaluated["reason_codes"] = ["overall_and_slope_guards_passed"]
        evaluated["event_id"] = digest(_event_body(evaluated))
        # The next event must point at the changed immutable evaluation event.
        candidate = next_event(
            evaluated, "deployment_candidate", "sam3_stage0_deployment_decision",
            human_gate=evaluated["human_gate"], classification="known_label",
            truth_version=evaluated["truth_version"], evaluation=passing,
            training_run=evaluated["training_run"], effect_mode="simulated")
        deployed = next_event(candidate, "deployed", "deployment_receipt",
                              human_gate=candidate["human_gate"],
                              classification="known_label",
                              truth_version=candidate["truth_version"],
                              evaluation=passing,
                              training_run=candidate["training_run"],
                              effect_mode="simulated")
        rolled_back = next_event(deployed, "rolled_back", "rollback_receipt",
                                 human_gate=deployed["human_gate"],
                                 classification="known_label",
                                 truth_version=deployed["truth_version"],
                                 evaluation=passing,
                                 training_run=deployed["training_run"],
                                 effect_mode="simulated")
        ledger = FactoryOrchestrationLedger(ledger_id="deploy-rollback")
        for event in prefix + [candidate, deployed, rolled_back]:
            ledger.append(event)
        self.assertEqual(rolled_back["state"], "rolled_back")
        self.assertFalse(rolled_back["effect"]["production_mutation"])

    def test_ledger_chain_or_event_tampering_is_rejected(self):
        snapshot = load(LEDGER_PATH)
        snapshot["records"][0]["event"]["reason_codes"] = ["tampered"]
        snapshot["ledger_sha256"] = digest({key: value for key, value in snapshot.items()
                                             if key != "ledger_sha256"})
        with self.assertRaises(ValueError):
            validate_ledger_snapshot(snapshot)

    def test_append_only_writer_refuses_existing_path(self):
        snapshot = load(LEDGER_PATH)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            target = Path(directory) / "ledger.json"
            write_snapshot_exclusive(target, snapshot)
            with self.assertRaises(FileExistsError):
                write_snapshot_exclusive(target, snapshot)

    def test_all_contract_schema_paths_referenced_by_static_events_exist(self):
        for record in load(LEDGER_PATH)["records"]:
            for reference in record["event"]["contract_refs"]:
                self.assertTrue((ROOT / reference["schema_path"]).is_file(),
                                reference["schema_path"])

    def test_required_business_states_are_present(self):
        required = {"ingested", "qc_passed", "quarantined", "candidate_planned",
                    "vlm_reviewed", "agent_corrected", "vlm_rechecked",
                    "human_review_required", "candidate_accepted",
                    "truth_promoted", "evaluation_truth", "training_eligible",
                    "trained", "evaluated", "deployment_candidate",
                    "baseline_retained", "deployed", "rolled_back"}
        self.assertTrue(required.issubset(BUSINESS_STATES))


if __name__ == "__main__":
    unittest.main()
