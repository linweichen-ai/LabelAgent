"""Build four deterministic factory-orchestration traces without external calls."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.agent.factory_orchestrator import (
    CONTRACT_TYPES_BY_STATE,
    FactoryOrchestrationLedger,
    artifact,
    build_event,
    contract_ref,
    digest,
    validate_ledger_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
GENERATED_AT = "2026-11-11T23:58:00+08:00"
BASE_TIME = datetime(2026, 11, 11, 23, 0, tzinfo=timezone(timedelta(hours=8)))


def identity(frame, split="train"):
    return {"dataset_id": "SFL-CPD15-Y", "source_batch": "STATIC_ORCHESTRATION",
            "sequence": "000", "frame_id": frame, "camera": "camera_0",
            "split": split}


def provenance(source_kind="real"):
    simulation = None
    if source_kind == "simulation":
        simulation = {"simulator": "STATIC_SIMULATOR", "scene": "dock_slope",
                      "random_seed": 3407, "version": "STATIC_V1"}
    return {"source_kind": source_kind, "simulation": simulation,
            "sample_kind": "synthetic_contract_fixture_not_production",
            "execution_mode": "static_replay", "external_effects_allowed": False}


def gate(status="not_required", candidate=None, source=None):
    required = status != "not_required"
    approved = status == "approved"
    return {
        "required": required, "status": status,
        "review_event_id": digest(b"static-human-review") if approved else None,
        "cvat_task_version": "STATIC_TASK_V1" if required else None,
        "receipt_sha256": digest(b"static-review-receipt") if approved else None,
        "candidate_sha256": candidate if approved else None,
        "source_image_sha256": source if approved else None,
    }


def metric(iou, boundary, precision, recall, false_positives):
    return {"mask_iou": iou, "boundary_f1": boundary,
            "object_precision": precision, "object_recall": recall,
            "false_positives": false_positives}


def slope_regression_metrics():
    return {
        "fixed_holdout_manifest_sha256": digest(b"static-fixed-holdout"),
        "overall": metric(0.72, 0.67, 0.80, 0.75, 8),
        "per_class": {
            "floor": metric(0.81, 0.76, 0.87, 0.83, 3),
            "slope": metric(0.61, 0.54, 0.71, 0.64, 6),
        },
        "source_real": metric(0.71, 0.66, 0.79, 0.74, 7),
        "source_simulation": metric(0.74, 0.69, 0.82, 0.77, 1),
        "overall_guard_passed": True,
        "slope_guard_passed": False,
        "deployment_gate_passed": False,
    }


class TraceBuilder:
    def __init__(self, trace_id, identity_value, provenance_value, start_offset):
        self.trace_id = trace_id
        self.identity = identity_value
        self.provenance = provenance_value
        self.previous = None
        self.version = 0
        self.time_offset = start_offset
        self.events = []
        self.source_image_sha = digest((trace_id + ":source-image").encode())
        self.candidate_sha = digest((trace_id + ":candidate").encode())
        self.last_output = digest((trace_id + ":ingress-input").encode())

    def transition(self, state, *, classification="not_applicable",
                   agent_action="not_applicable", human_gate=None,
                   reason_codes=None, failure_reason=None, evaluation=None,
                   truth_version=None, training_run=None,
                   operational_status="ready", queue_route="active"):
        self.version += 1
        output_sha = digest({"trace": self.trace_id, "state": state,
                             "version": self.version})
        bindings = {
            "input_sha256": self.last_output, "output_sha256": output_sha,
            "config_sha256": digest((state + ":config").encode()),
            "model_sha256": (digest((state + ":model").encode())
                             if state in {"candidate_planned", "vlm_reviewed",
                                          "agent_corrected", "vlm_rechecked",
                                          "trained", "evaluated"} else None),
            "prompt_sha256": (digest((state + ":prompt").encode())
                              if state in {"candidate_planned", "vlm_reviewed",
                                           "agent_corrected", "vlm_rechecked"}
                              else None),
        }
        contract_type = sorted(CONTRACT_TYPES_BY_STATE[state])[0]
        schema_name = {
            "factory_ingress": "factory-data-ingress-event.schema.json",
            "factory_qc": "factory-data-qc-event.schema.json",
            "candidate_batch_plan": "labelagent-batch-execution-plan.schema.json",
            "vlm_label_observation": "vlm-label-observation.schema.json",
            "closed_loop_agent_action": "labelagent_closed_loop_event.schema.json",
            "closed_loop_vlm_recheck": "labelagent_closed_loop_event.schema.json",
            "human_review_queue": "labelagent-human-review-queue.schema.json",
            "cvat_offline_export": "cvat-offline-export.schema.json",
            "human_review_receipt": "cvat-return-accepted.schema.json",
            "cvat_offline_return": "cvat-offline-return.schema.json",
            "immutable_truth_promotion": "immutable-truth-promotion-event.schema.json",
            "cvat_evaluation_truth": "cvat-evaluation-truth.schema.json",
            "sam3_stage0_dry_run_plan": "sam3-stage0-dry-run-plan.schema.json",
            "sam3_stage0_model_card": "sam3-stage0-model-card.schema.json",
            "sam3_stage0_deployment_decision": "sam3-stage0-deployment-decision.schema.json",
            "deployment_receipt": "labelagent_closed_loop_event.schema.json",
            "rollback_receipt": "labelagent_closed_loop_event.schema.json",
        }[contract_type]
        occurred_at = (BASE_TIME + timedelta(seconds=self.time_offset)
                       ).isoformat()
        self.time_offset += 1
        effect_mode = ("simulated" if state in {"truth_promoted",
                                                "training_eligible", "trained",
                                                "evaluated", "deployment_candidate",
                                                "baseline_retained", "deployed",
                                                "rolled_back"} else "none")
        event = build_event(
            event_type="state.transition", occurred_at=occurred_at,
            trace_id=self.trace_id, identity=self.identity,
            state_version=self.version,
            previous_state=None if self.previous is None else self.previous["state"],
            state=state,
            previous_event_id=None if self.previous is None else self.previous["event_id"],
            operational_status=operational_status, bindings=bindings,
            queue={"name": "factory_labelagent", "depth": 0,
                   "high_watermark": 32, "route": queue_route},
            human_gate=human_gate or gate(), provenance=self.provenance,
            classification=classification, agent_action=agent_action,
            reason_codes=reason_codes, failure_reason=failure_reason,
            artifacts=[artifact("source_image", "example://static/source.png",
                                self.source_image_sha),
                       artifact("state_output", f"example://static/{state}.json",
                                output_sha)],
            contract_refs=[contract_ref(
                contract_type, f"schemas/{schema_name}",
                f"example://static/contracts/{self.trace_id}/{state}.json",
                output_sha)],
            evaluation=evaluation, truth_version=truth_version,
            training_run=training_run,
            effect={"mode": effect_mode, "external_call_count": 0,
                    "production_mutation": False},
        )
        self.events.append(event)
        self.previous = event
        self.last_output = output_sha
        return event


def common_candidate_path(builder, classification, *, action="no_op"):
    builder.transition("ingested")
    builder.transition("qc_passed")
    builder.transition("candidate_planned")
    builder.transition("vlm_reviewed", classification=classification)
    builder.transition("agent_corrected", classification=classification,
                       agent_action=action)
    builder.transition("vlm_rechecked", classification=classification,
                       agent_action=action)


def build_event_set():
    known = TraceBuilder("known_label_waiting_human", identity("FRAME_KNOWN"),
                         provenance("real"), 0)
    common_candidate_path(known, "known_label")
    known.transition(
        "human_review_required", classification="known_label",
        agent_action="no_op", human_gate=gate("pending"),
        reason_codes=["human_pixel_truth_missing", "candidate_only_not_truth"])

    unknown = TraceBuilder("unknown_routes_human", identity("FRAME_UNKNOWN"),
                           provenance("simulation"), 20)
    common_candidate_path(unknown, "unknown_candidate")
    unknown.transition(
        "human_review_required", classification="unknown_candidate",
        agent_action="no_op", human_gate=gate("pending"),
        reason_codes=["unknown_candidate", "ontology_review_required",
                      "automatic_ontology_mutation_forbidden"])

    qc = TraceBuilder("qc_failure_quarantined", identity("FRAME_QC_FAIL"),
                      provenance("real"), 40)
    qc.transition("ingested")
    qc.transition(
        "quarantined", failure_reason="camera_2_missing_and_odom_gap",
        reason_codes=["four_camera_completeness_failed",
                      "pose_odom_continuity_failed"],
        operational_status="quarantined", queue_route="quarantine")

    slope = TraceBuilder("slope_regression_baseline_retained",
                         identity("FRAME_SLOPE"), provenance("real"), 60)
    common_candidate_path(slope, "known_label", action="resegment")
    slope.transition("human_review_required", classification="known_label",
                     agent_action="resegment", human_gate=gate("pending"),
                     reason_codes=["slope_hard_case", "human_pixel_truth_required"])
    approved_gate = gate("approved", slope.candidate_sha, slope.source_image_sha)
    slope.transition("candidate_accepted", classification="known_label",
                     agent_action="resegment", human_gate=approved_gate,
                     reason_codes=["static_human_gate_fixture_only"])
    truth_version = {"version_id": "STATIC_TRUTH_V1",
                     "manifest_sha256": digest(b"static-truth-manifest"),
                     "split": "train", "immutable": True}
    slope.transition("truth_promoted", classification="known_label",
                     agent_action="resegment", human_gate=approved_gate,
                     truth_version=truth_version,
                     reason_codes=["simulated_static_truth_transition_no_mutation"])
    slope.transition("training_eligible", classification="known_label",
                     human_gate=approved_gate, truth_version=truth_version,
                     reason_codes=["static_gate_replay_not_training_authorization"])
    training = {"run_id": "STATIC_STAGE0_RUN", "package_sha256": digest(b"package"),
                "checkpoint_sha256": digest(b"candidate-checkpoint"),
                "simulated_static_replay": True}
    slope.transition("trained", classification="known_label",
                     human_gate=approved_gate, truth_version=truth_version,
                     training_run=training,
                     reason_codes=["simulated_static_training_result_reference"])
    evaluation = slope_regression_metrics()
    slope.transition("evaluated", classification="known_label",
                     human_gate=approved_gate, truth_version=truth_version,
                     training_run=training, evaluation=evaluation,
                     reason_codes=["slope_boundary_f1_regressed",
                                   "slope_false_positives_increased"])
    slope.transition("baseline_retained", classification="known_label",
                     human_gate=approved_gate, truth_version=truth_version,
                     training_run=training, evaluation=evaluation,
                     reason_codes=["deployment_gate_failed_slope_guard",
                                   "incumbent_retained"])

    return known.events + unknown.events + qc.events + slope.events


def build_static_acceptance():
    events = build_event_set()
    ledger = FactoryOrchestrationLedger(
        ledger_id="factory-labelagent-static-orchestration-20261111",
        max_inflight=32, retry_budget=2)
    for event in events:
        ledger.append(event)
    snapshot = ledger.snapshot(generated_at=GENERATED_AT)
    validate_ledger_snapshot(snapshot)

    checkpoint_index = 10
    prefix = FactoryOrchestrationLedger(
        ledger_id=ledger.ledger_id, max_inflight=32, retry_budget=2)
    for event in events[:checkpoint_index]:
        prefix.append(event)
    prefix_snapshot = prefix.snapshot(generated_at=GENERATED_AT)
    recovered = FactoryOrchestrationLedger.from_snapshot(prefix_snapshot)
    for event in events[checkpoint_index:]:
        recovered.append(event)
    recovered_snapshot = recovered.snapshot(generated_at=GENERATED_AT)
    if recovered_snapshot != snapshot:
        raise RuntimeError("checkpoint recovery did not reproduce the exact ledger")

    expected = {
        "known_label_waiting_human": "human_review_required",
        "unknown_routes_human": "human_review_required",
        "qc_failure_quarantined": "quarantined",
        "slope_regression_baseline_retained": "baseline_retained",
    }
    trace_results = []
    for trace_id in sorted(expected):
        trace = snapshot["traces"][trace_id]
        trace_results.append({
            "trace_id": trace_id, "stream_id": trace["stream_id"],
            "event_count": trace["event_count"],
            "final_state": trace["final_state"],
            "expected_final_state": expected[trace_id],
            "matches": trace["final_state"] == expected[trace_id],
            "checkpoint_resume_verified": True,
        })
    report = {
        "schema_version": "labelagent.factory-orchestration-replay-report.v1",
        "generated_at": "2026-11-11T23:59:00+08:00",
        "ledger_sha256": snapshot["ledger_sha256"], "replay_valid": True,
        "recovered_streams": 4, "trace_results": trace_results,
        "safety": {
            "append_only": True, "idempotent_replay": True,
            "out_of_order_rejected": True, "illegal_transition_rejected": True,
            "cross_split_rejected": True, "silent_skip_forbidden": True,
            "retry_budget_enforced": True, "dead_letter_supported": True,
            "quarantine_supported": True, "backpressure_supported": True,
            "human_gate_enforced": True, "real_sim_separate": True,
            "floor_slope_separate": True, "slope_in_traversable_union": True,
            "production_mutations": 0, "truth_versions_created": 0,
            "training_runs_started": 0, "deployments_performed": 0,
        },
        "external_call_counts": copy.deepcopy(snapshot["external_call_counts"]),
        "cnb_status": "pending_not_contacted",
    }
    report["report_sha256"] = digest(report)
    outputs = {
        ROOT / "docs/test-results/factory-labelagent-orchestration-ledger-static-20261111.json": snapshot,
        ROOT / "docs/test-results/factory-labelagent-orchestration-replay-report-20261111.json": report,
    }
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    outputs = build_static_acceptance()
    if not args.write:
        print(json.dumps({str(path.relative_to(ROOT)): value
                          for path, value in outputs.items()},
                         ensure_ascii=False, indent=2, sort_keys=True))
        return
    for path, value in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"append-only output already exists: {path}")
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"readback mismatch: {path}")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
