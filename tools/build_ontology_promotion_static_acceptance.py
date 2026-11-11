"""Build deterministic ontology contract fixtures without model/GPU/remote calls."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.agent.factory_data_contract import digest
from source.agent.ontology_promotion_contract import (
    build_cluster_proposal,
    build_label_observation,
    build_ontology_ledger,
    evaluate_ontology_promotion,
)


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def artifact(name):
    return {"artifact_id": digest({"artifact": name}),
            "uri": "example://ontology-static/" + name,
            "sha256": digest(("bytes:" + name).encode())}


def observation_payload(source_kind="real", outcome="unknown_candidate",
                        confidence=0.73, sam_relation="conflict"):
    simulation = None if source_kind == "real" else {
        "simulator": "STATIC_SIMULATOR", "scene": "STATIC_SCENE",
        "seed": "17", "version": "STATIC_V1"}
    views = [{"camera": f"camera_{index}", "artifact": artifact(f"{source_kind}/camera_{index}.png")}
             for index in range(3)]
    return {
        "identity": {"dataset_id": "STATIC_DATASET", "source_batch": "STATIC_BATCH",
                     "sequence": "000", "frame_id": "STATIC_FRAME_001",
                     "input_id": digest(("input:" + source_kind).encode())},
        "provenance": {"source_kind": source_kind, "simulation": simulation},
        "inputs": {
            "views": views, "candidate": artifact(f"{source_kind}/candidate.json"),
            "model": {"id": "STATIC_VLM", "version": "STATIC_V1",
                      "sha256": digest(b"static-vlm-model")},
            "prompt": {"id": "label-discovery", "version": "STATIC_V1",
                       "sha256": digest(b"static-label-discovery-prompt")},
        },
        "result": {
            "outcome": outcome, "proposed_name": "loading_dock_guard",
            "aliases": ["dock_guard", "loading_bay_guard"],
            "nearest_known_labels": [{"label": "pallet", "similarity": 0.31,
                                      "difference": "fixed protective structure, not cargo"}],
            "evidence": {"summary": "upright protective structure appears across views",
                         "visual_cues": ["fixed base", "high-visibility surface"],
                         "view_artifact_ids": [view["artifact"]["artifact_id"] for view in views]},
            "confidence": confidence,
            "multiview_consistency": {"status": "consistent", "agreement": 0.89,
                                      "supporting_cameras": ["camera_0", "camera_1", "camera_2"],
                                      "conflicting_cameras": []},
            "sam_relation": sam_relation,
        },
    }


def approval_request(parent, proposal, *, valid=True):
    return {
        "proposal_event_id": proposal["event_id"],
        "human_decision": {
            "decision": "approved_create_new_label" if valid else "machine_suggested",
            "canonical_name": "loading_dock_guard", "definition": "Fixed protective structure at a loading dock.",
            "class_id": 8, "reviewer_id": "STATIC_AUTHENTICATED_REVIEWER",
            "identity_assurance": "authenticated" if valid else "machine",
            "reviewed_at": "2026-11-11T20:05:00+08:00",
            "approval_id": "STATIC_ONTOLOGY_APPROVAL_V1",
            "receipt_uri": "example://ontology-static/approval.json",
            "receipt_sha256": digest(b"static-ontology-approval"),
        },
        "regression_validation": {
            "fixed_set_id": "STATIC_FIXED_REGRESSION_V1",
            "manifest_uri": "example://ontology-static/regression-manifest.json",
            "manifest_sha256": digest(b"static-regression-manifest"),
            "config_sha256": digest(b"static-regression-config"),
            "real_metrics": {"sample_count": 12, "precision": 0.92, "recall": 0.83,
                             "false_positive_rate": 0.04},
            "simulation_metrics": {"sample_count": 8, "precision": 0.88, "recall": 0.75,
                                   "false_positive_rate": 0.06},
            "truth_versions": {"real": "STATIC_REAL_TRUTH_V1",
                               "simulation": "STATIC_SIM_TRUTH_V1"},
            "all_required_gates_passed": True,
            "validated_at": "2026-11-11T20:04:00+08:00",
        },
        "version": {"registry_version": "labelagent-ontology-STATIC-v2",
                    "parent_registry_version": parent["registry_version"],
                    "parent_registry_sha256": parent["registry_content_sha256"]},
    }


def build_static_acceptance():
    registry = load(ROOT / "configs/labelagent_label_registry.json")
    real = build_label_observation(observation_payload("real"), registry,
                                   observed_at="2026-11-11T20:00:00+08:00")
    simulation_payload = observation_payload("simulation", confidence=0.69,
                                             sam_relation="insufficient_evidence")
    simulation_payload["identity"]["frame_id"] = "STATIC_SIM_FRAME_001"
    simulation = build_label_observation(simulation_payload, registry,
                                         observed_at="2026-11-11T20:00:01+08:00")
    cluster = build_cluster_proposal(
        [real, simulation], cluster_id="STATIC_CLUSTER_LOADING_DOCK_GUARD",
        config={"algorithm": "deterministic_evidence_alias_cluster",
                "version": "STATIC_V1", "config_sha256": digest(b"static-cluster-config")},
        proposed_at="2026-11-11T20:02:00+08:00")
    promoted = evaluate_ontology_promotion(
        registry, cluster, approval_request(registry, cluster),
        occurred_at="2026-11-11T20:06:00+08:00")
    bad = approval_request(registry, cluster, valid=False)
    bad["version"]["registry_version"] = "labelagent-ontology-STATIC-rejected"
    rejected = evaluate_ontology_promotion(
        registry, cluster, bad, occurred_at="2026-11-11T20:06:01+08:00")
    ledger = build_ontology_ledger(
        [real, simulation, cluster, promoted, rejected],
        ledger_id="ontology-static-audit-ledger-20261111",
        generated_at="2026-11-11T20:07:00+08:00",
        sample_kind="synthetic_contract_fixture_not_production_ontology")
    outputs = {
        ROOT / "examples/vlm_unknown_candidate_multiview_static.json": real,
        ROOT / "examples/ontology_unknown_cluster_proposal_static.json": cluster,
        ROOT / "examples/ontology_registry_promotion_positive_static.json": promoted,
        ROOT / "examples/ontology_promotion_machine_rejected_static.json": rejected,
        ROOT / "docs/test-results/ontology-promotion-audit-ledger-static-20261111.json": ledger,
    }
    receipt = {
        "schema_version": "labelagent.ontology-static-validation.v1",
        "generated_at": "2026-11-11T20:08:00+08:00",
        "sample_kind": "synthetic_contract_fixture_not_production_ontology",
        "execution_boundary": {"cpu_only": True, "vlm_called": False,
                               "gpu_called": False, "remote_service_called": False,
                               "cvat_or_oss_modified": False},
        "contracts": {"unknown_routes_to_human": True,
                      "low_confidence_routes_to_human": True,
                      "sam_conflict_routes_to_human": True,
                      "clustering_is_proposal_only": True,
                      "human_approval_and_regression_required": True,
                      "old_registry_preserved": True,
                      "real_sim_accounting_separate": True,
                      "floor_slope_distinct": True,
                      "slope_downstream_traversable": True,
                      "replay_planned_not_executed": True},
        "artifact_content_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): digest(value)
            for path, value in outputs.items()},
        "production_registry_versions_created": 0,
        "production_replays_executed": 0,
        "cnb_status": "pending_not_contacted",
    }
    receipt["receipt_sha256"] = digest({key: value for key, value in receipt.items()
                                        if key != "receipt_sha256"})
    outputs[ROOT / "docs/test-results/ontology-promotion-static-validation-20261111.json"] = receipt
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    outputs = build_static_acceptance()
    if not args.write:
        print(json.dumps({str(path.relative_to(ROOT)): value for path, value in outputs.items()},
                         ensure_ascii=False, indent=2))
        return
    for path, value in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"append-only output already exists: {path}")
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"readback mismatch: {path}")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
