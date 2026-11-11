"""Fail-closed resource admission for the protected 4090 and experimental 5090 roles.

The tool is read-only: it evaluates a supplied live snapshot and never starts,
stops, or signals a process.  PID ownership is not trusted on its own; an owned
service is recognized only when its observed command also contains the declared
command marker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _unique(items):
    return list(dict.fromkeys(items))


def evaluate(snapshot):
    snapshot_content_sha256 = hashlib.sha256(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    services = snapshot["dual_5090"]["services"]
    processes = snapshot["dual_5090"]["processes"]
    owned_pids = set()
    service_by_role = {}
    for service in services:
        matches = [
            process for process in processes
            if process["pid"] == service["pid"]
            and service["command_contains"] in process["command"]
            and process["gpu_index"] == service["gpu_index"]
        ]
        verified = len(matches) == 1 and service["healthy"]
        service_by_role[service["role"]] = verified
        if verified:
            owned_pids.add(service["pid"])

    foreign_by_gpu = {0: [], 1: []}
    for process in processes:
        if process["pid"] not in owned_pids:
            foreign_by_gpu[process["gpu_index"]].append(process)

    gates = snapshot["training_gates"]

    def decision(blockers):
        blockers = _unique(blockers)
        return {"allowed": not blockers, "blockers": blockers}

    gpu0_foreign = [
        f"foreign_compute_process:{item['pid']}:{item['owner']}"
        for item in foreign_by_gpu[0]
    ]
    gpu1_foreign = [
        f"foreign_compute_process:{item['pid']}:{item['owner']}"
        for item in foreign_by_gpu[1]
    ]
    training_blockers = []
    if not gates["genuine_human_truth_version_available"]:
        training_blockers.append("genuine_human_truth_version_missing")
    if not gates["fixed_independent_holdout_baseline_available"]:
        training_blockers.append("fixed_independent_holdout_baseline_missing")
    if not gates["formal_training_authorized"]:
        training_blockers.append("formal_training_not_authorized")
    if not gates["owned_service_drain_acknowledged"]:
        training_blockers.append("owned_5090_services_not_drained")

    decisions = {
        "new_work_on_factory_4090": decision(["factory_4090_reserved_for_baseline_services"]),
        "sam3_candidate_batch_via_gpu0_service": decision(
            ([] if service_by_role.get("sam3") else ["sam3_service_unhealthy_or_unverified"])
            + gpu0_foreign
        ),
        "labelagent_batch_via_gpu1_service": decision(
            ([] if service_by_role.get("vlm") else ["vlm_service_unhealthy_or_unverified"])
            + gpu1_foreign
        ),
        "sam3_finetune_gpu0": decision(gpu0_foreign + training_blockers),
        "dual_5090_training": decision(
            gpu0_foreign
            + gpu1_foreign
            + training_blockers
            + ([] if gates["dual_card_qualifier_present"] else ["dual_card_qualifier_missing"])
        ),
    }
    return {
        "schema_version": "labelagent.gpu-resource-admission-decision.v1",
        "observed_at": snapshot["observed_at"],
        "source_snapshot_id": snapshot["snapshot_id"],
        "source_snapshot_content_sha256": snapshot_content_sha256,
        "foreign_processes": {
            "gpu0": foreign_by_gpu[0],
            "gpu1": foreign_by_gpu[1],
        },
        "decisions": decisions,
        "mutation_performed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    print(json.dumps(evaluate(snapshot), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
