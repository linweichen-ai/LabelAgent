#!/usr/bin/env python3
"""Unify LabelAgent post-training evidence without promoting machine labels.

The audit distinguishes an executed teacher-distillation LoRA run from a
human-supervised training-ready dataset.  It is read-only and fails closed on
test leakage, adapter mismatch, deployment claims, or hard examples that are
still machine candidates.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_posttraining_readiness(
    training_gate: dict[str, Any],
    lora_receipt: dict[str, Any],
    lora_audit: dict[str, Any],
    external_eval: dict[str, Any],
    hard_examples: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    boundaries = lora_audit.get("boundaries") or {}
    receipt_adapter = lora_receipt.get("adapter_sha256")
    audit_adapter = lora_audit.get("adapter_sha256")
    eval_adapter = external_eval.get("adapter_sha256")

    require(lora_receipt.get("status") == "training_complete",
            "LoRA receipt does not prove completed training")
    require(lora_receipt.get("track") == "teacher_distillation",
            "LoRA receipt is not bound to teacher distillation")
    require(lora_receipt.get("human_truth") is False,
            "LoRA receipt must not claim human truth")
    require(lora_receipt.get("production_deployed") is False,
            "LoRA receipt unexpectedly claims production deployment")
    require(lora_receipt.get("test24_used_for_training") is False,
            "test24 leaked into LoRA training")
    require(lora_audit.get("status") == "passed",
            "LoRA tensor audit did not pass")
    require(boundaries.get("test24_training_false") is True,
            "LoRA audit does not prove test24 exclusion")
    require(boundaries.get("human_truth_false") is True,
            "LoRA audit does not preserve the teacher-label boundary")
    require(bool(receipt_adapter), "LoRA receipt has no adapter hash")
    require(receipt_adapter == audit_adapter,
            "LoRA receipt and tensor audit adapter hashes differ")
    require(receipt_adapter == eval_adapter,
            "LoRA receipt and external evaluation adapter hashes differ")

    require(external_eval.get("status") == "external_evaluated_not_deployed",
            "external evaluation status is not frozen and undeployed")
    require(external_eval.get("split") == "test24",
            "external evaluation is not bound to test24")
    require(external_eval.get("evaluation_only") is True,
            "test24 is not marked evaluation-only")
    require(external_eval.get("training_return_allowed") is False,
            "test24 is allowed to return to training")
    require(external_eval.get("human_truth") is False,
            "teacher comparison must not claim human accuracy")
    require(external_eval.get("production_deployed") is False,
            "external evaluation unexpectedly claims deployment")

    require(hard_examples.get("candidate_only") is True,
            "hard-example report is not explicitly candidate-only")
    require(hard_examples.get("human_truth") is False,
            "hard-example report unexpectedly claims human truth")
    require(hard_examples.get("approved_for_training") is False,
            "hard-example report unexpectedly permits training")
    require(hard_examples.get("production_deployed") is False,
            "hard-example report unexpectedly permits deployment")
    require(hard_examples.get("reference_labels_used") is False,
            "hard-example diagnostic read reference labels")
    require(hard_examples.get("workbench_writes") == 0,
            "hard-example diagnostic wrote to the workbench")

    gate_ready = training_gate.get("training_authorized") is True
    if gate_ready:
        require(training_gate.get("status") == "ready_for_isolated_training",
                "authorized training gate has an inconsistent status")
    else:
        require(training_gate.get("status") == "blocked_waiting_for_human_review",
                "blocked training gate has an inconsistent status")
    require(training_gate.get("test_split_policy") ==
            "evaluation_only_never_train",
            "training gate does not permanently isolate test24")
    require(training_gate.get("machine_preannotations_are_ground_truth") is False,
            "training gate promotes machine preannotations to truth")

    summary = external_eval.get("summary") or {}
    lora_candidate_valid = not errors
    sam3_ready = gate_ready and not errors
    if errors:
        status = "invalid_or_inconsistent_evidence"
    elif sam3_ready:
        status = "human_supervised_training_ready_for_isolated_run"
    else:
        status = "offline_teacher_lora_exists_human_supervised_training_blocked"

    return {
        "schema_version": "labelagent.posttraining_readiness.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "integrity_errors": errors,
        "vlm_teacher_lora": {
            "training_executed": lora_receipt.get("status") == "training_complete",
            "candidate_valid": lora_candidate_valid,
            "human_truth": False,
            "production_deployed": False,
            "test24_used_for_training": False,
            "samples_loaded": lora_receipt.get("samples_loaded"),
            "steps": lora_receipt.get("steps"),
            "adapter_sha256": receipt_adapter,
            "test24_teacher_evidence_accuracy":
                summary.get("teacher_evidence_accuracy"),
            "test24_teacher_evidence_matches":
                summary.get("teacher_evidence_matches"),
            "test24_teacher_evidence_total":
                summary.get("teacher_evidence_total"),
        },
        "human_supervised_training": {
            "training_authorized": gate_ready and not errors,
            "gate_status": training_gate.get("status"),
            "checklist": training_gate.get("checklist"),
            "human_exports": training_gate.get("human_exports"),
            "blockers": training_gate.get("blockers") or [],
            "test_split_policy": training_gate.get("test_split_policy"),
        },
        "distant_pallet_hard_examples": {
            "selected_count": hard_examples.get("selected_count"),
            "safe_expansion_count": hard_examples.get("safe_expansion_count"),
            "eligible_for_training": 0,
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
        },
        "sam3_pixel_finetune_ready": sam3_ready,
        "production_model_change_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-gate", required=True, type=Path)
    parser.add_argument("--lora-receipt", required=True, type=Path)
    parser.add_argument("--lora-audit", required=True, type=Path)
    parser.add_argument("--external-eval", required=True, type=Path)
    parser.add_argument("--hard-examples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    inputs = {
        "training_gate": args.training_gate.resolve(),
        "lora_receipt": args.lora_receipt.resolve(),
        "lora_audit": args.lora_audit.resolve(),
        "external_eval": args.external_eval.resolve(),
        "hard_examples": args.hard_examples.resolve(),
    }
    report = audit_posttraining_readiness(
        _read(inputs["training_gate"]),
        _read(inputs["lora_receipt"]),
        _read(inputs["lora_audit"]),
        _read(inputs["external_eval"]),
        _read(inputs["hard_examples"]),
    )
    report["input_sha256"] = {
        name: _sha256(path) for name, path in inputs.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["integrity_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
