#!/usr/bin/env python3
"""Fail-closed release gate for LabelAgent post-training candidates.

This tool does not train or deploy a model.  It binds the reviewed training
data, training receipt, evaluation protocol and rollback artifact so that a
completed run cannot be confused with an approved production replacement.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


HEX64 = set("0123456789abcdef")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value.lower()) <= HEX64
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate_posttraining_release(manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    boundaries = manifest.get("boundaries") or {}
    dataset = manifest.get("dataset") or {}
    training = manifest.get("training") or {}
    baseline = manifest.get("baseline_eval") or {}
    candidate = manifest.get("candidate_eval") or {}
    rollback = manifest.get("rollback") or {}
    thresholds = manifest.get("thresholds") or {}

    require(boundaries.get("test24_evaluation_only") is True,
            "test24 must be evaluation-only")
    require(boundaries.get("test24_used_for_training") is False,
            "test24 leaked into training")
    require(boundaries.get("machine_candidates_are_human_truth") is False,
            "machine candidates were promoted to human truth")
    require(boundaries.get("production_deployed") is False,
            "manifest unexpectedly claims production deployment")

    total = dataset.get("sample_count")
    reviewed = dataset.get("human_reviewed_count")
    approved = dataset.get("approved_for_training_count")
    require(isinstance(total, int) and total > 0,
            "training dataset sample_count must be positive")
    require(total == reviewed == approved,
            "every training sample must be human-reviewed and approved")
    require(dataset.get("candidate_only") is False,
            "candidate-only data cannot pass the training gate")
    require(dataset.get("human_truth") is True,
            "training dataset is not bound to human truth")
    require(dataset.get("contains_test24") is False,
            "training dataset contains test24")
    require(_is_sha256(dataset.get("manifest_sha256")),
            "dataset manifest hash is missing or invalid")

    require(training.get("status") == "training_complete",
            "training receipt does not prove completion")
    for key, label in (
        ("dataset_manifest_sha256", "training dataset hash"),
        ("config_sha256", "training config hash"),
        ("log_sha256", "training log hash"),
        ("adapter_sha256", "adapter hash"),
    ):
        require(_is_sha256(training.get(key)), f"{label} is missing or invalid")
    require(training.get("dataset_manifest_sha256") ==
            dataset.get("manifest_sha256"),
            "training receipt is not bound to the reviewed dataset")
    require(isinstance(training.get("steps"), int) and training.get("steps") > 0,
            "training step count is missing")
    gpu = training.get("gpu") or {}
    require(isinstance(gpu.get("device_count"), int) and
            gpu.get("device_count") > 0, "GPU usage receipt is missing")
    require(bool(gpu.get("devices")), "GPU device identity is missing")
    require(isinstance(gpu.get("max_memory_mib"), (int, float)) and
            gpu.get("max_memory_mib") > 0, "GPU memory evidence is missing")

    for name, evaluation in (("baseline", baseline), ("candidate", candidate)):
        require(evaluation.get("status") == "complete",
                f"{name} evaluation is incomplete")
        require(evaluation.get("evaluation_only") is True,
                f"{name} evaluation is not evaluation-only")
        require(_is_sha256(evaluation.get("protocol_sha256")),
                f"{name} evaluation protocol hash is invalid")
        require(_is_sha256(evaluation.get("sample_manifest_sha256")),
                f"{name} sample manifest hash is invalid")
        require(evaluation.get("human_reference") is True,
                f"{name} evaluation lacks human reference")
        require(evaluation.get("test24_returned_to_training") is False,
                f"{name} evaluation allows test24 to return to training")

    require(baseline.get("protocol_sha256") == candidate.get("protocol_sha256"),
            "baseline and candidate use different protocols")
    require(baseline.get("sample_manifest_sha256") ==
            candidate.get("sample_manifest_sha256"),
            "baseline and candidate use different samples")
    require(training.get("adapter_sha256") == candidate.get("adapter_sha256"),
            "candidate evaluation is not bound to the trained adapter")

    bm = baseline.get("metrics") or {}
    cm = candidate.get("metrics") or {}
    min_f1_delta = thresholds.get("min_overall_f1_delta", 0.0)
    max_slope_fp_delta = thresholds.get("max_slope_fp_delta", 0)
    require(isinstance(bm.get("overall_f1"), (int, float)) and
            isinstance(cm.get("overall_f1"), (int, float)),
            "overall F1 is missing")
    require(isinstance(bm.get("slope_false_positives"), int) and
            isinstance(cm.get("slope_false_positives"), int),
            "slope false-positive counts are missing")
    if isinstance(bm.get("overall_f1"), (int, float)) and isinstance(
            cm.get("overall_f1"), (int, float)):
        require(cm["overall_f1"] - bm["overall_f1"] >= min_f1_delta,
                "overall F1 improvement does not meet the threshold")
    if isinstance(bm.get("slope_false_positives"), int) and isinstance(
            cm.get("slope_false_positives"), int):
        require(cm["slope_false_positives"] -
                bm["slope_false_positives"] <= max_slope_fp_delta,
                "slope false positives exceed the guard")

    require(_is_sha256(rollback.get("incumbent_artifact_sha256")),
            "incumbent rollback artifact hash is missing")
    require(bool(rollback.get("restore_command")),
            "rollback restore command is missing")
    require(rollback.get("verified") is True,
            "rollback path has not been verified")

    if not errors:
        warnings.append(
            "Gate passed for isolated candidate review only; deployment still "
            "requires an explicit independent decision.")

    return {
        "schema_version": "labelagent.posttraining_release_gate.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "eligible_for_isolated_release_review" if not errors
                  else "blocked",
        "ready": not errors,
        "errors": errors,
        "warnings": warnings,
        "production_deployment_allowed": False,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "summary": {
            "training_samples": total,
            "human_reviewed_samples": reviewed,
            "approved_training_samples": approved,
            "overall_f1_delta": (
                cm.get("overall_f1") - bm.get("overall_f1")
                if isinstance(cm.get("overall_f1"), (int, float)) and
                isinstance(bm.get("overall_f1"), (int, float)) else None
            ),
            "slope_false_positive_delta": (
                cm.get("slope_false_positives") -
                bm.get("slope_false_positives")
                if isinstance(cm.get("slope_false_positives"), int) and
                isinstance(bm.get("slope_false_positives"), int) else None
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = gate_posttraining_release(manifest)
    report["input_manifest_sha256"] = _sha256(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
