"""Reconcile SAM3 pseudo-truth projection training with frozen test24 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(paths: dict[str, Path]) -> dict[str, Any]:
    values = {name: read(path.resolve()) for name, path in paths.items()}
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    rounds = [values["round54"], values["round81"]]
    expected_steps = [54, 81]
    for index, (receipt, steps) in enumerate(zip(rounds, expected_steps), 1):
        prefix = f"round{index}"
        require(receipt.get("schema_version") == "labelagent.sam3-projection-night-round.v1",
                f"{prefix}: schema differs")
        require(receipt.get("candidate_only") is True, f"{prefix}: not candidate-only")
        require(receipt.get("pseudo_truth") is True, f"{prefix}: not pseudo-truth")
        require(receipt.get("not_human_truth") is True, f"{prefix}: human-truth boundary differs")
        require(receipt.get("not_for_deployment") is True, f"{prefix}: deployment boundary differs")
        require(receipt.get("deployment_performed") is False, f"{prefix}: deployment occurred")
        require(receipt.get("test24_in_training") is False, f"{prefix}: test24 entered training")
        require(receipt.get("cumulative_optimizer_steps") == steps,
                f"{prefix}: optimizer steps differ")
        require(receipt.get("exit_code") == 0 and receipt.get("runtime_failure") is None,
                f"{prefix}: training did not finish cleanly")

    comparisons = [values["compare54"], values["compare81"]]
    for index, receipt in enumerate(comparisons, 1):
        prefix = f"compare{index}"
        require(receipt.get("evaluation_only") is True, f"{prefix}: not evaluation-only")
        require(receipt.get("machine_reference_only") is True,
                f"{prefix}: reference boundary differs")
        require(receipt.get("test24_training_return_allowed") is False,
                f"{prefix}: test24 return allowed")
        require(receipt.get("not_human_truth") is True, f"{prefix}: human truth claimed")
        require(receipt.get("not_for_deployment") is True, f"{prefix}: deployment allowed")
        require(receipt.get("deployment_decision") == "forbidden_without_human_truth",
                f"{prefix}: deployment decision differs")

    three = values["threeway"]
    review = values["review81"]
    require(three.get("evaluation_only") is True, "three-way: not evaluation-only")
    require(three.get("machine_reference_only") is True,
            "three-way: reference boundary differs")
    require(three.get("test24_training_steps") == 0,
            "three-way: test24 optimizer steps are nonzero")
    require(three.get("not_human_truth") is True, "three-way: human truth claimed")
    require(three.get("not_for_deployment") is True, "three-way: deployment allowed")
    require(review.get("pseudo_truth") is True, "review: pseudo-truth boundary differs")
    require(review.get("long_training_gate") == "blocked", "review: long gate is not blocked")
    require(review.get("not_human_truth") is True, "review: human truth claimed")
    require(review.get("not_for_deployment") is True, "review: deployment allowed")

    means = three.get("means") or {}
    base_iou = (means.get("base") or {}).get("union_iou")
    best_iou = (means.get("best") or {}).get("union_iou")
    current_iou = (means.get("current") or {}).get("union_iou")
    base_boundary = (means.get("base") or {}).get("boundary_f1")
    current_boundary = (means.get("current") or {}).get("boundary_f1")
    numeric = all(isinstance(value, (int, float)) for value in
                  (base_iou, best_iou, current_iou, base_boundary, current_boundary))
    require(numeric, "three-way aggregate metrics missing")
    if numeric:
        require(current_iou < base_iou, "expected current regression is absent")
        require(best_iou < base_iou, "expected best projection regression is absent")

    catastrophic = review.get("catastrophic_frames")
    require(isinstance(catastrophic, list), "review: catastrophic frame list missing")

    return {
        "schema_version": "labelagent.sam3-projection-training-audit.v1",
        "status": "passed_negative_result_audit" if not errors else "failed",
        "errors": errors,
        "training": {
            "kind": "projection_filtered_pseudo_truth",
            "human_truth": False,
            "candidate_only": True,
            "deployed": False,
            "rounds": [
                {"optimizer_steps": receipt.get("cumulative_optimizer_steps"),
                 "checkpoint_sha256": receipt.get("checkpoint_sha256"),
                 "elapsed_seconds": receipt.get("elapsed_seconds")}
                for receipt in rounds
            ],
        },
        "frozen_test24": {
            "images": three.get("images"),
            "optimizer_steps": three.get("test24_training_steps"),
            "machine_reference_only": True,
            "training_return_allowed": False,
            "base_mean_union_iou": base_iou,
            "best_projection_mean_union_iou": best_iou,
            "step81_mean_union_iou": current_iou,
            "step81_minus_base_iou": (current_iou - base_iou) if numeric else None,
            "step81_minus_best_iou": (current_iou - best_iou) if numeric else None,
            "base_mean_boundary_f1": base_boundary,
            "step81_mean_boundary_f1": current_boundary,
            "catastrophic_frames": catastrophic if isinstance(catastrophic, list) else None,
        },
        "decision": {
            "keep_incumbent_base_sam3": True,
            "deploy_projection_candidate": False,
            "continue_pseudo_truth_long_training": False,
            "human_truth_microtuning_completed": False,
            "human_truth_microtuning_next_gate":
                "approved train masks plus frozen independent evaluation",
            "reason": "Both projection checkpoints regress against the frozen base; step81 also has a catastrophic frame.",
        },
        "source_sha256": {name: sha256(path.resolve()) for name, path in paths.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("round54", "round81", "compare54", "compare81", "threeway", "review81"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in
             ("round54", "round81", "compare54", "compare81", "threeway", "review81")}
    value = audit(paths)
    output = args.output.resolve()
    if output.exists():
        raise ValueError("choose a fresh output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False))
    return 0 if value["status"] == "passed_negative_result_audit" else 1


if __name__ == "__main__":
    raise SystemExit(main())
