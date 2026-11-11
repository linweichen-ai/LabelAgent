"""One-shot, read-only, no-autostart preflight for formal dual-5090 DDP.

Passing this preflight never creates a process.  It only permits a separate,
explicit launch step after every fail-closed condition is satisfied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


EXPECTED_STAGE0_NAMES = (
    "segmentation_head.mask_predictor.mask_embed.layers.0.weight",
    "segmentation_head.mask_predictor.mask_embed.layers.0.bias",
    "segmentation_head.mask_predictor.mask_embed.layers.1.weight",
    "segmentation_head.mask_predictor.mask_embed.layers.1.bias",
    "segmentation_head.mask_predictor.mask_embed.layers.2.weight",
    "segmentation_head.mask_predictor.mask_embed.layers.2.bias",
)
EXPECTED_STAGE0_NAMES_SHA256 = hashlib.sha256(
    ("\n".join(EXPECTED_STAGE0_NAMES) + "\n").encode("utf-8")
).hexdigest()
HASH = re.compile(r"^[0-9a-f]{64}$")


def evaluate(payload):
    if payload["execution_contract"] != {
        "mode": "one_shot_read_only_no_polling",
        "auto_start_when_idle": False,
    }:
        raise ValueError("preflight must be one-shot, read-only, and no-autostart")

    blockers = []
    truth = payload["truth_gate"]
    if not truth["truth_version_available"]:
        blockers.append("genuine_human_pixel_truth_version_missing")
    if not truth["truth_manifest_sha256_verified"]:
        blockers.append("truth_manifest_sha256_not_verified")

    holdout = payload["holdout_gate"]
    if not holdout["fixed_independent_baseline_available"]:
        blockers.append("fixed_independent_holdout_baseline_missing")
    if not holdout["protected_from_training"]:
        blockers.append("fixed_holdout_not_protected_from_training")

    stage0 = payload["stage0_gate"]
    names_valid = (
        tuple(stage0["exact_parameter_names"]) == EXPECTED_STAGE0_NAMES
        and stage0["tensor_count"] == 6
        and stage0["parameter_count"] == 197376
        and stage0["exact_names_sha256"] == EXPECTED_STAGE0_NAMES_SHA256
    )
    if not names_valid:
        blockers.append("stage0_exact_six_parameter_scope_not_verified")
    config_hash_valid = bool(HASH.fullmatch(stage0["config_sha256"] or ""))
    if not stage0["config_path"] or not config_hash_valid:
        blockers.append("stage0_config_path_or_sha256_missing")
    if not stage0["training_config_executable"]:
        blockers.append("stage0_training_config_not_executable")

    resources = payload["resource_gate"]
    if resources["other_user_processes"]:
        blockers.append("other_user_gpu_processes_present")
    baseline = resources["factory_4090_baseline"]
    if not (
        baseline["sam3_healthy"]
        and baseline["vlm_healthy"]
        and baseline["new_work_scheduled"] is False
        and baseline["mutation_performed"] is False
    ):
        blockers.append("factory_4090_baseline_not_protected")

    paths = payload["path_gate"]
    if not paths["output_path"]:
        blockers.append("training_output_path_missing")
    rollback_valid = (
        bool(paths["rollback_checkpoint_path"])
        and bool(HASH.fullmatch(paths["rollback_checkpoint_sha256"] or ""))
        and paths["rollback_checkpoint_verified"]
    )
    if not rollback_valid:
        blockers.append("rollback_checkpoint_path_or_sha256_not_verified")
    if not payload["formal_training_authorized"]:
        blockers.append("formal_training_not_authorized")

    requirements = {
        "genuine_human_truth_version": not any(
            item in blockers for item in (
                "genuine_human_pixel_truth_version_missing",
                "truth_manifest_sha256_not_verified",
            )
        ),
        "fixed_independent_holdout_baseline": not any(
            item in blockers for item in (
                "fixed_independent_holdout_baseline_missing",
                "fixed_holdout_not_protected_from_training",
            )
        ),
        "stage0_exact_six_parameters": names_valid,
        "stage0_config_sha256": config_hash_valid,
        "stage0_executable_config": stage0["training_config_executable"],
        "no_other_user_gpu_processes": not resources["other_user_processes"],
        "factory_4090_baseline_protected": (
            "factory_4090_baseline_not_protected" not in blockers
        ),
        "explicit_output_path": bool(paths["output_path"]),
        "verified_rollback_path_and_sha256": rollback_valid,
        "formal_training_authorized": payload["formal_training_authorized"],
    }
    passed = not blockers
    return {
        "schema_version": "labelagent.formal-ddp-preflight-decision.v1",
        "observed_at": payload["observed_at"],
        "input_id": payload["input_id"],
        "mode": "one_shot_read_only_no_polling",
        "preflight_passed": passed,
        "requirements": requirements,
        "blockers": blockers,
        "process_created": False,
        "automatic_launch_performed": False,
        "mutation_performed": False,
        "next_action": (
            "explicit_separate_launch_required"
            if passed else "remain_blocked_no_process_created"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    print(json.dumps(evaluate(payload), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
