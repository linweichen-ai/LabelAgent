"""Validate the blocked SAM3 PEFT launch plan without starting a process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_schema(instance, schema_path: Path):
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as error:
        raise RuntimeError("jsonschema is required for launch-gate validation") from error
    errors = sorted(
        Draft202012Validator(
            load_json(schema_path), format_checker=FormatChecker()
        ).iter_errors(instance),
        key=lambda item: list(item.absolute_path),
    )
    if errors:
        raise ValueError("; ".join(error.message for error in errors))


def blocked_smoke(plan_path: Path, manifest_path: Path):
    plan = load_json(plan_path)
    manifest = load_json(manifest_path)
    validate_schema(plan, ROOT / "schemas/sam3-peft-plan.schema.json")
    validate_schema(manifest, ROOT / "schemas/sam3-peft-data-manifest.schema.json")
    staged_path = ROOT / plan["staged_unfreeze_config"]
    staged = load_json(staged_path)
    validate_schema(staged, ROOT / "schemas/sam3-staged-unfreeze.schema.json")
    if plan["execution_enabled"] or plan["launch_gate"]["formal_training_authorized"]:
        raise ValueError("blocked smoke requires training authorization to remain false")
    if manifest["truth"]["train_annotations_verified"]:
        raise ValueError("blocked smoke cannot consume an unreviewed truth manifest")
    if manifest["isolation"]["test24_in_training"]:
        raise ValueError("test24 isolation violation")
    if manifest["isolation"]["fixed_640_holdout_in_training"]:
        raise ValueError("fixed 640 holdout isolation violation")
    return {
        "status": "blocked_launch_gate_smoke_passed",
        "process_started": False,
        "formal_training_authorized": False,
        "world_size": plan["distributed"]["world_size"],
        "backend": plan["distributed"]["backend"],
        "selected_route": plan["strategy"]["selected_route"],
        "full_model_post_training_selected": False,
        "verified_trainable_scope": plan["model"]["verified_trainable_scope"],
        "stage0_is_minimal_baseline": plan["model"]["stage0_is_minimal_baseline"],
        "stage0_is_final_upper_bound": plan["model"]["stage0_is_final_upper_bound"],
        "unverified_module_claims": plan["model"]["unverified_module_claims"],
        "staged_unfreeze": {
            "config": plan["staged_unfreeze_config"],
            "current_stage": staged["current_verified_stage"],
            "conditional_expansion": staged["conditional_expansion"],
            "dual_5090_named_parameters_verified": staged["runtime_inventory_gate"][
                "dual_5090_named_parameters_verified"
            ],
            "forward_backward_dependency_verified": staged["runtime_inventory_gate"][
                "forward_backward_dependency_verified"
            ],
            "holdout_metrics": staged["holdout_gate"]["required_metrics"],
            "threshold_status": staged["holdout_gate"]["threshold_status"],
            "unverified_module_policy": staged["unverified_module_policy"],
            "full_model_policy": staged["full_model_policy"],
        },
        "without_human_truth_allowed": plan["launch_gate"]["without_human_truth_allowed"],
        "without_human_truth_forbidden": plan["launch_gate"]["without_human_truth_forbidden"],
        "training_metrics_available": False,
        "checkpoint_available": False,
        "slope_improvement_established": False,
        "command_preview": None,
        "command_contract": {
            "entrypoint": ["python", "-m", "sam3.train.train"],
            "num_gpus": 2,
            "num_nodes": 1,
            "use_cluster": 0,
            "executable_hydra_config_ready": False,
            "config_blocker": "human truth and live source-compatible config validation are pending",
        },
        "blocker": manifest["status"],
        "isolation": manifest["isolation"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", type=Path,
        default=ROOT / "configs/sam3_sfl_cpd15y_peft_plan.json",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "configs/sam3_sfl_cpd15y_peft_data_manifest.blocked.json",
    )
    args = parser.parse_args()
    result = blocked_smoke(args.plan.resolve(), args.manifest.resolve())
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
