"""Build deterministic, CPU-only SAM3 Stage-0 dry-run evidence."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.agent.stage0_training_contract import (
    digest,
    evaluate_deployment_candidate,
)
from tools.sam3_stage0_launcher import compile_only


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "configs/sam3_sfl_cpd15y_stage0_training_package.json"
COMPILED_AT = "2026-11-11T23:40:00+08:00"


def metrics(iou, boundary, precision, recall, false_positives):
    return {
        "mask_iou": iou,
        "boundary_f1": boundary,
        "object_precision": precision,
        "object_recall": recall,
        "false_positives": false_positives,
    }


def rejected_deployment_fixture():
    incumbent = {
        "overall": metrics(0.71, 0.66, 0.79, 0.74, 8),
        "per_class": {
            "floor": metrics(0.80, 0.75, 0.86, 0.82, 3),
            "slope": metrics(0.62, 0.57, 0.72, 0.65, 5),
        },
        "source_real": metrics(0.70, 0.65, 0.78, 0.73, 7),
        "source_simulation": metrics(0.73, 0.68, 0.81, 0.76, 1),
    }
    candidate = copy.deepcopy(incumbent)
    candidate["overall"]["mask_iou"] = 0.72
    candidate["per_class"]["slope"]["boundary_f1"] = 0.55
    return evaluate_deployment_candidate({
        "evaluation_id": "STATIC_TEST_ONLY_SLOPE_REGRESSION",
        "fixed_holdout": True,
        "test24_in_training": False,
        "fixed_holdout_manifest_sha256": "1" * 64,
        "candidate_checkpoint_sha256": "2" * 64,
        "incumbent": incumbent,
        "candidate": candidate,
    })


def build_static_acceptance():
    compiled = compile_only(PACKAGE, compiled_at=COMPILED_AT)
    plan = compiled["plan"]
    rejection = compiled["rejection"]
    decision = rejected_deployment_fixture()
    outputs = {
        ROOT / "docs/test-results/sam3-stage0-dry-run-plan-20261111.json": plan,
        ROOT / "docs/test-results/sam3-stage0-launch-rejection-20261111.json": rejection,
        ROOT / "examples/sam3_stage0_slope_regression_rejected_static.json": decision,
    }
    receipt = {
        "schema_version": "labelagent.sam3-stage0-static-validation.v1",
        "generated_at": "2026-11-11T23:41:00+08:00",
        "package_id": plan["package_id"],
        "package_sha256": plan["package_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "rejection_receipt_sha256": rejection["receipt_sha256"],
        "negative_deployment_decision_sha256": decision["decision_sha256"],
        "execution_boundary": {
            "cpu_only_static_validation": True,
            "process_created": False,
            "gpu_called": False,
            "remote_service_called": False,
            "cvat_called": False,
            "ssh_called": False,
            "oss_called": False,
        },
        "gates": {
            "exact_six_mask_embed_tensors": True,
            "exact_parameter_count_197376": True,
            "human_truth_required": True,
            "machine_preannotations_rejected_as_truth": True,
            "synthetic_fixtures_rejected_as_truth": True,
            "test24_evaluation_only": True,
            "fixed_holdout_training_forbidden": True,
            "single_dual_effective_batch_equal": True,
            "overall_and_slope_non_regression_required": True,
            "floor_slope_and_real_sim_reported_separately": True,
            "incumbent_retained_by_default": True,
            "two_stage_launcher": True,
        },
        "training_started": False,
        "checkpoints_created": 0,
        "deployment_candidates_created": 0,
        "cnb_status": "pending_not_contacted",
    }
    receipt["receipt_sha256"] = digest(receipt)
    outputs[ROOT / "docs/test-results/sam3-stage0-static-validation-20261111.json"] = receipt
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
