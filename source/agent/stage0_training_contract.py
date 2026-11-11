"""Fail-closed compile and acceptance contracts for SAM3 Stage-0 training."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re


HASH = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_NAMES = (
    "segmentation_head.mask_predictor.mask_embed.layers.0.weight",
    "segmentation_head.mask_predictor.mask_embed.layers.0.bias",
    "segmentation_head.mask_predictor.mask_embed.layers.1.weight",
    "segmentation_head.mask_predictor.mask_embed.layers.1.bias",
    "segmentation_head.mask_predictor.mask_embed.layers.2.weight",
    "segmentation_head.mask_predictor.mask_embed.layers.2.bias",
)
EXPECTED_NAMES_SHA256 = "efacf600a68248109986db8b610f9861a8255e586aa64a2909cfcaeaded686a1"
EXPECTED_PARAMETER_COUNT = 197376
METRICS = ("mask_iou", "boundary_f1", "object_precision", "object_recall",
           "false_positives")


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes)
                          else canonical_json(value)).hexdigest()


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _time(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be ISO-8601")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include timezone")


def _sha(value):
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def validate_package(package, root):
    if package.get("schema_version") != "labelagent.sam3-stage0-training-package.v1":
        raise ValueError("unsupported Stage-0 package schema")
    execution = package["execution_policy"]
    if execution != {
        "allowed_modes": ["compile", "dry_run", "explicit_launch_after_new_authorization"],
        "default_mode": "compile", "auto_start": False,
        "process_creation_currently_allowed": False,
    }:
        raise ValueError("current Stage-0 package must remain compile-only")
    scope = package["trainable_scope"]
    if (tuple(scope["exact_parameter_names"]) != EXPECTED_NAMES
            or scope["tensor_count"] != 6
            or scope["parameter_count"] != EXPECTED_PARAMETER_COUNT
            or scope["exact_names_sha256"] != EXPECTED_NAMES_SHA256
            or scope["image_backbone_frozen"] is not True
            or scope["language_backbone_frozen"] is not True):
        raise ValueError("Stage-0 exact six-parameter scope drift")
    if digest(("\n".join(EXPECTED_NAMES) + "\n").encode()) != EXPECTED_NAMES_SHA256:
        raise ValueError("Stage-0 names hash constant is invalid")
    model = package["model"]
    for path_field, sha_field in (
        ("stage0_recipe_path", "stage0_recipe_sha256"),
        ("stage0_single_gpu_recipe_path", "stage0_single_gpu_recipe_sha256"),
        ("base_recipe_path", "base_recipe_sha256"),
        ("scope_module_path", "scope_module_sha256"),
        ("trainer_module_path", "trainer_module_sha256"),
    ):
        path = Path(root) / model[path_field]
        if not path.is_file() or file_sha256(path) != model[sha_field]:
            raise ValueError(f"Stage-0 bound source hash mismatch: {path_field}")
    if not _sha(model["base_checkpoint_sha256"]):
        raise ValueError("base checkpoint SHA-256 is missing")
    if model["full_model_training"] is not False or model["adapter_or_lora_claimed"] is not False:
        raise ValueError("full-model/adapter/LoRA claims are forbidden")
    data = package["data"]
    if data["test24_role"] != "evaluation_only" or data["test24_in_training"] is not False:
        raise ValueError("test24 must remain evaluation-only")
    if data["fixed_holdout_in_training"] is not False:
        raise ValueError("fixed holdout cannot enter training")
    if data["machine_preannotations_truth_eligible"] is not False or data["synthetic_fixtures_truth_eligible"] is not False:
        raise ValueError("machine/synthetic artifacts cannot be truth")
    if package["paths"]["overwrite_existing_output"] is not False:
        raise ValueError("Stage-0 outputs must not overwrite existing runs")
    single, dual = package["topology"]["single_gpu"], package["topology"]["dual_gpu"]
    if single["effective_global_batch"] != dual["effective_global_batch"]:
        raise ValueError("single/dual topology effective batch mismatch")
    if dual["strategy"] != "synchronous_ddp" or dual["world_size"] != 2 or dual["backend"] != "nccl":
        raise ValueError("formal Stage-0 requires synchronous dual-5090 NCCL DDP")
    gate = package["evaluation_gate"]
    if (tuple(gate["required_metrics"]) != METRICS
            or set(gate["required_classes"]) != {"floor", "slope"}
            or gate["incumbent_default"] != "retain_original_sam3"):
        raise ValueError("Stage-0 evaluation gate is incomplete")
    return True


def compile_dry_run(package, *, root, package_path, compiled_at):
    """Compile an auditable command plan. This function cannot create a process."""
    _time(compiled_at, "compiled_at")
    validate_package(package, root)
    blockers = []
    truth = package["truth"]
    if (truth["required_source"] != "cvat_human_completed_pixel_masks"
            or truth["human_completed"] is not True):
        blockers.append("genuine_human_pixel_truth_missing")
    if not (truth["immutable_version_verified"] and truth["truth_version_id"]
            and _sha(truth["truth_version_sha256"])
            and truth["manifest_path"] and _sha(truth["manifest_sha256"])
            and _sha(truth["annotation_export_sha256"])):
        blockers.append("immutable_truth_version_or_manifest_binding_missing")
    data = package["data"]
    if not (_sha(data["train_manifest_sha256"])
            and _sha(data["validation_manifest_sha256"])
            and data["split_isolation_verified"]):
        blockers.append("train72_train_val_manifest_or_isolation_missing")
    holdout = package["holdout"]
    if not (holdout["baseline_verified"] and holdout["protected_from_training"]
            and holdout["fixed_independent_manifest_path"]
            and _sha(holdout["fixed_independent_manifest_sha256"])
            and holdout["incumbent_baseline_metrics_path"]
            and _sha(holdout["incumbent_baseline_metrics_sha256"])):
        blockers.append("fixed_independent_holdout_baseline_missing")
    resources = package["resources"]
    if not (resources["resource_snapshot_current"]
            and resources["no_other_user_gpu_processes_verified"]
            and resources["factory_4090_baseline_protected"]
            and resources["ddp_initialization_verified_for_this_package"]):
        blockers.append("dual_5090_resource_or_ddp_admission_missing")
    if resources["stage0_runtime_controls_verified"] is not True:
        blockers.append("stage0_early_stop_checkpoint_runtime_control_unverified")
    authorization = package["authorization"]
    if not (authorization["formal_training_authorized"]
            and authorization["launch_authorization_receipt_path"]
            and _sha(authorization["launch_authorization_receipt_sha256"])):
        blockers.append("explicit_training_authorization_missing")
    if package["execution_policy"]["process_creation_currently_allowed"] is not True:
        blockers.append("current_package_compile_only_process_creation_forbidden")
    package_path = Path(package_path).resolve()
    root_path = Path(root).resolve()
    try:
        portable_package_path = package_path.relative_to(root_path)
    except ValueError as exc:
        raise ValueError("package_path must stay within the repository root") from exc
    plan = {
        "schema_version": "labelagent.sam3-stage0-dry-run-plan.v1",
        "compiled_at": compiled_at,
        "package_id": package["package_id"],
        "package_path": str(portable_package_path).replace("\\", "/"),
        "package_sha256": file_sha256(package_path),
        "stage0_recipe_path": package["model"]["stage0_recipe_path"],
        "stage0_recipe_sha256": package["model"]["stage0_recipe_sha256"],
        "stage0_single_gpu_recipe_path": package["model"]["stage0_single_gpu_recipe_path"],
        "stage0_single_gpu_recipe_sha256": package["model"]["stage0_single_gpu_recipe_sha256"],
        "base_checkpoint_path": package["model"]["base_checkpoint_path"],
        "base_checkpoint_sha256": package["model"]["base_checkpoint_sha256"],
        "truth_version_id": truth["truth_version_id"],
        "truth_manifest_sha256": truth["manifest_sha256"],
        "fixed_holdout_manifest_sha256": holdout["fixed_independent_manifest_sha256"],
        "train_manifest_sha256": data["train_manifest_sha256"],
        "validation_manifest_sha256": data["validation_manifest_sha256"],
        "parameter_names_sha256": package["trainable_scope"]["exact_names_sha256"],
        "tensor_count": 6, "parameter_count": EXPECTED_PARAMETER_COUNT,
        "single_gpu_equivalence_profile": package["topology"]["single_gpu"],
        "dual_gpu_profile": package["topology"]["dual_gpu"],
        "command_preview": ["python", "-m", "sam3.train.train",
                            "--config", "sfl_cpd15y/stage0_mask_embed",
                            "--num-gpus", "2", "--num-nodes", "1",
                            "--use-cluster", "0"],
        "single_gpu_equivalence_command_preview": [
            "python", "-m", "sam3.train.train", "--config",
            "sfl_cpd15y/stage0_mask_embed_single_gpu", "--num-gpus", "1",
            "--num-nodes", "1", "--use-cluster", "0"],
        "required_environment": ["SAM3_STAGE0_DATA_ROOT", "SAM3_STAGE0_TRAIN_ROOT",
                                 "SAM3_STAGE0_VAL_ROOT", "SAM3_STAGE0_OUTPUT_ROOT",
                                 "SAM3_STAGE0_BPE_PATH", "SAM3_STAGE0_CHECKPOINT"],
        "optimizer": package["optimization"],
        "checkpointing": package["checkpointing"],
        "evaluation_gate": package["evaluation_gate"],
        "status": "launch_authorized" if not blockers else "rejected_fail_closed",
        "blockers": blockers,
        "process_created": False,
        "gpu_called": False,
        "automatic_launch_performed": False,
        "next_action": "explicit_second_stage_launch" if not blockers else "remain_blocked_compile_only",
    }
    plan["plan_sha256"] = digest({key: value for key, value in plan.items()
                                  if key != "plan_sha256"})
    return plan


def rejection_receipt(plan):
    if plan["status"] != "rejected_fail_closed" or not plan["blockers"]:
        raise ValueError("rejection receipt requires a blocked dry-run plan")
    body = {
        "schema_version": "labelagent.sam3-stage0-launch-rejection.v1",
        "package_id": plan["package_id"], "plan_sha256": plan["plan_sha256"],
        "status": "rejected_fail_closed", "blockers": plan["blockers"],
        "process_created": False, "gpu_called": False,
        "training_started": False, "checkpoint_created": False,
        "deployment_candidate_created": False,
        "incumbent_retained": True,
        "next_action": "await_genuine_truth_holdout_and_resource_authorization",
    }
    body["receipt_sha256"] = digest(body)
    return body


def _validate_metrics(metrics):
    if not isinstance(metrics, dict) or set(metrics) != set(METRICS):
        raise ValueError("metric set is incomplete")
    for name in METRICS[:-1]:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{name} is invalid")
    if isinstance(metrics["false_positives"], bool) or not isinstance(metrics["false_positives"], int) or metrics["false_positives"] < 0:
        raise ValueError("false_positives is invalid")


def evaluate_deployment_candidate(payload):
    """Apply overall + slope non-regression gates on the fixed holdout."""
    if payload.get("fixed_holdout") is not True or payload.get("test24_in_training") is not False:
        raise ValueError("deployment evaluation must use protected fixed holdout")
    for side in ("incumbent", "candidate"):
        values = payload[side]
        if set(values) != {"overall", "per_class", "source_real", "source_simulation"}:
            raise ValueError("evaluation accounting views are incomplete")
        _validate_metrics(values["overall"])
        if set(values["per_class"]) < {"floor", "slope"}:
            raise ValueError("floor/slope metrics are required")
        _validate_metrics(values["per_class"]["floor"])
        _validate_metrics(values["per_class"]["slope"])
        _validate_metrics(values["source_real"])
        _validate_metrics(values["source_simulation"])

    failures = []
    for view in ("overall",):
        before, after = payload["incumbent"][view], payload["candidate"][view]
        for metric in METRICS[:-1]:
            if after[metric] < before[metric]:
                failures.append(f"{view}.{metric}_regressed")
        if after["false_positives"] > before["false_positives"]:
            failures.append(f"{view}.false_positives_increased")
    before = payload["incumbent"]["per_class"]["slope"]
    after = payload["candidate"]["per_class"]["slope"]
    for metric in METRICS[:-1]:
        if after[metric] < before[metric]:
            failures.append(f"slope.{metric}_regressed")
    if after["false_positives"] > before["false_positives"]:
        failures.append("slope.false_positives_increased")
    passed = not failures
    body = {
        "schema_version": "labelagent.sam3-stage0-deployment-decision.v1",
        "evaluation_id": payload["evaluation_id"],
        "fixed_holdout_manifest_sha256": payload["fixed_holdout_manifest_sha256"],
        "candidate_checkpoint_sha256": payload["candidate_checkpoint_sha256"],
        "overall_and_slope_non_regression_passed": passed,
        "failures": failures,
        "deployment_candidate": passed,
        "incumbent_retained": not passed,
        "real_sim_metrics_reported_separately": True,
    }
    body["decision_sha256"] = digest(body)
    return body
