"""Fail-closed launch gate for supervised parameter-efficient SAM3 training."""

import re


HEX64 = re.compile(r"^[0-9a-f]{64}$")
ROUTE = "supervised_parameter_efficient_sam3"
TRUTH_SOURCE = "human_confirmed_pixel_masks"
REQUIRED_HOLDOUTS = {"test24", "fixed_640_holdout"}
ALLOWED_TRAINABLE_COMPONENTS = {"segmentation_head.mask_predictor.mask_embed"}


def _hash_or_none(value):
    return value is None or isinstance(value, str) and HEX64.fullmatch(value)


def validate_training_manifest(manifest):
    """Return semantic gate errors without mutating the supplied manifest."""
    errors = []
    if not isinstance(manifest, dict):
        return ["training manifest must be an object"]
    if manifest.get("schema_version") != "labelagent.sam3-peft-training-gate.v1":
        errors.append("unsupported training manifest schema_version")
    if manifest.get("training_route") != ROUTE:
        errors.append("training route must be supervised parameter-efficient SAM3")

    strategy = manifest.get("model_strategy")
    if not isinstance(strategy, dict):
        errors.append("model_strategy is required")
    else:
        if strategy.get("full_model_training") is not False:
            errors.append("full-model training is forbidden by the current decision")
        if strategy.get("freeze_image_backbone") is not True:
            errors.append("image backbone must remain frozen")
        if strategy.get("freeze_language_backbone") is not True:
            errors.append("language backbone must remain frozen")
        components = strategy.get("trainable_components")
        if (not isinstance(components, list) or not components
                or not set(components) <= ALLOWED_TRAINABLE_COMPONENTS):
            errors.append("only the source-verified mask-embedding head may be trainable")
        if strategy.get("adapter_status") != "not_verified_in_current_source":
            errors.append("adapter support must remain unverified until implemented and tested")

    distributed = manifest.get("distributed")
    if not isinstance(distributed, dict):
        errors.append("distributed configuration is required")
    elif (distributed.get("strategy") != "ddp"
          or distributed.get("accelerator") != "dual_5090"
          or distributed.get("world_size") != 2):
        errors.append("formal training must use dual-5090 DDP with world_size=2")

    sources = manifest.get("data_sources")
    if not isinstance(sources, list) or not sources:
        errors.append("data_sources are required")
    else:
        by_id = {item.get("source_id"): item for item in sources if isinstance(item, dict)}
        for source_id in ("machine_preannotations_72", "machine_preannotations_24",
                          "bev_3d_boxes"):
            item = by_id.get(source_id)
            if not item or item.get("truth_eligible") is not False:
                errors.append(f"{source_id} must be explicitly ineligible as truth")
            elif set(item.get("allowed_roles", [])) - {"prompt", "roi", "candidate"}:
                errors.append(f"{source_id} has a forbidden auxiliary role")

    truth = manifest.get("truth_dataset")
    if not isinstance(truth, dict):
        errors.append("truth_dataset is required")
        truth = {}
    if truth.get("required_source") != TRUTH_SOURCE:
        errors.append("truth source must be genuine human-confirmed pixel masks")
    if not _hash_or_none(truth.get("manifest_sha256")):
        errors.append("truth manifest_sha256 must be lowercase SHA-256 or null")

    isolation = manifest.get("isolation")
    if not isinstance(isolation, dict):
        errors.append("isolation evidence is required")
        isolated = {}
    else:
        entries = isolation.get("protected_sets")
        isolated = {item.get("set_id"): item for item in entries or []
                    if isinstance(item, dict)}
        if not REQUIRED_HOLDOUTS <= set(isolated):
            errors.append("test24 and fixed_640_holdout must both be protected")
        for set_id in REQUIRED_HOLDOUTS & set(isolated):
            if isolated[set_id].get("included_in_training") is not False:
                errors.append(f"{set_id} must be excluded from training")

    gates = manifest.get("gates")
    if not isinstance(gates, dict):
        errors.append("training gates are required")
        gates = {}
    authorized = manifest.get("formal_training_authorized")
    started = manifest.get("formal_training_started")
    if not isinstance(authorized, bool) or not isinstance(started, bool):
        errors.append("formal training authorization and start flags must be boolean")
    required_gates = ("human_pixel_truth_export_complete", "immutable_truth_version_committed",
                      "train_val_test_isolation_verified", "protected_holdouts_verified")
    if authorized:
        missing = [name for name in required_gates if gates.get(name) is not True]
        if missing:
            errors.append("formal training authorization requires all truth and isolation gates")
        if truth.get("mask_count", 0) <= 0:
            errors.append("formal training authorization requires non-empty human masks")
        if not truth.get("version_id"):
            errors.append("formal training authorization requires an immutable truth version")
        if not HEX64.fullmatch(str(truth.get("manifest_sha256", ""))):
            errors.append("formal training authorization requires a truth manifest hash")
    if started and not authorized:
        errors.append("formal training cannot start before authorization")
    return errors


def assert_training_authorized(manifest):
    errors = validate_training_manifest(manifest)
    if errors:
        raise ValueError("; ".join(errors))
    if manifest["formal_training_authorized"] is not True:
        raise PermissionError("formal SAM3 training is blocked by the truth gate")
    return True
