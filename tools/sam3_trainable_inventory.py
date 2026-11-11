"""Inventory SAM3 named parameters without authorizing broader training scopes.

Only ``segmentation_head.mask_predictor.mask_embed.*`` is currently verified
as trainable Stage-0 scope. Decoder-like or adapter/LoRA-like names are reported
as inventory-only observations; they are never enabled by this module.
"""
from __future__ import annotations


STAGE0_PREFIX = "segmentation_head.mask_predictor.mask_embed."


def _names(model):
    return [(name, parameter) for name, parameter in model.named_parameters()]


def _numel(parameter):
    value = parameter.numel()
    return int(value)


def _summary(named):
    return {
        "tensor_count": len(named),
        "parameter_count": sum(_numel(parameter) for _, parameter in named),
        "names": [name for name, _ in named],
    }


def inventory_named_parameters(model):
    """Return exact runtime names/counts while keeping expansion unauthorized."""
    named = _names(model)
    stage0 = [item for item in named if item[0].startswith(STAGE0_PREFIX)]
    decoder_observations = [
        item for item in named
        if ".decoder." in item[0] or item[0].startswith("transformer.decoder.")
    ]
    adapter_or_lora_observations = [
        item for item in named
        if "adapter" in item[0].lower() or "lora" in item[0].lower()
    ]
    return {
        "scope": "inventory_only_not_training_authorization",
        "named_parameter_tensor_count": len(named),
        "named_parameter_count": sum(_numel(parameter) for _, parameter in named),
        "verified_stage0_scope": {
            "status": "only_currently_verified_trainable_scope",
            "prefix": STAGE0_PREFIX,
            **_summary(stage0),
        },
        "conditional_inventory_only": {
            "authorized_for_training": False,
            "decoder_like": _summary(decoder_observations),
            "adapter_or_lora_like": _summary(adapter_or_lora_observations),
        },
        "conditional_expansion_enabled": False,
        "full_model_training_enabled": False,
    }


def set_trainable_stage(model, stage):
    """Freeze everything and enable only the verified Stage-0 prefix."""
    if stage != 0:
        raise ValueError(
            "only Stage-0 is authorized; broader scopes require dual-5090 "
            "named_parameters inventory, human truth, and a fixed holdout baseline"
        )
    trainable = []
    trainable_parameter_count = 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(STAGE0_PREFIX)
        if parameter.requires_grad:
            trainable.append(name)
            trainable_parameter_count += _numel(parameter)
    if not trainable:
        raise ValueError("Stage-0 selected no real named parameters")
    return {
        "stage": 0,
        "selectors": [STAGE0_PREFIX],
        "trainable_names": trainable,
        "trainable_parameter_count": trainable_parameter_count,
        "rollback_to": "original_sam3_checkpoint_with_sha256",
    }
