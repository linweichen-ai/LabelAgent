"""Inventory real SAM3 named-parameter groups for staged unfreezing.

The module is torch-agnostic: pass any model exposing ``named_parameters``.
It never invents adapters and does not perform a forward, backward, or optimizer
step. Runtime forward/backward dependency checks remain a separate GPU gate.
"""
from __future__ import annotations

import re


STAGE0_PREFIX = "segmentation_head.mask_predictor.mask_embed."
SEGMENTATION_PREFIX = "segmentation_head."
GEOMETRY_PREFIX = "geometry_encoder."
DECODER_LAYER_RE = re.compile(r"^transformer\.decoder\.layers\.(\d+)\.")
DECODER_NORM_PREFIX = "transformer.decoder.norm."


def _names(model):
    return [(name, parameter) for name, parameter in model.named_parameters()]


def inventory_named_parameters(model):
    named = _names(model)
    all_names = [name for name, _ in named]
    decoder_indices = sorted({
        int(match.group(1))
        for name in all_names
        if (match := DECODER_LAYER_RE.match(name))
    })
    upper_indices = decoder_indices[len(decoder_indices) // 2:]
    upper_prefixes = [f"transformer.decoder.layers.{index}." for index in upper_indices]

    groups = {
        "stage0_mask_embed": [name for name in all_names if name.startswith(STAGE0_PREFIX)],
        "stage1_full_segmentation_head": [
            name for name in all_names if name.startswith(SEGMENTATION_PREFIX)
        ],
        "stage2_geometry_prompt_encoder": [
            name for name in all_names if name.startswith(GEOMETRY_PREFIX)
        ],
        "stage3_upper_transformer_decoder": [
            name for name in all_names
            if any(name.startswith(prefix) for prefix in upper_prefixes)
            or name.startswith(DECODER_NORM_PREFIX)
        ],
    }
    return {
        "named_parameter_count": len(all_names),
        "decoder_layer_indices": decoder_indices,
        "runtime_upper_decoder_indices": upper_indices,
        "runtime_upper_decoder_prefixes": upper_prefixes,
        "groups": {
            key: {"parameter_count": len(names), "names": names}
            for key, names in groups.items()
        },
        "adapter_or_lora_parameters": [
            name for name in all_names if "adapter" in name.lower() or "lora" in name.lower()
        ],
        "stage0_is_final_upper_bound": False,
    }


def selectors_for_stage(model, stage):
    inventory = inventory_named_parameters(model)
    selectors = [STAGE0_PREFIX]
    if stage >= 1:
        selectors = [SEGMENTATION_PREFIX]
    if stage >= 2:
        selectors.append(GEOMETRY_PREFIX)
    if stage >= 3:
        selectors.extend(inventory["runtime_upper_decoder_prefixes"])
        selectors.append(DECODER_NORM_PREFIX)
    if stage not in {0, 1, 2, 3}:
        raise ValueError("stage must be 0, 1, 2, or 3")
    return selectors


def set_trainable_stage(model, stage):
    """Freeze all parameters, then enable only the selected real prefixes."""
    selectors = selectors_for_stage(model, stage)
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(name.startswith(prefix) for prefix in selectors)
        if parameter.requires_grad:
            trainable.append(name)
    if not trainable:
        raise ValueError(f"stage {stage} selected no real named parameters")
    return {"stage": stage, "selectors": selectors, "trainable_names": trainable}
