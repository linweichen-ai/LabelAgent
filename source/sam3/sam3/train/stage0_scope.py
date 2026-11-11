"""Exact Stage-0 SAM3 model scope used by the supervised PEFT package."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re


EXPECTED_PARAMETER_NAMES = (
    "segmentation_head.mask_predictor.mask_embed.layers.0.weight",
    "segmentation_head.mask_predictor.mask_embed.layers.0.bias",
    "segmentation_head.mask_predictor.mask_embed.layers.1.weight",
    "segmentation_head.mask_predictor.mask_embed.layers.1.bias",
    "segmentation_head.mask_predictor.mask_embed.layers.2.weight",
    "segmentation_head.mask_predictor.mask_embed.layers.2.bias",
)
EXPECTED_PARAMETER_COUNT = 197376
EXPECTED_NAMES_SHA256 = "efacf600a68248109986db8b610f9861a8255e586aa64a2909cfcaeaded686a1"
HASH = re.compile(r"^[0-9a-f]{64}$")


def _file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def apply_exact_stage0_scope(model):
    """Freeze the complete model, then enable exactly the six verified tensors."""
    named = dict(model.named_parameters())
    missing = [name for name in EXPECTED_PARAMETER_NAMES if name not in named]
    if missing:
        raise ValueError(f"Stage-0 exact parameters missing: {missing}")
    for parameter in named.values():
        parameter.requires_grad_(False)
    for name in EXPECTED_PARAMETER_NAMES:
        named[name].requires_grad_(True)
    actual = tuple(name for name, parameter in model.named_parameters()
                   if parameter.requires_grad)
    count = sum(parameter.numel() for name, parameter in model.named_parameters()
                if name in EXPECTED_PARAMETER_NAMES)
    if actual != EXPECTED_PARAMETER_NAMES or count != EXPECTED_PARAMETER_COUNT:
        raise ValueError("Stage-0 trainable scope/count drift")
    return model


def build_stage0_image_model(*, checkpoint_path, checkpoint_sha256,
                             load_from_HF=False, **kwargs):
    """Build SAM3, verify the immutable base checkpoint, and apply Stage-0."""
    if load_from_HF is not False:
        raise ValueError("Stage-0 requires the explicitly bound local checkpoint")
    if not HASH.fullmatch(checkpoint_sha256 or ""):
        raise ValueError("checkpoint_sha256 must be lowercase SHA-256")
    if _file_sha256(checkpoint_path) != checkpoint_sha256:
        raise ValueError("base checkpoint SHA-256 mismatch")
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path, load_from_HF=False, **kwargs)
    return apply_exact_stage0_scope(model)
