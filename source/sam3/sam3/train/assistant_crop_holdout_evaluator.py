"""Evaluation-only SAM3 loader for the 15-crop assistant holdout."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from sam3.train.optim.optimizer import construct_optimizer
from sam3.train.stage0_scope import (
    EXPECTED_PARAMETER_COUNT, EXPECTED_PARAMETER_NAMES, build_stage0_image_model,
)
from sam3.train.stage0_trainer import Stage0MaskEmbedTrainer


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_assistant_crop_holdout_model(*, candidate_checkpoint_path,
                                       candidate_checkpoint_sha256, **kwargs):
    model = build_stage0_image_model(**kwargs)
    if candidate_checkpoint_path == "BASELINE":
        if candidate_checkpoint_sha256 != "BASELINE":
            raise ValueError("baseline sentinel mismatch")
        return model
    if _sha256(candidate_checkpoint_path) != candidate_checkpoint_sha256:
        raise ValueError("candidate checkpoint hash mismatch")
    checkpoint = torch.load(candidate_checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("assistant_reviewed") is not True:
        raise ValueError("assistant-review boundary missing")
    if checkpoint.get("human_truth") is not False:
        raise ValueError("candidate must not claim human truth")
    if checkpoint.get("not_for_deployment") is not True:
        raise ValueError("candidate deployment boundary missing")
    if checkpoint.get("test24_in_training") is not False:
        raise ValueError("test24 boundary missing")
    state = checkpoint.get("state_dict") or {}
    if tuple(state) != EXPECTED_PARAMETER_NAMES:
        raise ValueError("candidate tensor names drift")
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name in EXPECTED_PARAMETER_NAMES:
            if tuple(named[name].shape) != tuple(state[name].shape):
                raise ValueError(f"candidate tensor shape drift: {name}")
            named[name].copy_(state[name])
    return model


class AssistantCropHoldoutEvaluator(Stage0MaskEmbedTrainer):
    """Forbid training and evaluate only the frozen 15-crop split."""

    def __init__(self, *, evaluation_contract, **kwargs):
        expected = {
            "assistant_reviewed": True, "human_truth": False,
            "evaluation_only": True, "not_for_deployment": True,
            "training_return_allowed": False, "test24_included": False,
        }
        if evaluation_contract != expected:
            raise ValueError("assistant holdout evaluation boundary drift")
        if kwargs.get("mode") != "val":
            raise ValueError("assistant holdout evaluator only supports val mode")
        data = kwargs.get("data")
        ann_file = str(data["val"]["dataset"]["ann_file"])
        if not ann_file.endswith("/" +
                "_assistant_delegated_pallet_crops_holdout15.coco.json"):
            raise ValueError("assistant holdout annotation path drift")
        if data.get("train") is not None:
            raise ValueError("assistant holdout evaluator cannot instantiate training data")
        super().__init__(**kwargs)

    def _construct_optimizers(self):
        named = dict(self.model.named_parameters())
        actual = tuple(name for name, parameter in named.items() if parameter.requires_grad)
        count = sum(named[name].numel() for name in EXPECTED_PARAMETER_NAMES if name in named)
        if actual != EXPECTED_PARAMETER_NAMES or count != EXPECTED_PARAMETER_COUNT:
            raise ValueError("assistant holdout Stage-0 scope/count drift")
        self.optim = construct_optimizer(
            self.model, self.optim_conf.optimizer, self.optim_conf.options,
            self.optim_conf.param_group_modifiers,
            param_allowlist=set(EXPECTED_PARAMETER_NAMES), validate_param_groups=False)

    def _step(self, batch, model, phase):
        model.train()
        try:
            return super()._step(batch, model, phase)
        finally:
            model.eval()

    def run_train(self):
        raise RuntimeError("assistant holdout can never train")
