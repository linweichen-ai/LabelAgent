"""Bounded Stage-0 trainer for assistant-reviewed manual pallet crops."""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import torch

from sam3.train.optim.optimizer import construct_optimizer
from sam3.train.stage0_scope import EXPECTED_PARAMETER_COUNT, EXPECTED_PARAMETER_NAMES
from sam3.train.stage0_trainer import Stage0MaskEmbedTrainer
from sam3.train.trainer import unwrap_ddp_if_wrapped
from sam3.train.utils.train_utils import Phase


ACK = "I_ACKNOWLEDGE_ASSISTANT_REVIEW_IS_NOT_HUMAN_TRUTH"
EXPECTED_STEPS = 36
EXPECTED_ANNOTATION = "_assistant_delegated_pallet_crops.coco.json"


class _LimitedLoader:
    def __init__(self, loader, limit):
        self.loader = loader
        self.limit = min(int(limit), len(loader))

    def __len__(self):
        return self.limit

    def __iter__(self):
        return itertools.islice(iter(self.loader), self.limit)


class AssistantInstanceCropTrainer(Stage0MaskEmbedTrainer):
    """Train exactly one two-GPU epoch on 71 positive pallet crops."""

    def __init__(self, *, max_optimizer_steps, candidate_contract, **kwargs):
        if os.environ.get("SAM3_ASSISTANT_INSTANCE_ACK") != ACK:
            raise PermissionError("assistant-review acknowledgement is absent")
        if max_optimizer_steps != EXPECTED_STEPS:
            raise ValueError("assistant crop run is limited to exactly 36 steps")
        expected = {
            "assistant_reviewed": True,
            "source_manual_instances": True,
            "human_truth": False,
            "not_for_deployment": True,
            "positive_instance_crops": True,
            "test24_in_training": False,
        }
        if candidate_contract != expected:
            raise ValueError("assistant crop candidate metadata drift")
        data = kwargs.get("data")
        ann_file = str(data["train"]["dataset"]["ann_file"])
        if not ann_file.endswith("/" + EXPECTED_ANNOTATION):
            raise ValueError("trainer only accepts the versioned assistant crop COCO")
        if "/test" in ann_file.lower() or "test24" in ann_file.lower():
            raise ValueError("test input is forbidden")
        if kwargs.get("mode") != "train_only" or kwargs.get("max_epochs") != 1:
            raise ValueError("assistant crop run must be one train-only epoch")
        self.max_optimizer_steps = max_optimizer_steps
        self.candidate_contract = dict(candidate_contract)
        self._candidate_train_stats = None
        super().__init__(**kwargs)

    def _construct_optimizers(self):
        named = dict(self.model.named_parameters())
        actual = tuple(name for name, parameter in named.items() if parameter.requires_grad)
        count = sum(named[name].numel() for name in EXPECTED_PARAMETER_NAMES if name in named)
        if actual != EXPECTED_PARAMETER_NAMES or count != EXPECTED_PARAMETER_COUNT:
            raise ValueError("assistant crop Stage-0 scope/count drift")
        self.optim = construct_optimizer(
            self.model,
            self.optim_conf.optimizer,
            self.optim_conf.options,
            self.optim_conf.param_group_modifiers,
            param_allowlist=set(EXPECTED_PARAMETER_NAMES),
            validate_param_groups=False,
        )

    def train_epoch(self, train_loader):
        remaining = self.max_optimizer_steps - self.steps[Phase.TRAIN]
        if remaining <= 0:
            raise RuntimeError("optimizer-step budget already exhausted")
        result = super().train_epoch(_LimitedLoader(train_loader, remaining))
        if self.steps[Phase.TRAIN] > self.max_optimizer_steps:
            raise RuntimeError("optimizer-step budget exceeded")
        self._candidate_train_stats = result
        return result

    def run_train(self):
        super().run_train()
        steps = self.steps[Phase.TRAIN]
        if steps != self.max_optimizer_steps:
            raise RuntimeError(f"expected {self.max_optimizer_steps} steps, got {steps}")
        if self.distributed_rank != 0:
            return
        checkpoint_dir = Path(self.checkpoint_conf.save_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model = unwrap_ddp_if_wrapped(self.model)
        named = dict(model.named_parameters())
        state = {name: named[name].detach().cpu() for name in EXPECTED_PARAMETER_NAMES}
        checkpoint = {
            "schema_version": "labelagent.sam3-assistant-pallet-crop-checkpoint.v1",
            "assistant_reviewed": True,
            "source_manual_instances": True,
            "human_truth": False,
            "not_for_deployment": True,
            "test24_in_training": False,
            "optimizer_steps": steps,
            "state_dict": state,
        }
        target = checkpoint_dir / "assistant_pallet_crop_mask_embed.pt"
        temporary = target.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, target)
        evidence = {
            key: checkpoint[key]
            for key in (
                "schema_version",
                "assistant_reviewed",
                "source_manual_instances",
                "human_truth",
                "not_for_deployment",
                "test24_in_training",
                "optimizer_steps",
            )
        }
        evidence["train_stats"] = self._candidate_train_stats
        evidence["checkpoint_path"] = str(target)
        (checkpoint_dir.parent / "training_evidence.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
