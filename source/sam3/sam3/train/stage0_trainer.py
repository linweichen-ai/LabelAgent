"""SAM3 Trainer specialization that only optimizes the six Stage-0 tensors."""
from sam3.train.optim.optimizer import construct_optimizer
from sam3.train.stage0_scope import (
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_PARAMETER_NAMES,
)
from sam3.train.trainer import Trainer


class Stage0MaskEmbedTrainer(Trainer):
    """Fail closed if model scope drifts before optimizer construction."""

    def _construct_optimizers(self):
        named = dict(self.model.named_parameters())
        actual = tuple(name for name, parameter in named.items()
                       if parameter.requires_grad)
        count = sum(named[name].numel() for name in EXPECTED_PARAMETER_NAMES
                    if name in named)
        if actual != EXPECTED_PARAMETER_NAMES or count != EXPECTED_PARAMETER_COUNT:
            raise ValueError("Stage-0 optimizer scope/count drift")
        self.optim = construct_optimizer(
            self.model,
            self.optim_conf.optimizer,
            self.optim_conf.options,
            self.optim_conf.param_group_modifiers,
            param_allowlist=set(EXPECTED_PARAMETER_NAMES),
        )
