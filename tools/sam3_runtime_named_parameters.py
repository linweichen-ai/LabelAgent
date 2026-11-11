"""Load SAM3 on CPU and emit an inventory-only named-parameter receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


STAGE0_PREFIX = "segmentation_head.mask_predictor.mask_embed."


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(named, *, include_names=False):
    names = [name for name, _ in named]
    result = {
        "tensor_count": len(named),
        "parameter_count": sum(parameter.numel() for _, parameter in named),
        "exact_names_sha256": hashlib.sha256(
            ("\n".join(names) + "\n").encode("utf-8")
        ).hexdigest(),
    }
    if include_names:
        result["exact_names"] = names
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam3-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    source = args.sam3_source.resolve()
    checkpoint = args.checkpoint.resolve()
    sys.path.insert(0, str(source))

    import torch

    torch.set_grad_enabled(False)
    # The upstream constructor precomputes two caches with explicit
    # ``device="cuda"`` even when ``device="cpu"`` is requested.  Redirect
    # only those tensor factories inside this short-lived inventory process;
    # do not modify the deployed source or the running service.
    original_zeros = torch.zeros
    original_arange = torch.arange

    def cpu_factory(original):
        def wrapped(*factory_args, **factory_kwargs):
            if str(factory_kwargs.get("device", "")) == "cuda":
                factory_kwargs["device"] = "cpu"
            return original(*factory_args, **factory_kwargs)
        return wrapped

    torch.zeros = cpu_factory(original_zeros)
    torch.arange = cpu_factory(original_arange)
    from sam3 import build_sam3_image_model

    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        enable_inst_interactivity=True,
        device="cpu",
    )
    named = list(model.named_parameters())
    stage0 = [item for item in named if item[0].startswith(STAGE0_PREFIX)]
    decoder_like = [item for item in named if "decoder" in item[0].lower()]
    adapter_lora_like = [
        item for item in named
        if "adapter" in item[0].lower() or "lora" in item[0].lower()
    ]
    top_level = {}
    for name, parameter in named:
        prefix = name.split(".", 1)[0]
        group = top_level.setdefault(prefix, {"tensor_count": 0, "parameter_count": 0})
        group["tensor_count"] += 1
        group["parameter_count"] += parameter.numel()

    print(json.dumps({
        "schema_version": "labelagent.sam3-runtime-named-parameters.v1",
        "scope": "inventory_only_not_training_authorization",
        "host_execution": "dual_5090_cpu_short_process",
        "constructor_device_redirects": ["torch.zeros:cuda->cpu", "torch.arange:cuda->cpu"],
        "torch_version": torch.__version__,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
        },
        "model": {
            "named_parameter_tensor_count": len(named),
            "named_parameter_count": sum(parameter.numel() for _, parameter in named),
            "top_level_groups": top_level,
        },
        "verified_stage0_scope": {
            "prefix": STAGE0_PREFIX,
            **summarize(stage0, include_names=True),
        },
        "conditional_observations": {
            "authorized_for_training": False,
            "decoder_like": summarize(decoder_like),
            "adapter_or_lora_like": summarize(
                adapter_lora_like, include_names=True
            ),
        },
        "gates": {
            "genuine_human_truth_version_available": False,
            "fixed_holdout_baseline_available": False,
            "conditional_expansion_authorized": False,
            "full_model_training_authorized": False,
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
