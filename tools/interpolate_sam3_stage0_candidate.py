"""Create a guarded Stage-0 candidate between the base and a trained candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import torch

from sam3.train.stage0_scope import (
    EXPECTED_PARAMETER_NAMES,
    build_stage0_image_model,
)


HASH = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def checked_path(value: str, expected_sha256: str, label: str) -> Path:
    path = Path(value).resolve()
    if "test24" in str(path).lower():
        raise ValueError(f"{label} cannot reference test24")
    if not path.is_file() or not HASH.fullmatch(expected_sha256):
        raise ValueError(f"invalid {label} binding")
    if sha256(path) != expected_sha256:
        raise ValueError(f"{label} hash mismatch")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--bpe-path", required=True)
    parser.add_argument("--bpe-sha256", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    output = args.output.resolve()
    if "test24" in str(output).lower() or output.exists():
        raise ValueError("output must be a fresh non-test24 path")

    base_path = checked_path(args.base_checkpoint, args.base_sha256, "base checkpoint")
    bpe_path = checked_path(args.bpe_path, args.bpe_sha256, "BPE")
    candidate_path = checked_path(args.candidate, args.candidate_sha256, "candidate")
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    expected = {
        "assistant_reviewed": True,
        "human_truth": False,
        "not_for_deployment": True,
        "test24_in_training": False,
    }
    for key, value in expected.items():
        if candidate.get(key) is not value:
            raise ValueError(f"candidate boundary mismatch: {key}")
    state = candidate.get("state_dict") or {}
    if tuple(state) != EXPECTED_PARAMETER_NAMES:
        raise ValueError("candidate tensor names drift")

    model = build_stage0_image_model(
        checkpoint_path=str(base_path),
        checkpoint_sha256=args.base_sha256,
        bpe_path=str(bpe_path),
        device="cpu",
        eval_mode=False,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )
    base = dict(model.named_parameters())
    blended = {}
    with torch.no_grad():
        for name in EXPECTED_PARAMETER_NAMES:
            before = base[name].detach().cpu()
            after = state[name].detach().cpu()
            if before.shape != after.shape:
                raise ValueError(f"candidate tensor shape drift: {name}")
            blended[name] = before + args.alpha * (after - before)

    result = dict(candidate)
    result.update({
        "schema_version": "labelagent.sam3-assistant-pallet-interpolation.v1",
        "state_dict": blended,
        "candidate_only": True,
        "assistant_reviewed": True,
        "human_truth": False,
        "not_human_truth": True,
        "not_for_deployment": True,
        "deployment_performed": False,
        "test24_in_training": False,
        "interpolation_alpha": args.alpha,
        "source_candidate_sha256": args.candidate_sha256,
        "base_checkpoint_sha256": args.base_sha256,
        "optimizer_steps": 0,
        "training_return_allowed": False,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(result, temporary)
    os.replace(temporary, output)
    print(json.dumps({
        "output": str(output),
        "sha256": sha256(output),
        "alpha": args.alpha,
        "source_candidate_sha256": args.candidate_sha256,
        "human_truth": False,
        "not_for_deployment": True,
        "test24_in_training": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
