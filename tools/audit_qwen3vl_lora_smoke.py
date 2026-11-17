"""Audit an isolated Qwen3-VL LoRA smoke artifact without loading the base model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(output_dir: Path, dataset_dir: Path, expected_status: str,
          expected_steps: int) -> dict:
    receipt_path = output_dir / "run_receipt.json"
    adapter_dir = output_dir / "adapter"
    adapter_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    manifest_path = dataset_dir / "manifest.json"
    train_path = dataset_dir / "train.jsonl"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    boundaries = {
        "status": receipt.get("status") == expected_status,
        "teacher_distillation": receipt.get("track") == "teacher_distillation",
        "production_prompt_order": receipt.get("prompt_order") == "text_then_image",
        "human_truth_false": receipt.get("human_truth") is False,
        "production_deployed_false": receipt.get("production_deployed") is False,
        "test24_training_false": receipt.get("test24_used_for_training") is False,
        "manifest_test24_false": manifest.get("test24_included") is False,
        "manifest_human_truth_false": manifest.get("human_truth") is False,
        "manifest_not_production_approved": manifest.get("production_training_approved") is False,
        "manifest_hash_matches": receipt.get("dataset_manifest_sha256") == file_sha256(manifest_path),
        "train_hash_matches": receipt.get("train_jsonl_sha256") == file_sha256(train_path),
        "expected_steps": receipt.get("steps") == expected_steps,
        "finite_loss": isinstance(receipt.get("train_loss"), (int, float))
        and math.isfinite(receipt["train_loss"]),
        "lora_rank_8": config.get("r") == 8,
    }

    from safetensors import safe_open

    with safe_open(adapter_path, framework="pt", device="cpu") as tensors:
        keys = list(tensors.keys())
        lora_a = [key for key in keys if "lora_A" in key]
        lora_b = [key for key in keys if "lora_B" in key]
        all_finite = True
        nonzero_a = 0
        nonzero_b = 0
        for key in keys:
            tensor = tensors.get_tensor(key)
            all_finite = all_finite and bool(tensor.isfinite().all())
            if key in lora_a:
                nonzero_a += int(bool((tensor != 0).any()))
            if key in lora_b:
                nonzero_b += int(bool((tensor != 0).any()))
    tensor_checks = {
        "tensor_count": len(keys),
        "lora_a_tensors": len(lora_a),
        "nonzero_lora_a_tensors": nonzero_a,
        "lora_b_tensors": len(lora_b),
        "nonzero_lora_b_tensors": nonzero_b,
        "all_finite": all_finite,
        "all_lora_a_nonzero": bool(lora_a) and nonzero_a == len(lora_a),
        "all_lora_b_updated": bool(lora_b) and nonzero_b == len(lora_b),
    }
    passed = all(boundaries.values()) and all([
        tensor_checks["all_finite"], tensor_checks["all_lora_a_nonzero"],
        tensor_checks["all_lora_b_updated"],
    ])
    return {
        "schema_version": "labelagent.qwen3vl2b-lora-smoke-audit.v1",
        "status": "passed" if passed else "failed",
        "boundaries": boundaries,
        "tensor_checks": tensor_checks,
        "adapter_bytes": adapter_path.stat().st_size,
        "adapter_sha256": file_sha256(adapter_path),
        "receipt_sha256": file_sha256(receipt_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--write", type=Path)
    parser.add_argument("--expected-status", choices=["smoke_complete", "training_complete"],
                        default="smoke_complete")
    parser.add_argument("--expected-steps", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.expected_steps <= 100:
        raise ValueError("unsafe expected step count")
    result = audit(args.output_dir.resolve(), args.dataset_dir.resolve(),
                   args.expected_status, args.expected_steps)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.write:
        args.write.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
