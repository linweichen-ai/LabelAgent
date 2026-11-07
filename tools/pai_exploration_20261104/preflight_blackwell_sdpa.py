"""Actual CUDA/BF16/SDPA forward+backward checks; no model weights needed."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess


def check_device(torch, index):
    device = f"cuda:{index}"
    with torch.cuda.device(index):
        assert torch.cuda.is_bf16_supported(), f"BF16 unsupported on {device}"
        matrix = torch.ones((256, 256), device=device, dtype=torch.bfloat16)
        product = matrix @ matrix
        assert float(product[0, 0]) == 256.0
        torch.manual_seed(3407)
        tensors = [torch.randn((1, 4, 128, 64), device=device,
                               dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        result = torch.nn.functional.scaled_dot_product_attention(
            *tensors, dropout_p=0.0, is_causal=True)
        assert bool(torch.isfinite(result).all()), "Nonfinite SDPA output"
        result.float().square().mean().backward()
        assert all(x.grad is not None and bool(torch.isfinite(x.grad).all())
                   and float(x.grad.float().abs().sum()) > 0 for x in tensors), "Invalid SDPA gradients"
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return {"index": index, "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "free_bytes": free, "total_bytes": total,
                "bf16_matmul_passed": True, "sdpa_forward_backward_passed": True}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", required=True)
    p.add_argument("--expected-gpus", type=int, default=2)
    a = p.parse_args()
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    import peft
    assert torch.cuda.is_available(), "CUDA unavailable"
    assert torch.cuda.device_count() == a.expected_gpus, "Unexpected visible GPU count"
    devices = [check_device(torch, i) for i in range(torch.cuda.device_count())]
    data = {"status": "CUDA_BF16_SDPA_PASSED_NOT_MODEL_TRAINING",
            "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "compiled_arches": torch.cuda.get_arch_list(), "devices": devices,
            "packages": {k: importlib.metadata.version(k) for k in ["transformers", "peft", "accelerate"]},
            "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv"], text=True),
            "third_party_flash_attention_required": False,
            "model_weights_loaded": False, "LoRA_training_verified": False}
    with Path(a.out).open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
