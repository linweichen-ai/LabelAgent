"""One-shot CUDA/BF16 smoke; no optimizer, checkpoint, or persistent process."""
from __future__ import annotations

import json

import torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    left = torch.randn((512, 512), device=device, dtype=torch.bfloat16)
    right = torch.randn((512, 512), device=device, dtype=torch.bfloat16)
    result = left @ right
    torch.cuda.synchronize()
    print(json.dumps({
        "event": "bf16_compute",
        "gpu": properties.name,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "dtype": str(result.dtype),
        "checksum": float(result.float().mean()),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
    }, sort_keys=True))
    del left, right, result
    torch.cuda.empty_cache()
    print(json.dumps({
        "event": "memory_release",
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
