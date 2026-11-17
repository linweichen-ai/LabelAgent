"""Profile PyTorch SDPA backend selection for LabelAgent decode shapes.

This diagnostic allocates only synthetic BF16 attention tensors.  It does not
load model weights, call the workbench, or write candidates/truth/training data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time


def timed_attention(torch, functional, backend, q, k, v, mask,
                    is_causal, warmups, iterations):
    context = (torch.nn.attention.sdpa_kernel(backend)
               if backend is not None else None)

    def call():
        return functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0,
            is_causal=is_causal)

    try:
        if context is None:
            for _ in range(warmups):
                output = call()
            torch.cuda.synchronize()
            samples = []
            for _ in range(iterations):
                started = time.perf_counter()
                output = call()
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - started) * 1000)
        else:
            with context:
                for _ in range(warmups):
                    output = call()
                torch.cuda.synchronize()
                samples = []
                for _ in range(iterations):
                    started = time.perf_counter()
                    output = call()
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - started) * 1000)
        return {
            "supported": True,
            "median_ms": round(statistics.median(samples), 6),
            "minimum_ms": round(min(samples), 6),
            "maximum_ms": round(max(samples), 6),
            "output_mean": round(float(output.float().mean().item()), 8),
        }
    except (RuntimeError, ValueError) as error:
        return {
            "supported": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=529)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--decode-iterations", type=int, default=100)
    parser.add_argument("--prefill-iterations", type=int, default=10)
    args = parser.parse_args()
    if min(args.prompt_tokens, args.max_new_tokens,
           args.decode_iterations, args.prefill_iterations) < 1:
        raise ValueError("all numeric arguments must be positive")

    import torch
    import torch.nn.functional as functional

    device = "cuda"
    dtype = torch.bfloat16
    heads = 32
    head_dim = 128
    cache_length = args.prompt_tokens + args.max_new_tokens
    backend_enum = torch.nn.attention.SDPBackend
    backend_names = {
        None: "default",
        backend_enum.FLASH_ATTENTION: "flash_attention",
        backend_enum.EFFICIENT_ATTENTION: "efficient_attention",
        backend_enum.CUDNN_ATTENTION: "cudnn_attention",
        backend_enum.MATH: "math",
    }
    choice_names = {value.value: name for name, value in
                    backend_enum.__members__.items()}
    cases = [
        ("prefill_static_mask", args.prompt_tokens, cache_length, True,
         False, args.prefill_iterations),
        ("decode_static_mask", 1, cache_length, True,
         False, args.decode_iterations),
        ("prefill_dynamic_causal", args.prompt_tokens,
         args.prompt_tokens, False, True, args.prefill_iterations),
        ("decode_dynamic_nomask", 1, cache_length, False,
         False, args.decode_iterations),
    ]
    rows = []
    for case_name, query_length, key_length, masked, causal, iterations in cases:
        q = torch.randn(1, heads, query_length, head_dim,
                        device=device, dtype=dtype)
        k = torch.randn(1, heads, key_length, head_dim,
                        device=device, dtype=dtype)
        v = torch.randn_like(k)
        mask = (torch.zeros(1, 1, query_length, key_length,
                            device=device, dtype=dtype)
                if masked else None)
        selected = torch._fused_sdp_choice(
            q, k, v, mask, 0.0, causal)
        measurements = {}
        for backend, name in backend_names.items():
            measurements[name] = timed_attention(
                torch, functional, backend, q, k, v, mask, causal,
                warmups=3, iterations=iterations)
        row = {
            "case": case_name,
            "query_length": query_length,
            "key_length": key_length,
            "masked": masked,
            "is_causal": causal,
            "selected_backend": choice_names.get(selected, str(selected)),
            "iterations": iterations,
            "measurements": measurements,
        }
        rows.append(row)
        print(json.dumps({
            "case": case_name,
            "selected_backend": row["selected_backend"],
            "medians_ms": {name: result.get("median_ms")
                           for name, result in measurements.items()
                           if result["supported"]},
        }), flush=True)
        del q, k, v, mask
        torch.cuda.empty_cache()

    result = {
        "schema_version": "labelagent.sdpa-backend-profile.v1",
        "read_only": True,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dtype": str(dtype),
        "heads": heads,
        "head_dim": head_dim,
        "prompt_tokens": args.prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "cache_length": cache_length,
        "rows": rows,
        "isolation": {
            "model_loaded": False,
            "production_service_modified": False,
            "candidate_written": False,
            "human_truth_modified": False,
            "training_data_modified": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n",
                          encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
