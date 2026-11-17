"""Compare Transformers prefix masking with an exact greedy schema selector.

This is a decoder microbenchmark: it loads the production tokenizer and LM
Format Enforcer parser but never loads model weights.  For deterministic,
single-row, single-beam generation, selecting the maximum-scoring allowed token
is mathematically equivalent to masking every disallowed vocabulary entry and
then taking a global argmax.  Every step must choose the same target token in
both implementations or the candidate is rejected.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import statistics
import time


class GreedySchemaLogitsProcessor:
    """Force the exact greedy winner among schema-allowed tokens.

    The processor is intentionally valid only for one batch row and one beam.
    Allowed token tensors are cached by their complete, sorted token-id tuple;
    this avoids rebuilding the same device index for recurring JSON states.
    """

    def __init__(self, prefix_allowed_tokens_fn, torch_module):
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.torch = torch_module
        self.allowed_tensor_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def __call__(self, input_ids, scores):
        if input_ids.ndim != 2 or scores.ndim != 2:
            raise ValueError("greedy schema processor requires rank-two tensors")
        if input_ids.shape[0] != 1 or scores.shape[0] != 1:
            raise ValueError("greedy schema processor requires one batch row and one beam")
        allowed = self.prefix_allowed_tokens_fn(0, input_ids[0])
        if not allowed:
            raise ValueError("schema parser returned no allowed tokens")
        key = tuple(sorted(set(int(token) for token in allowed)))
        allowed_tensor = self.allowed_tensor_cache.get(key)
        if allowed_tensor is None:
            allowed_tensor = self.torch.tensor(
                key, dtype=self.torch.long, device=scores.device)
            self.allowed_tensor_cache[key] = allowed_tensor
            self.cache_misses += 1
        else:
            self.cache_hits += 1
        allowed_scores = scores[0].index_select(0, allowed_tensor)
        # Keep the winner on device. Calling item() here would introduce two
        # CPU/GPU synchronizations per generated token and erase any benefit
        # from avoiding the full-vocabulary add.
        winner_offset = self.torch.argmax(allowed_scores)
        winner = allowed_tensor[winner_offset]
        winner_score = scores[0, winner].clone()
        scores.fill_(-self.torch.inf)
        scores[0, winner] = winner_score
        return scores


class CachedSchemaMaskLogitsProcessor:
    """Cache recurring boolean vocabulary masks and apply them in place."""

    def __init__(self, prefix_allowed_tokens_fn, torch_module):
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.torch = torch_module
        self.allowed_mask_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def __call__(self, input_ids, scores):
        if input_ids.ndim != 2 or scores.ndim != 2:
            raise ValueError("cached schema mask requires rank-two tensors")
        if input_ids.shape[0] != 1 or scores.shape[0] != 1:
            raise ValueError("cached schema mask requires one batch row and one beam")
        allowed = self.prefix_allowed_tokens_fn(0, input_ids[0])
        if not allowed:
            raise ValueError("schema parser returned no allowed tokens")
        key = tuple(sorted(set(int(token) for token in allowed)))
        disallowed_mask = self.allowed_mask_cache.get(key)
        if disallowed_mask is None:
            disallowed_mask = self.torch.ones(
                scores.shape[-1], dtype=self.torch.bool, device=scores.device)
            allowed_tensor = self.torch.tensor(
                key, dtype=self.torch.long, device=scores.device)
            disallowed_mask[allowed_tensor] = False
            self.allowed_mask_cache[key] = disallowed_mask
            self.cache_misses += 1
        else:
            self.cache_hits += 1
        return scores.masked_fill_(disallowed_mask.unsqueeze(0), -self.torch.inf)


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_callback(tokenizer_data, schema):
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )

    return build_transformers_prefix_allowed_tokens_fn(
        tokenizer_data, JsonSchemaParser(schema))


def run_mode(mode, tokenizer_data, schema, prefixes, score_rows, completion,
             torch_module):
    from transformers.generation.logits_process import (
        PrefixConstrainedLogitsProcessor,
    )

    callback = build_callback(tokenizer_data, schema)
    if mode == "transformers_prefix_mask":
        processor = PrefixConstrainedLogitsProcessor(callback, num_beams=1)
    elif mode == "cached_schema_mask":
        processor = CachedSchemaMaskLogitsProcessor(callback, torch_module)
    else:
        processor = GreedySchemaLogitsProcessor(callback, torch_module)
    chosen = []
    torch_module.cuda.synchronize()
    started = time.perf_counter()
    for prefix, base_scores in zip(prefixes, score_rows):
        scores = base_scores.clone()
        processed = processor(prefix.unsqueeze(0), scores)
        chosen.append(int(torch_module.argmax(processed, dim=-1).item()))
    torch_module.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "elapsed_ms": round(elapsed_ms, 3),
        "chosen": chosen,
        "target_equal": chosen == completion,
        "mask_cache_hits": getattr(processor, "cache_hits", None),
        "mask_cache_misses": getattr(processor, "cache_misses", None),
        "mask_cache_entries": len(
            getattr(processor, "allowed_mask_cache",
                    getattr(processor, "allowed_tensor_cache", {}))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-base64", required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=529)
    parser.add_argument("--repetitions", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 20:
        raise ValueError("repetitions must be between one and twenty")
    if not 1 <= args.prompt_tokens <= 16384:
        raise ValueError("prompt-tokens must be between one and 16384")
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch
    from transformers import AutoTokenizer
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data,
    )

    fixture = json.loads(base64.b64decode(
        args.fixture_base64).decode("utf-8"))
    schema = fixture["schema"]
    target = fixture["target"]
    target_text = json.dumps(
        target, ensure_ascii=False, separators=(",", ":"))
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir), local_files_only=True)
    tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
    completion = tokenizer.encode(target_text, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    if eos is None:
        raise RuntimeError("tokenizer has no EOS token")
    completion.append(eos)
    neutral = tokenizer.encode("0", add_special_tokens=False)[-1]
    full = torch.tensor(
        [neutral] * args.prompt_tokens + completion,
        dtype=torch.long, device="cuda")
    prefixes = [full[:args.prompt_tokens + index]
                for index in range(len(completion))]

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20261117)
    vocab_size = len(tokenizer)
    score_rows = []
    for target_token in completion:
        scores = torch.randn(
            (1, vocab_size), generator=generator, device="cuda",
            dtype=torch.bfloat16)
        scores[0, target_token] = 100
        score_rows.append(scores)

    rows = []
    for repetition in range(args.repetitions):
        modes = ["transformers_prefix_mask", "cached_schema_mask",
                 "greedy_schema_selector"]
        shift = repetition % len(modes)
        order = modes[shift:] + modes[:shift]
        records = {}
        for mode in order:
            records[mode] = run_mode(
                mode, tokenizer_data, schema, prefixes, score_rows,
                completion, torch)
        baseline = records["transformers_prefix_mask"]
        cached_mask = records["cached_schema_mask"]
        candidate = records["greedy_schema_selector"]
        row = {
            "repetition": repetition + 1,
            "order": order,
            **records,
            "chosen_tokens_equal": (baseline["chosen"]
                                     == cached_mask["chosen"]
                                     == candidate["chosen"]),
            "accepted": (baseline["target_equal"]
                         and cached_mask["target_equal"]
                         and candidate["target_equal"]
                         and baseline["chosen"] == cached_mask["chosen"]
                         == candidate["chosen"]),
        }
        rows.append(row)
        print(json.dumps({
            "repetition": row["repetition"], "order": order,
            "baseline_ms": baseline["elapsed_ms"],
            "cached_mask_ms": cached_mask["elapsed_ms"],
            "greedy_ms": candidate["elapsed_ms"],
            "accepted": row["accepted"],
        }), flush=True)

    valid = [row for row in rows if row["accepted"]]
    baseline_values = [row["transformers_prefix_mask"]["elapsed_ms"]
                       for row in valid]
    cached_values = [row["cached_schema_mask"]["elapsed_ms"]
                     for row in valid]
    greedy_values = [row["greedy_schema_selector"]["elapsed_ms"]
                     for row in valid]
    baseline_median = statistics.median(baseline_values) if valid else None
    cached_median = statistics.median(cached_values) if valid else None
    greedy_median = statistics.median(greedy_values) if valid else None
    cached_saved_ms = (baseline_median - cached_median
                       if baseline_median is not None else None)
    result = {
        "schema_version": "labelagent.greedy-schema-logits-processor.v1",
        "read_only": True,
        "device": torch.cuda.get_device_name(0),
        "prompt_tokens": args.prompt_tokens,
        "completion_tokens": len(completion),
        "vocab_size": vocab_size,
        "repetitions": rows,
        "summary": {
            "valid_repetitions": len(valid),
            "baseline_median_ms": (round(baseline_median, 3)
                                    if baseline_median is not None else None),
            "baseline_p95_ms": (round(percentile(baseline_values, 95), 3)
                                 if valid else None),
            "cached_mask_median_ms": (round(cached_median, 3)
                                       if cached_median is not None else None),
            "cached_mask_p95_ms": (round(percentile(cached_values, 95), 3)
                                    if valid else None),
            "cached_mask_saved_ms": (round(cached_saved_ms, 3)
                                      if cached_saved_ms is not None else None),
            "cached_mask_saved_pct": (
                round(cached_saved_ms / baseline_median * 100, 3)
                if baseline_median else None),
            "greedy_median_ms": (round(greedy_median, 3)
                                 if greedy_median is not None else None),
            "greedy_p95_ms": (round(percentile(greedy_values, 95), 3)
                              if valid else None),
            "all_chosen_tokens_equal": len(valid) == len(rows),
            "model_loaded": False,
            "production_service_modified": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("RESULT=" + json.dumps(result["summary"]), flush=True)
    return 0 if len(valid) == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
