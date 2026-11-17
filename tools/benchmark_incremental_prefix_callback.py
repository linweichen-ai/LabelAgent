"""Benchmark a greedy-only incremental LMFE callback without loading a model.

The current Transformers integration converts the full CUDA prefix to a Python
list at every generated token.  This diagnostic compares that implementation
with a guarded callback that copies the full prefix once and then reads only the
new final token.  Both callbacks still use the same LM Format Enforcer parser
and must return exactly the same allowed-token list at every target token.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import statistics
import time


class GreedyIncrementalPrefixAllowedTokensFn:
    """Incrementally mirror one greedy sequence per batch row.

    The optimization is deliberately restricted to monotonically growing,
    single-beam decoding.  Unexpected lengths fall back to a full readback.
    Product integration must retain the same restriction.
    """

    def __init__(self, token_enforcer):
        self.token_enforcer = token_enforcer
        self._sequences: dict[int, list[int]] = {}
        self.full_readbacks = 0
        self.incremental_readbacks = 0
        self.same_length_reuses = 0

    def __call__(self, batch_id: int, sent):
        previous = self._sequences.get(batch_id)
        length = int(sent.shape[-1])
        if previous is None or length < len(previous) or length > len(previous) + 1:
            sequence = sent.tolist()
            self.full_readbacks += 1
        elif length == len(previous):
            sequence = previous
            self.same_length_reuses += 1
        else:
            sequence = previous + [int(sent[-1].item())]
            self.incremental_readbacks += 1
        self._sequences[batch_id] = sequence
        return self.token_enforcer.get_allowed_tokens(sequence).allowed_tokens


def _configure_parser(callback, force_field_order: bool):
    if force_field_order:
        callback.token_enforcer.root_parser.config.force_json_field_order = True
    return callback


def build_incremental(tokenizer_data, parser, force_field_order=False):
    from lmformatenforcer.tokenenforcer import TokenEnforcer

    return _configure_parser(GreedyIncrementalPrefixAllowedTokensFn(
        TokenEnforcer(tokenizer_data, parser)), force_field_order)


def build_current(tokenizer_data, parser, force_field_order=False):
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )

    return _configure_parser(build_transformers_prefix_allowed_tokens_fn(
        tokenizer_data, parser), force_field_order)


def align_required_order(schema, value):
    """Align required-key order to one already valid target JSON value."""
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object" and isinstance(value, dict):
        required = schema.get("required")
        if isinstance(required, list):
            schema["required"] = ([key for key in value if key in required]
                                  + [key for key in required if key not in value])
        properties = schema.get("properties") or {}
        for key, child_value in value.items():
            if key in properties:
                align_required_order(properties[key], child_value)
    elif schema.get("type") == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        for item in value:
            align_required_order(item_schema, item)


def run_callback(callback, prefixes, completion, torch_module) -> dict:
    mismatches = []
    target_rejections = []
    torch_module.cuda.synchronize()
    started = time.perf_counter()
    allowed_rows = []
    for index, (prefix, target_token) in enumerate(zip(prefixes, completion)):
        allowed = callback(0, prefix)
        allowed_rows.append(allowed)
        if target_token not in allowed:
            target_rejections.append(index)
    torch_module.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    allowed_counts = [len(allowed) for allowed in allowed_rows]
    return {
        "elapsed_ms": round(elapsed_ms, 3),
        "allowed_rows": allowed_rows,
        "target_rejections": target_rejections,
        "full_readbacks": getattr(callback, "full_readbacks", None),
        "incremental_readbacks": getattr(callback, "incremental_readbacks", None),
        "same_length_reuses": getattr(callback, "same_length_reuses", None),
        "mismatches": mismatches,
        "allowed_tokens_min": min(allowed_counts),
        "allowed_tokens_median": statistics.median(allowed_counts),
        "allowed_tokens_max": max(allowed_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    fixture_group = parser.add_mutually_exclusive_group(required=True)
    fixture_group.add_argument("--fixture", type=Path)
    fixture_group.add_argument("--fixture-base64")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=529)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--force-field-order", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 20:
        raise ValueError("repetitions must be between one and twenty")
    if not 1 <= args.prompt_tokens <= 16384:
        raise ValueError("prompt-tokens must be between one and 16384")

    import torch
    from transformers import AutoTokenizer
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data,
    )

    fixture_raw = (args.fixture.read_text(encoding="utf-8")
                   if args.fixture is not None
                   else base64.b64decode(args.fixture_base64).decode("utf-8"))
    fixture = json.loads(fixture_raw)
    schema = fixture["schema"]
    target = fixture["target"]
    if args.force_field_order:
        align_required_order(schema, target)
    target_text = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir), local_files_only=True)
    tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
    completion = tokenizer.encode(target_text, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    if eos is None:
        raise RuntimeError("tokenizer has no EOS token")
    completion.append(eos)
    neutral = tokenizer.encode("0", add_special_tokens=False)[-1]
    prompt = [neutral] * args.prompt_tokens
    full = torch.tensor(prompt + completion, dtype=torch.long, device="cuda")
    prefixes = [full[:args.prompt_tokens + index]
                for index in range(len(completion))]

    rows = []
    for repetition in range(args.repetitions):
        order = ["current", "incremental"]
        if repetition % 2:
            order.reverse()
        records = {}
        allowed_by_mode = {}
        for mode in order:
            callback = (build_current(tokenizer_data, JsonSchemaParser(schema),
                                      args.force_field_order)
                        if mode == "current"
                        else build_incremental(
                            tokenizer_data, JsonSchemaParser(schema),
                            args.force_field_order))
            record = run_callback(callback, prefixes, completion, torch)
            allowed_by_mode[mode] = record.pop("allowed_rows")
            records[mode] = record
        mismatch_steps = [index for index, (left, right) in enumerate(zip(
            allowed_by_mode["current"], allowed_by_mode["incremental"]))
            if left != right]
        rows.append({
            "repetition": repetition + 1,
            "order": order,
            **records,
            "allowed_token_mismatch_steps": mismatch_steps,
            "equivalent": (not mismatch_steps
                           and not records["current"]["target_rejections"]
                           and not records["incremental"]["target_rejections"]),
        })
        print(json.dumps({
            "repetition": repetition + 1,
            "order": order,
            "current_ms": records["current"]["elapsed_ms"],
            "incremental_ms": records["incremental"]["elapsed_ms"],
            "equivalent": rows[-1]["equivalent"],
        }), flush=True)

    valid = [row for row in rows if row["equivalent"]]
    current_median = (statistics.median(
        row["current"]["elapsed_ms"] for row in valid) if valid else None)
    incremental_median = (statistics.median(
        row["incremental"]["elapsed_ms"] for row in valid) if valid else None)
    result = {
        "schema_version": "labelagent.incremental-prefix-callback.v1",
        "read_only": True,
        "device": torch.cuda.get_device_name(0),
        "prompt_tokens": args.prompt_tokens,
        "completion_tokens": len(completion),
        "force_field_order": args.force_field_order,
        "repetitions": rows,
        "summary": {
            "equivalent_repetitions": len(valid),
            "current_median_ms": (round(current_median, 3)
                                  if current_median is not None else None),
            "incremental_median_ms": (round(incremental_median, 3)
                                      if incremental_median is not None else None),
            "saved_ms": (round(current_median - incremental_median, 3)
                         if valid else None),
            "saved_pct": (round((current_median - incremental_median)
                                 / current_median * 100, 3)
                          if valid and current_median else None),
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
