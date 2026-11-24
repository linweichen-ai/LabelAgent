"""Compare external-VLM token usage across synthetic image resolutions."""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.expert_vlm_smoke import run_smoke


SCHEMA_VERSION = "labelagent.external-vlm-synthetic-resolution-matrix.v1"


def parse_size(value):
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (AttributeError, TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("size must use WIDTHxHEIGHT") from error
    if not 128 <= width <= 2048 or not 128 <= height <= 2048:
        raise argparse.ArgumentTypeError("size dimensions must be in 128..2048")
    return width, height


def run_matrix(*, provider, url, model, key_env, sizes, execute=False,
               acknowledge_synthetic_upload=False, environ=None,
               smoke_fn=run_smoke):
    if (not isinstance(sizes, (list, tuple)) or not 2 <= len(sizes) <= 6
            or len(set(sizes)) != len(sizes)):
        raise ValueError("resolution matrix requires 2-6 unique sizes")
    environ = os.environ if environ is None else environ
    rows = []
    for width, height in sizes:
        receipt = smoke_fn(
            provider=provider, url=url, model=model, key_env=key_env,
            execute=execute,
            acknowledge_synthetic_upload=acknowledge_synthetic_upload,
            environ=environ, width=width, height=height)
        metadata = receipt.get("expert_metadata", {}).get("external_vlm", {})
        usage = metadata.get("usage", {})
        rows.append({
            "provider": receipt.get("provider"),
            "width": width, "height": height,
            "pixels": width * height,
            "status": receipt.get("status", "dry_run"),
            "latency_ms": receipt.get("latency_ms"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "synthetic_image_sha256": receipt["synthetic_image_sha256"],
            "factory_data_uploaded": receipt["factory_data_uploaded"],
        })
    executed = execute is True
    latency_values = [row["latency_ms"] for row in rows
                      if isinstance(row["latency_ms"], (int, float))]
    usage_complete = all(type(row["total_tokens"]) is int for row in rows)
    canonical_providers = {row["provider"] for row in rows
                           if isinstance(row["provider"], str)}
    if len(canonical_providers) > 1:
        raise ValueError("smoke receipts used inconsistent providers")
    lowest = (min(rows, key=lambda row: row["total_tokens"])
              if usage_complete else None)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "execute" if executed else "dry_run",
        "provider": (next(iter(canonical_providers))
                     if canonical_providers else provider),
        "model": model,
        "rows": rows,
        "summary": {
            "sample_count": len(rows),
            "usage_complete": usage_complete,
            "latency_mean_ms": (round(statistics.mean(latency_values), 3)
                                if latency_values else None),
            "lowest_token_size": ({"width": lowest["width"],
                                   "height": lowest["height"],
                                   "total_tokens": lowest["total_tokens"]}
                                  if lowest else None),
            "quality_evaluated": False,
            "deployment_recommendation": False,
        },
        "factory_data_uploaded": False,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "test24_used": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--key-env", required=True)
    parser.add_argument("--sizes", nargs="+", type=parse_size, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-synthetic-upload", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run_matrix(
        provider=args.provider, url=args.url, model=args.model,
        key_env=args.key_env, sizes=args.sizes, execute=args.execute,
        acknowledge_synthetic_upload=args.acknowledge_synthetic_upload)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

