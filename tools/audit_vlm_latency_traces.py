#!/usr/bin/env python3
"""Summarize bounded VLM timing metadata without copying prompts or images."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    return round(ordered[lower] * (upper - position) + ordered[upper] * (position - lower), 3)


def summary(values):
    numeric = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if not numeric:
        return {"count": 0}
    return {
        "count": len(numeric),
        "min": round(min(numeric), 3),
        "median": round(statistics.median(numeric), 3),
        "p90": percentile(numeric, 0.90),
        "p95": percentile(numeric, 0.95),
        "max": round(max(numeric), 3),
    }


def find_paths(roots, extras):
    paths = set()
    for root in roots:
        if root.is_file() and root.name == "model_debug.json":
            paths.add(root.resolve())
        elif root.is_dir():
            paths.update(path.resolve() for path in root.rglob("model_debug.json"))
    for path in extras:
        if path.is_file():
            paths.add(path.resolve())
    return sorted(paths, key=str)


def local_metadata(record):
    output = record.get("output")
    if not isinstance(output, dict):
        return None, None
    local = output.get("local_vlm")
    if not isinstance(local, dict):
        return None, output
    stats = local.get("inference_stats")
    return (stats if isinstance(stats, dict) else {}), output


def bounded_record(record, source):
    stats, output = local_metadata(record)
    if stats is None:
        return None
    local = output.get("local_vlm", {})
    generation = stats.get("generation_seconds", local.get("seconds"))
    completion = stats.get("completion_tokens", local.get("completion_tokens"))
    prompt = stats.get("prompt_tokens", local.get("prompt_tokens"))
    image_count = local.get("image_count")
    if not isinstance(image_count, int):
        images = record.get("input", {}).get("images")
        image_count = len(images) if isinstance(images, list) else None
    raw_text = output.get("raw_text")
    parsed = output.get("parsed")
    operations = parsed.get("operations") if isinstance(parsed, dict) else None
    holes = parsed.get("hole_assessment") if isinstance(parsed, dict) else None
    elapsed_ms = record.get("elapsed_ms")
    return {
        "id": record.get("id"),
        "source": str(source),
        "status": record.get("status"),
        "purpose": record.get("purpose"),
        "frame": record.get("frame"),
        "review_attempt": record.get("review_attempt"),
        "finish_reason": output.get("finish_reason"),
        "parsed_status": parsed.get("status") if isinstance(parsed, dict) else None,
        "operation_count": len(operations) if isinstance(operations, list) else None,
        "hole_assessment_count": len(holes) if isinstance(holes, list) else None,
        "raw_text_chars": len(raw_text) if isinstance(raw_text, str) else None,
        "max_tokens": record.get("input", {}).get("max_tokens"),
        "elapsed_seconds": round(elapsed_ms / 1000, 3) if isinstance(elapsed_ms, (int, float)) else None,
        "generation_seconds": generation,
        "preprocess_seconds": stats.get("preprocess_seconds", local.get("preprocess_seconds")),
        "constraint_setup_seconds": stats.get("constraint_setup_seconds"),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "image_count": image_count,
        "image_token_count": sum(local.get("actual_image_token_counts", []))
            if isinstance(local.get("actual_image_token_counts"), list) else None,
        "cache_hit": local.get("cache_hit"),
        "coalesced": local.get("coalesced"),
        "model_route": output.get("model_route"),
        "generation_cache": stats.get("generation_cache_implementation"),
        "cache_fallback_reason": stats.get("generation_cache_fallback_reason"),
        "tokens_per_second": (round(completion / generation, 3)
            if isinstance(completion, int) and isinstance(generation, (int, float)) and generation > 0 else None),
    }


def group_rows(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get(key)].append(row)
    groups = {}
    for value, items in sorted(grouped.items(), key=lambda item: str(item[0])):
        groups[str(value)] = {
            "count": len(items),
            "generation_seconds": summary(row.get("generation_seconds") for row in items),
            "completion_tokens": summary(row.get("completion_tokens") for row in items),
            "prompt_tokens": summary(row.get("prompt_tokens") for row in items),
            "tokens_per_second": summary(row.get("tokens_per_second") for row in items),
        }
    return groups


def audit_paths(paths, long_seconds=50.0):
    """Build a bounded report from model-debug traces without copying payloads."""
    paths = sorted({Path(path).resolve() for path in paths}, key=str)
    unique, duplicates, invalid = {}, 0, []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            invalid.append({"path": str(path), "error": str(error)})
            continue
        trace = payload.get("trace") if isinstance(payload, dict) else None
        if not isinstance(trace, list):
            invalid.append({"path": str(path), "error": "trace is not an array"})
            continue
        for index, record in enumerate(trace):
            if not isinstance(record, dict) or record.get("service") != "GPT":
                continue
            row = bounded_record(record, path)
            if row is None:
                continue
            identity = record.get("id")
            if not isinstance(identity, str) or not identity:
                identity = f"{path}:{index}"
            if identity in unique:
                duplicates += 1
            else:
                unique[identity] = row

    rows = list(unique.values())
    uncached = [row for row in rows if row.get("cache_hit") is False]
    long_rows = [row for row in uncached
                 if isinstance(row.get("generation_seconds"), (int, float))
                 and row["generation_seconds"] >= long_seconds]
    top = sorted(uncached, key=lambda row: row.get("generation_seconds") or -1, reverse=True)[:30]
    return {
        "schema_version": "labelagent.vlm_latency_trace_audit.v1",
        "scope": {
            "files_scanned": len(paths),
            "invalid_files": invalid,
            "unique_calls_with_local_metadata": len(rows),
            "duplicate_copies_dropped": duplicates,
            "uncached_calls": len(uncached),
            "long_generation_threshold_seconds": long_seconds,
            "long_uncached_calls": len(long_rows),
            "content_boundary": "No prompts, raw model text, images, credentials or full file state are copied into this report."
        },
        "uncached_overall": {
            "generation_seconds": summary(row.get("generation_seconds") for row in uncached),
            "preprocess_seconds": summary(row.get("preprocess_seconds") for row in uncached),
            "prompt_tokens": summary(row.get("prompt_tokens") for row in uncached),
            "completion_tokens": summary(row.get("completion_tokens") for row in uncached),
            "tokens_per_second": summary(row.get("tokens_per_second") for row in uncached),
        },
        "by_image_count": group_rows(uncached, "image_count"),
        "by_purpose": group_rows(uncached, "purpose"),
        "by_finish_reason": group_rows(uncached, "finish_reason"),
        "by_record_status": group_rows(uncached, "status"),
        "long_uncached_calls": long_rows,
        "slowest_uncached_calls": top,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", type=Path, default=[])
    parser.add_argument("--extra", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--long-seconds", type=float, default=50.0)
    args = parser.parse_args(argv)

    paths = find_paths(args.root, args.extra)
    output = audit_paths(paths, args.long_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["scope"], ensure_ascii=False))


if __name__ == "__main__":
    main()
