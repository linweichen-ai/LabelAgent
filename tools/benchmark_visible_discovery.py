"""Benchmark visible-label discovery on explicit local images.

The benchmark reads only model settings and the label registry from an existing
LabelAgent state.  It does not load the workbench server, save candidates,
change truth, or approve anything for training.  Output excludes prompts and
model prose while retaining bounded semantic decisions and request timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only visible-label discovery latency benchmark.")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--llm-url",
                        help="Explicit model endpoint override; never saved to source state")
    parser.add_argument("--llm-model",
                        help="Explicit model-id override; never saved to source state")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bounded_assessments(result: dict) -> list[dict]:
    return [{
        "label": item.get("label"),
        "evidence": item.get("evidence"),
        "confidence": item.get("confidence"),
    } for item in result.get("assessments", []) if isinstance(item, dict)]


def bounded_unknown(result: dict) -> list[dict]:
    return [{
        "candidate_name": item.get("candidate_name"),
        "confidence": item.get("confidence"),
        "route": item.get("route"),
    } for item in result.get("unknown_candidates", []) if isinstance(item, dict)]


def bounded_consistency_flags(result: dict) -> list[dict]:
    allowed = {"label", "normalized_evidence", "reason_code", "route",
               "conflicting_labels", "shared_anchor"}
    return [{key: item.get(key) for key in allowed if key in item}
            for item in result.get("consistency_flags", [])
            if isinstance(item, dict)]


def request_metadata(trace: list[dict]) -> dict:
    for event in reversed(trace):
        if (isinstance(event, dict) and event.get("service") == "GPT"
                and event.get("status") == "succeeded"):
            output = event.get("output") or {}
            metadata = output.get("local_vlm")
            return {
                "elapsed_ms": event.get("elapsed_ms"),
                "model_route": output.get("model_route"),
                "fast_fallback_reason": output.get("fast_fallback_reason"),
                "raw_text_chars": (len(output.get("raw_text"))
                                   if isinstance(output.get("raw_text"), str)
                                   else None),
                "local_vlm": metadata if isinstance(metadata, dict) else None,
            }
    return {"elapsed_ms": None, "model_route": None,
            "fast_fallback_reason": None, "raw_text_chars": None,
            "local_vlm": None}


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def model_config(source: dict, llm_url: str | None = None,
                 llm_model: str | None = None) -> dict:
    config = dict(source or {})
    if llm_url:
        parsed = urlsplit(llm_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None):
            raise ValueError("llm-url must be an HTTP(S) endpoint without credentials")
        config["llm_url"] = llm_url.rstrip("/")
    if llm_model:
        if not llm_model.strip():
            raise ValueError("llm-model must not be blank")
        config["llm_model"] = llm_model.strip()
    return config


def main() -> int:
    args = parse_args()
    state_path = args.source_state.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    images = [path.expanduser().resolve() for path in args.images]
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    for image in images:
        if not image.is_file():
            raise FileNotFoundError(image)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    config = model_config(state.get("config") or {}, args.llm_url, args.llm_model)
    labels = list(state.get("labels") or [])
    if not config.get("llm_url") or not config.get("llm_model") or not labels:
        raise ValueError("source state lacks VLM settings or label registry")

    from agent.workflow import discover_visible_labels, trace_calls

    rows = []
    for position, image_path in enumerate(images, 1):
        trace = []
        started = time.perf_counter()
        try:
            with trace_calls(trace.append):
                result = discover_visible_labels(
                    config, image_path.read_bytes(), labels)
            failure = None
        except Exception as error:  # retain an explicit failed sample
            result = {}
            failure = {"type": type(error).__name__, "message": str(error)[:300]}
        with Image.open(image_path) as image:
            size = [image.width, image.height]
        rows.append({
            "position": position,
            "image": str(image_path),
            "image_sha256": sha256(image_path),
            "bytes": image_path.stat().st_size,
            "size": size,
            "wall_elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            **request_metadata(trace),
            "assessments": bounded_assessments(result),
            "visible_labels": result.get("visible_labels", []),
            "uncertain_labels": result.get("uncertain_labels", []),
            "unknown_candidates": bounded_unknown(result),
            "consistency_flags": bounded_consistency_flags(result),
            "open_set_scan_complete": result.get("open_set_scan_complete"),
            "registry_mutated": result.get("registry_mutated"),
            "failure": failure,
        })

    successful = [row for row in rows if row["failure"] is None]
    elapsed = sorted(float(row["elapsed_ms"]) for row in successful
                     if row["elapsed_ms"] is not None)
    summary = {
        "schema_version": "labelagent.visible_discovery_benchmark.v1",
        "read_only": True,
        "candidate_written": False,
        "human_truth_modified": False,
        "training_data_modified": False,
        "source_state_modified": False,
        "labels": labels,
        "model": config.get("llm_model"),
        "image_count": len(rows),
        "success_count": len(successful),
        "failure_count": len(rows) - len(successful),
        "elapsed_ms": {
            "minimum": min(elapsed) if elapsed else None,
            "median": median(elapsed),
            "maximum": max(elapsed) if elapsed else None,
        },
        "runs": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "image_count": len(rows),
        "success_count": len(successful),
        "failure_count": len(rows) - len(successful),
        "elapsed_ms": summary["elapsed_ms"],
    }, ensure_ascii=False))
    return 0 if len(successful) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
