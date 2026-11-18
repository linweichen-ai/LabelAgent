"""Benchmark legacy full-frame versus bounded candidate-mask filtering."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from PIL import Image

from source.agent.edit_agent import _rasterize
from tools.evaluate_reference_point_scan import (
    filtered_candidates, fixed_point_grid, known_coverage,
    known_instance_masks, point_batch)
from tools.tiled_candidate_utils import _iou, _mask_record


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = (REPOSITORY_ROOT / "artifacts").resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_filtered_candidates(batch, width, height, known_masks):
    raw = []
    for item in batch["items"]:
        result = item.get("result")
        if not result or not result.get("parts"):
            continue
        shape = {
            "id": len(raw) + 1, "label": "unknown_object",
            "points": result["points"], "holes": result.get("holes", []),
            "parts": result["parts"], "score": float(result["score"]),
            "point": item["point"],
        }
        record = _mask_record(shape, width, height)
        shape.update(_record=record, bbox=list(record["bbox"]),
                     area=record["area"])
        raw.append(shape)
    minimum_area = max(32, round(width * height * 0.000033))
    geometric = [shape for shape in raw
                 if shape["score"] >= 0.80
                 and minimum_area <= shape["area"] <= width * height * 0.08
                 and shape["bbox"][2] - shape["bbox"][0] <= width * 0.5
                 and shape["bbox"][3] - shape["bbox"][1] <= height * 0.5]
    kept = []
    for shape in sorted(geometric, key=lambda item: item["score"], reverse=True):
        if any(_iou(shape["_record"], prior["_record"]) >= 0.85
               for prior in kept):
            continue
        mask = _rasterize(shape, width, height).astype(bool)
        coverage = known_coverage(mask, shape["area"], known_masks)
        shape["known_coverage"] = coverage
        shape["passes_known_exclusion"] = (
            max(coverage.values(), default=0.0) < 0.5)
        kept.append(shape)
    return raw, geometric, kept


def candidate_core(groups):
    raw, geometric, kept = groups
    return {
        "raw": len(raw), "geometric": len(geometric), "kept": len(kept),
        "rows": [{
            "id": row["id"], "point": row["point"], "score": row["score"],
            "bbox": row["bbox"], "area": row["area"],
            "known_coverage": row["known_coverage"],
            "passes_known_exclusion": row["passes_known_exclusion"],
        } for row in kept],
    }


def measured(function, repeats):
    durations, cores = [], []
    for _ in range(repeats):
        started = time.perf_counter()
        value = function()
        durations.append(time.perf_counter() - started)
        cores.append(candidate_core(value))
    if any(core != cores[0] for core in cores[1:]):
        raise RuntimeError("candidate filtering is not deterministic")
    return durations, cores[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--point-sam-url", default="http://127.0.0.1:28766")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 2 <= args.repeats <= 10:
        parser.error("--repeats must be within 2..10")
    output = args.output.resolve()
    if not output.is_relative_to(ARTIFACT_ROOT):
        raise ValueError("output must stay inside repository artifacts")
    if output.exists():
        raise FileExistsError(output)
    state_sha, image_sha = sha256(args.state), sha256(args.image)
    if image_sha != args.image_sha256.lower():
        raise ValueError("image hash differs")
    state = json.loads(args.state.read_text(encoding="utf-8"))
    config = dict(state.get("config") or {})
    image_bytes = args.image.read_bytes()
    with Image.open(args.image) as image:
        width, height = image.size
    batch = point_batch(
        image_bytes, fixed_point_grid(width, height),
        args.point_sam_url)
    known_masks, known_diagnostics = known_instance_masks(
        image_bytes, width, height, config)
    legacy_seconds, legacy_core = measured(
        lambda: legacy_filtered_candidates(
            batch, width, height, known_masks), args.repeats)
    bounded_seconds, bounded_core = measured(
        lambda: filtered_candidates(batch, width, height, known_masks),
        args.repeats)
    if legacy_core != bounded_core:
        raise RuntimeError("bounded filtering changed candidate semantics")
    legacy_median = statistics.median(legacy_seconds)
    bounded_median = statistics.median(bounded_seconds)
    result = {
        "schema_version": "labelagent.bounded-point-filter-benchmark.v1",
        "candidate_only": True,
        "human_truth": False,
        "production_deployed": False,
        "approved_for_training": False,
        "state_written": False,
        "repeats": args.repeats,
        "legacy_seconds": legacy_seconds,
        "bounded_seconds": bounded_seconds,
        "legacy_median_seconds": legacy_median,
        "bounded_median_seconds": bounded_median,
        "median_reduction_percent": (1 - bounded_median / legacy_median) * 100,
        "candidate_core_identical": True,
        "candidate_core": legacy_core,
        "point_batch_diagnostics": batch.get("diagnostics"),
        "known_prompt_diagnostics": known_diagnostics,
        "inputs": {
            "state": str(args.state.resolve()), "state_sha256": state_sha,
            "image": str(args.image.resolve()), "image_sha256": image_sha,
            "width": width, "height": height,
        },
        "inputs_unchanged": {
            "state": sha256(args.state) == state_sha,
            "image": sha256(args.image) == image_sha,
        },
        "boundary": "Filtering benchmark only; no accuracy or production acceptance is computed."
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
