"""Diagnose deterministic dense SAM crops for small or edge-truncated slopes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from source.agent.edit_agent import _rasterize, _shape_from_mask
from source.agent.workflow import (
    AnnotationAgent, _shape_summary, candidate_overlay,
    filter_slope_candidate_geometry, sam_text_multi,
    suppress_near_duplicate_candidates,
)
from tools.evaluate_slope_static_images import file_sha256, load_inputs


PROMPTS = {"baseline": "traversable surface", "ramp": "ramp"}


def dense_crop_images(image: bytes, grid_size: int = 4,
                      tile_fraction: float = 0.35) -> list[dict]:
    if not 2 <= grid_size <= 6 or not 0.25 <= tile_fraction <= 0.6:
        raise ValueError("dense crop configuration is out of bounds")
    source = Image.open(io.BytesIO(image)).convert("RGB")
    span_x = max(1, round(source.width * tile_fraction))
    span_y = max(1, round(source.height * tile_fraction))
    starts_x = [round(index * (source.width - span_x) / (grid_size - 1))
                for index in range(grid_size)]
    starts_y = [round(index * (source.height - span_y) / (grid_size - 1))
                for index in range(grid_size)]
    crops = []
    for top in starts_y:
        for left in starts_x:
            right, bottom = left + span_x, top + span_y
            crop = source.crop((left, top, right, bottom))
            encoded = io.BytesIO()
            crop.save(encoded, "JPEG", quality=95)
            crops.append({"image": encoded.getvalue(),
                          "bbox": [left, top, right, bottom],
                          "width": crop.width, "height": crop.height})
    return crops


def map_shape(shape: dict, crop: dict, full_width: int, full_height: int,
              shape_id: int) -> dict | None:
    left, top, right, bottom = crop["bbox"]
    mask = _rasterize(shape, crop["width"], crop["height"])
    full_mask = np.zeros((full_height, full_width), dtype=np.uint8)
    full_mask[top:bottom, left:right] = mask
    mapped = _shape_from_mask(full_mask, "slope", shape_id)
    if mapped is not None:
        mapped.update(score=shape.get("score"), source="sam_text_dense_crop_diagnostic",
                      crop_bbox=crop["bbox"])
    return mapped


def summarize(rows: list[dict]) -> dict:
    output = {}
    for key, text_prompt in PROMPTS.items():
        tp = fp = fn = tn = 0
        for row in rows:
            expected = row["assistant_visible_slope_candidate"]
            predicted = row["prompts"][key]["candidate_count"] > 0
            tp += expected and predicted
            fp += (not expected) and predicted
            fn += expected and not predicted
            tn += (not expected) and not predicted
        output[key] = {
            "text_prompt": text_prompt, "true_positive_images": tp,
            "false_positive_images": fp, "false_negative_images": fn,
            "true_negative_images": tn,
            "precision_on_assistant_review": tp / (tp + fp) if tp + fp else None,
            "recall_on_assistant_review": tp / (tp + fn) if tp + fn else None,
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--sam-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grid-size", default=4, type=int)
    parser.add_argument("--tile-fraction", default=0.35, type=float)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.sam_url.startswith("http://127.0.0.1:"):
        raise ValueError("SAM URL must be a loopback endpoint")
    inputs, _ = load_inputs(args.dataset_dir, args.review, args.images_dir)
    state = json.loads(args.state.read_text(encoding="utf-8"))
    config = dict(state.get("config") or {})
    config["sam3_url"] = args.sam_url.rstrip("/")
    args.output.mkdir(parents=True)
    rows = []
    for source in inputs:
        image = source["image_path"].read_bytes()
        full = Image.open(io.BytesIO(image))
        started = time.monotonic()
        by_prompt = {key: [] for key in PROMPTS}
        suppressed = {key: [] for key in PROMPTS}
        next_id = 1
        for crop_index, crop in enumerate(dense_crop_images(
                image, args.grid_size, args.tile_fraction)):
            outputs, _ = sam_text_multi(crop["image"], PROMPTS, config)
            agent = AnnotationAgent(config, crop["image"], {"prompts": []}, ["slope"], 1)
            for key in PROMPTS:
                shapes = agent.candidates(outputs[key]["results"], "slope")
                shapes, rejected = filter_slope_candidate_geometry(shapes, crop["height"])
                for item in rejected:
                    suppressed[key].append(item | {"crop_index": crop_index,
                                                    "crop_bbox": crop["bbox"]})
                for shape in shapes:
                    mapped = map_shape(shape, crop, full.width, full.height, next_id)
                    next_id += 1
                    if mapped is not None:
                        mapped["crop_index"] = crop_index
                        by_prompt[key].append(mapped)
        prompt_results = {}
        for key, shapes in by_prompt.items():
            shapes, duplicates = suppress_near_duplicate_candidates(
                shapes, full.width, full.height, preserve_class_conflicts=False)
            overlay = args.output / f"{source['sample_id']}-{key}.jpg"
            overlay.write_bytes(candidate_overlay(image, shapes))
            prompt_results[key] = {
                "text_prompt": PROMPTS[key], "candidate_count": len(shapes),
                "raw_geometry_rejections": suppressed[key],
                "duplicate_rejections": duplicates,
                "candidates": [_shape_summary(shape) | {
                    "score": shape.get("score"), "crop_index": shape.get("crop_index"),
                    "crop_bbox": shape.get("crop_bbox")}
                    for shape in shapes],
                "overlay": overlay.name,
            }
        row = {
            "sample_id": source["sample_id"],
            "image_sha256": source["image_sha256"],
            "assistant_visible_slope_candidate": source[
                "assistant_visible_slope_candidate"],
            "assistant_view": source["assistant_view"],
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "prompts": prompt_results,
        }
        rows.append(row)
        print(json.dumps({"sample_id": row["sample_id"],
                          "expected": row["assistant_visible_slope_candidate"],
                          "counts": {key: value["candidate_count"]
                                     for key, value in prompt_results.items()},
                          "elapsed_seconds": row["elapsed_seconds"]},
                         ensure_ascii=False), flush=True)
    report = {
        "schema_version": "labelagent.slope-sam-dense-crop-diagnostic.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "assistant_review_candidate_only",
        "split": "dev8", "source_sequence": "008",
        "human_truth": False, "human_confirmed": False,
        "candidate_only": True, "production_deployed": False,
        "color_dependent_prompts": False,
        "grid_size": args.grid_size, "tile_fraction": args.tile_fraction,
        "state_sha256": file_sha256(args.state),
        "review_sha256": file_sha256(args.review),
        "summary": summarize(rows), "rows": rows,
        "evidence_note": (
            "Deterministic dense-crop SAM candidates are compared with bounded "
            "assistant visual review. These are not human-truth or pixel-IoU metrics."),
    }
    (args.output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "summary": report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
