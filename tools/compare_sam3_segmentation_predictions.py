"""Compare two SAM3 prediction dumps against an evaluation-only COCO reference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util


def _decode_rle(value):
    rle = dict(value)
    if isinstance(rle.get("counts"), list):
        height, width = rle["size"]
        rle = mask_util.frPyObjects(rle, height, width)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("ascii")
    return mask_util.decode(rle).astype(bool)


def _decode_segmentation(value, height, width):
    if isinstance(value, list):
        return mask_util.decode(mask_util.merge(
            mask_util.frPyObjects(value, height, width))).astype(bool)
    return _decode_rle(value)


def _boundary(mask):
    padded = np.pad(mask, 1, constant_values=False)
    eroded = padded[1:-1, 1:-1].copy()
    eroded &= padded[:-2, 1:-1]
    eroded &= padded[2:, 1:-1]
    eroded &= padded[1:-1, :-2]
    eroded &= padded[1:-1, 2:]
    return mask & ~eroded


def _dilate(mask, radius=2):
    padded = np.pad(mask, radius, constant_values=False)
    result = np.zeros_like(mask)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result |= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return result


def mask_metrics(prediction, target):
    intersection = int((prediction & target).sum())
    union = int((prediction | target).sum())
    pred_area, target_area = int(prediction.sum()), int(target.sum())
    pred_boundary, target_boundary = _boundary(prediction), _boundary(target)
    precision = ((pred_boundary & _dilate(target_boundary)).sum() /
                 pred_boundary.sum()) if pred_boundary.sum() else float(not target_boundary.sum())
    recall = ((target_boundary & _dilate(pred_boundary)).sum() /
              target_boundary.sum()) if target_boundary.sum() else float(not pred_boundary.sum())
    return {
        "iou": intersection / union if union else 1.0,
        "dice": (2 * intersection / (pred_area + target_area)
                 if pred_area + target_area else 1.0),
        "boundary_f1": (2 * precision * recall / (precision + recall)
                        if precision + recall else 0.0),
        "pred_area": pred_area,
        "target_area": target_area,
    }


def _by_image(rows):
    result = {}
    for row in rows:
        result.setdefault(row["image_id"], []).append(row)
    return result


def _selected_union(rows, target_counts, height, width):
    union = np.zeros((height, width), dtype=bool)
    selected = []
    for category_id, count in target_counts.items():
        candidates = sorted(
            (row for row in rows if row["category_id"] == category_id),
            key=lambda row: (-row["score"], row.get("id", 0)),
        )[:count]
        selected.extend(candidates)
        for row in candidates:
            union |= _decode_rle(row["segmentation"])
    return union, selected


def compare(coco, baseline_rows, candidate_rows):
    annotations = _by_image(coco["annotations"])
    baseline = _by_image(baseline_rows)
    candidate = _by_image(candidate_rows)
    categories = sorted(row["id"] for row in coco["categories"])
    frames = []
    for image in sorted(coco["images"], key=lambda row: row["file_name"]):
        image_id, height, width = image["id"], image["height"], image["width"]
        targets = annotations.get(image_id, [])
        counts = {category: sum(row["category_id"] == category for row in targets)
                  for category in categories}
        target_union = np.zeros((height, width), dtype=bool)
        for row in targets:
            target_union |= _decode_segmentation(row["segmentation"], height, width)
        base_union, base_selected = _selected_union(
            baseline.get(image_id, []), counts, height, width)
        candidate_union, candidate_selected = _selected_union(
            candidate.get(image_id, []), counts, height, width)
        base_metrics = mask_metrics(base_union, target_union)
        candidate_metrics = mask_metrics(candidate_union, target_union)
        invalid = {
            "baseline_empty": int(sum(not _decode_rle(row["segmentation"]).any()
                                      for row in base_selected)),
            "candidate_empty": int(sum(not _decode_rle(row["segmentation"]).any()
                                       for row in candidate_selected)),
            "baseline_full": int(sum(_decode_rle(row["segmentation"]).mean() > .95
                                     for row in base_selected)),
            "candidate_full": int(sum(_decode_rle(row["segmentation"]).mean() > .95
                                      for row in candidate_selected)),
        }
        catastrophic = bool(
            invalid["candidate_empty"] or invalid["candidate_full"]
            or candidate_metrics["iou"] < base_metrics["iou"] - .05
            or candidate_metrics["boundary_f1"] < base_metrics["boundary_f1"] - .05
        )
        frames.append({
            "file_name": image["file_name"], "image_id": image_id,
            "target_counts": counts, "baseline": base_metrics,
            "candidate": candidate_metrics, "invalid": invalid,
            "catastrophic_regression": catastrophic,
        })
    means = {}
    for label in ("baseline", "candidate"):
        means[label] = {metric: float(np.mean([row[label][metric] for row in frames]))
                        for metric in ("iou", "dice", "boundary_f1")}
    return {"images": len(frames), "means": means,
            "catastrophic_frames": [row["file_name"] for row in frames
                                    if row["catastrophic_regression"]],
            "frames": frames}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    values = [json.loads(path.read_text(encoding="utf-8"))
              for path in (args.coco, args.baseline, args.candidate)]
    result = compare(*values)
    result.update({
        "schema_version": "labelagent.sam3-segmentation-comparison.v1",
        "split": args.split,
        "evaluation_only": True,
        "machine_reference_only": True,
        "human_truth": False,
        "not_for_deployment": True,
        "training_return_allowed": False,
        "source_sha256": {
            "coco": _sha256(args.coco), "baseline": _sha256(args.baseline),
            "candidate": _sha256(args.candidate),
        },
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                      sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"images": result["images"], "means": result["means"],
                      "catastrophic_frames": result["catastrophic_frames"]},
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
