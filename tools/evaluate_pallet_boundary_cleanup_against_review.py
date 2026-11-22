#!/usr/bin/env python3
"""Evaluate a frozen pallet-cleanup pack against one bound human review.

This is a read-only, development-only gate.  Matching is chosen once from the
current masks and then reused for the alternative masks, so an alternative
cannot improve its score by silently switching to a different truth instance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.agent.edit_agent import _rasterize
from tools.audit_pallet_candidate_pixels import structure_metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_hash(path: Path, expected: str, name: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 differs")
    return actual


def _load_mask(path: Path, expected: str) -> np.ndarray:
    _require_hash(path, expected, path.name)
    with Image.open(path) as image:
        values = np.asarray(image.convert("L"))
    unique = set(np.unique(values).tolist())
    if not unique <= {0, 1, 255} or not np.any(values):
        raise ValueError(f"{path.name} is not a non-empty binary mask")
    return values > 0


def _boundary(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return mask & ~eroded


def _boundary_f1(prediction: np.ndarray, target: np.ndarray, radius: int) -> float:
    pred_boundary, target_boundary = _boundary(prediction), _boundary(target)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    target_near = cv2.dilate(target_boundary.astype(np.uint8), kernel) > 0
    pred_near = cv2.dilate(pred_boundary.astype(np.uint8), kernel) > 0
    precision = ((pred_boundary & target_near).sum() / pred_boundary.sum()
                 if pred_boundary.any() else float(not target_boundary.any()))
    recall = ((target_boundary & pred_near).sum() / target_boundary.sum()
              if target_boundary.any() else float(not pred_boundary.any()))
    return float(2 * precision * recall / (precision + recall)
                 if precision + recall else 0.0)


def mask_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    intersection = int((prediction & target).sum())
    union = int((prediction | target).sum())
    pred_area, target_area = int(prediction.sum()), int(target.sum())
    return {
        "iou": intersection / union if union else 1.0,
        "dice": (2 * intersection / (pred_area + target_area)
                 if pred_area + target_area else 1.0),
        "boundary_f1_1px": _boundary_f1(prediction, target, 1),
        "boundary_f1_2px": _boundary_f1(prediction, target, 2),
        "false_positive_pixels": int((prediction & ~target).sum()),
        "false_negative_pixels": int((~prediction & target).sum()),
        "prediction_pixels": pred_area,
        "target_pixels": target_area,
    }


def _delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, float]:
    return {key: float(after[key] - before[key]) for key in (
        "iou", "dice", "boundary_f1_1px", "boundary_f1_2px")}


def _panel(mask: np.ndarray, truth: np.ndarray, crop: tuple[int, int, int, int],
           title: str) -> Image.Image:
    left, top, right, bottom = crop
    mask, truth = mask[top:bottom, left:right], truth[top:bottom, left:right]
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[:] = (15, 24, 31)
    rgb[truth & ~mask] = (255, 92, 92)       # missed truth
    rgb[mask & ~truth] = (255, 190, 72)      # extra foreground
    rgb[mask & truth] = (45, 211, 145)       # agreement
    image = Image.fromarray(rgb)
    image.thumbnail((360, 220), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (380, 250), (9, 16, 23))
    canvas.paste(image, ((380 - image.width) // 2, 26))
    ImageDraw.Draw(canvas).text((10, 8), title, fill="white",
                                font=ImageFont.load_default())
    return canvas


def _contact_sheet(records: list[dict[str, Any]], masks: list[tuple[np.ndarray,
                   np.ndarray, np.ndarray]], output: Path) -> None:
    cards = []
    for record, (current, refined, truth) in zip(records, masks):
        union = current | refined | truth
        ys, xs = np.where(union)
        pad = 18
        crop = (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
                min(union.shape[1], int(xs.max()) + pad + 1),
                min(union.shape[0], int(ys.max()) + pad + 1))
        before = _panel(current, truth, crop,
                        f"current | IoU {record['current']['iou']:.4f}")
        after = _panel(refined, truth, crop,
                       f"cleanup | IoU {record['refined']['iou']:.4f}")
        card = Image.new("RGB", (760, 280), (9, 16, 23))
        card.paste(before, (0, 30)); card.paste(after, (380, 30))
        label = (f"pair {record['pair_index']:02d}  delta IoU "
                 f"{record['delta']['iou']:+.6f}  "
                 "green=agreement red=miss orange=extra")
        ImageDraw.Draw(card).text((10, 8), label, fill="white",
                                  font=ImageFont.load_default())
        cards.append(card)
    sheet = Image.new("RGB", (760, 280 * len(cards)), (9, 16, 23))
    for index, card in enumerate(cards):
        sheet.paste(card, (0, index * 280))
    sheet.save(output, optimize=True)


def evaluate(pack_dir: Path, sessions_path: Path, review_path: Path,
             output_dir: Path, *, expected_pack_sha256: str,
             expected_sessions_sha256: str,
             expected_review_sha256: str,
             minimum_match_iou: float = .5) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest_path = pack_dir / "manifest.json"
    hashes = {
        "pack_manifest": _require_hash(
            manifest_path, expected_pack_sha256, "pack manifest"),
        "file_sessions": _require_hash(
            sessions_path, expected_sessions_sha256, "file sessions"),
        "human_review": _require_hash(
            review_path, expected_review_sha256, "human review"),
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "labelagent.candidate-comparison-pack.v1":
        raise ValueError("unsupported comparison pack")
    if (manifest.get("human_truth") is not False
            or manifest.get("training_authorized") is not False):
        raise ValueError("comparison pack must remain candidate-only")
    if review.get("decision") != "correct_candidate":
        raise ValueError("review is not a correct-candidate decision")
    if review.get("training_return_allowed") is not False:
        raise ValueError("review unexpectedly allows training return")
    session = sessions.get(review.get("file"))
    if not isinstance(session, dict) or session.get("topic") != review.get("topic"):
        raise ValueError("reviewed file/topic is absent from sessions")
    frame = int(review["frame"])
    truth = (session.get("truth") or {}).get(str(frame))
    if not isinstance(truth, list) or canonical_sha256(truth) != review.get("truth_sha256"):
        raise ValueError("reviewed truth is absent or drifted")
    if review.get("frame_sha256") != review.get("expected_frame_sha256"):
        raise ValueError("review frame hash is internally inconsistent")
    truth_shapes = [shape for shape in truth if shape.get("label") == "pallet"]
    rows = [row for row in manifest.get("rows", []) if row.get("frame") == frame]
    if not truth_shapes or not rows:
        raise ValueError("no pallet truth or comparison rows for reviewed frame")
    if any(row.get("source_frame_sha256") != review.get("frame_sha256") for row in rows):
        raise ValueError("comparison source frame differs from reviewed frame")

    current_masks, refined_masks = [], []
    for row in rows:
        variants = row.get("variants") or []
        if len(variants) != 1:
            raise ValueError("each row must contain exactly one alternative")
        current_masks.append(_load_mask(
            pack_dir / row["current"]["mask"], row["current"]["mask_sha256"]))
        refined_masks.append(_load_mask(
            pack_dir / variants[0]["mask"], variants[0]["mask_sha256"]))
    shape = current_masks[0].shape
    if any(mask.shape != shape for mask in current_masks + refined_masks):
        raise ValueError("comparison masks do not share one image size")
    height, width = shape
    truth_masks = [_rasterize(item, width, height).astype(bool)
                   for item in truth_shapes]
    overlaps = np.zeros((len(current_masks), len(truth_masks)), dtype=float)
    for row_index, current in enumerate(current_masks):
        for truth_index, target in enumerate(truth_masks):
            intersection = int((current & target).sum())
            union = int((current | target).sum())
            overlaps[row_index, truth_index] = intersection / union if union else 1.0
    matched_rows, matched_truth = linear_sum_assignment(-overlaps)
    matches = [(int(row_index), int(truth_index))
               for row_index, truth_index in zip(matched_rows, matched_truth)
               if overlaps[row_index, truth_index] >= minimum_match_iou]
    if len(matches) != len(rows):
        raise ValueError("not every comparison row has an unambiguous truth match")

    records, render_masks = [], []
    for pair_index, (row_index, truth_index) in enumerate(matches, start=1):
        row, current = rows[row_index], current_masks[row_index]
        refined, target = refined_masks[row_index], truth_masks[truth_index]
        current_metrics = mask_metrics(current, target)
        refined_metrics = mask_metrics(refined, target)
        current_structure, refined_structure = (
            structure_metrics(current), structure_metrics(refined))
        record = {
            "pair_index": pair_index,
            "candidate_id": row.get("current_candidate_id"),
            "truth_shape_id": truth_shapes[truth_index].get("id"),
            "fixed_match_iou": float(overlaps[row_index, truth_index]),
            "current": current_metrics,
            "refined": refined_metrics,
            "delta": _delta(refined_metrics, current_metrics),
            "structure": {"current": current_structure,
                          "refined": refined_structure},
            "new_foreground_on_truth_background": int(
                (refined & ~current & ~target).sum()),
            "removed_truth_foreground": int(
                (current & ~refined & target).sum()),
            "regression_gate_failed": bool(
                refined_metrics["iou"] < current_metrics["iou"] - .01
                or refined_metrics["boundary_f1_2px"]
                < current_metrics["boundary_f1_2px"] - .02
                or refined_structure["small_island_count"]
                > current_structure["small_island_count"]),
        }
        records.append(record)
        render_masks.append((current, refined, target))

    metric_names = ("iou", "dice", "boundary_f1_1px", "boundary_f1_2px")
    aggregate = {}
    for metric in metric_names:
        before = float(np.mean([row["current"][metric] for row in records]))
        after = float(np.mean([row["refined"][metric] for row in records]))
        aggregate[metric] = {"current": before, "refined": after,
                             "delta": after - before}
    improved = sum(row["delta"]["iou"] > 1e-12 for row in records)
    unchanged = sum(abs(row["delta"]["iou"]) <= 1e-12 for row in records)
    regressed = len(records) - improved - unchanged
    gate_failures = sum(row["regression_gate_failed"] for row in records)
    changed_rows = sum(
        not np.array_equal(current, refined)
        for current, refined, _ in render_masks)
    effect_exercised = changed_rows > 0
    development_gate_passed = bool(
        effect_exercised
        and gate_failures == 0
        and aggregate["iou"]["delta"] >= 0
        and aggregate["boundary_f1_2px"]["delta"] >= 0)
    status = (
        "development_gate_passed" if development_gate_passed else
        "non_regression_replay_passed" if not effect_exercised else
        "development_gate_failed")
    report = {
        "schema_version": "labelagent.pallet-boundary-human-review-comparison.v2",
        "status": status,
        "development_gate_passed": development_gate_passed,
        "non_regression_replay_passed": not effect_exercised,
        "effect_exercised": effect_exercised,
        "scope": "single_frame_self_declared_human_review_candidate",
        "generalization_claim_allowed": False,
        "independent_human_truth": False,
        "training_authorized": False,
        # One self-declared development frame can reject a candidate, but it
        # can never authorize production deployment by itself.
        "production_deployment_authorized": False,
        "matching": {"method": "maximum_total_current_iou_hungarian",
                     "minimum_iou": minimum_match_iou,
                     "mapping_frozen_before_alternative_scoring": True,
                     "matched": len(matches), "comparison_rows": len(rows),
                     "truth_pallets": len(truth_shapes)},
        "review": {key: review.get(key) for key in (
            "review_id", "identity_assurance", "truth_role", "dataset_split",
            "split_policy", "training_return_allowed", "truth_sha256",
            "frame_sha256")},
        "source_sha256": hashes,
        "aggregate": aggregate,
        "outcomes": {"improved_iou": improved, "unchanged_iou": unchanged,
                     "regressed_iou": regressed,
                     "regression_gate_failures": gate_failures,
                     "changed_rows": changed_rows},
        "rows": records,
    }
    output_dir.mkdir(parents=True)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                      sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "per_instance.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("pair", "candidate_id", "truth_id", "current_iou",
                         "refined_iou", "delta_iou", "current_bf1_1px",
                         "refined_bf1_1px", "current_bf1_2px",
                         "refined_bf1_2px", "new_fg_on_truth_bg",
                         "removed_truth_fg", "gate_failed"))
        for row in records:
            writer.writerow((row["pair_index"], row["candidate_id"],
                             row["truth_shape_id"], row["current"]["iou"],
                             row["refined"]["iou"], row["delta"]["iou"],
                             row["current"]["boundary_f1_1px"],
                             row["refined"]["boundary_f1_1px"],
                             row["current"]["boundary_f1_2px"],
                             row["refined"]["boundary_f1_2px"],
                             row["new_foreground_on_truth_background"],
                             row["removed_truth_foreground"],
                             int(row["regression_gate_failed"])))
    _contact_sheet(records, render_masks, output_dir / "contact_sheet.png")
    public = {key: report[key] for key in (
        "schema_version", "status", "development_gate_passed", "scope",
        "non_regression_replay_passed", "effect_exercised",
        "generalization_claim_allowed",
        "independent_human_truth", "training_authorized",
        "production_deployment_authorized", "matching", "aggregate", "outcomes")}
    public.update({
        "private_instance_ids_published": False,
        "private_paths_published": False,
        "private_report_sha256": sha256(report_path),
        "contact_sheet_sha256": sha256(output_dir / "contact_sheet.png"),
    })
    (output_dir / "public_receipt.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", required=True, type=Path)
    parser.add_argument("--sessions", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-pack-sha256", required=True)
    parser.add_argument("--expected-sessions-sha256", required=True)
    parser.add_argument("--expected-review-sha256", required=True)
    args = parser.parse_args()
    evaluate(args.pack_dir, args.sessions, args.review, args.output,
             expected_pack_sha256=args.expected_pack_sha256,
             expected_sessions_sha256=args.expected_sessions_sha256,
             expected_review_sha256=args.expected_review_sha256)


if __name__ == "__main__":
    main()
