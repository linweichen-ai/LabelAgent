#!/usr/bin/env python3
"""Render read-only pallet consolidation proposals from semantic review evidence.

This tool never writes a replacement candidate collection.  It turns a bounded
geometry gate plus a high-confidence semantic review into an inspectable
original/before/proposed contact sheet and records every safety check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.audit_pallet_instance_fragmentation import (
    _draw_shape,
    _fit,
    _local_mask,
    _parts,
)
from tools.review_pallet_pair_semantics import _shape_lookup


def _bbox(shape: dict) -> tuple[float, float, float, float]:
    points = [point for part in _parts(shape) for point in part.get("points", [])]
    if len(points) < 3:
        raise ValueError("candidate has no valid polygon")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _component_count(mask: np.ndarray) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    return sum(int(stats[index, cv2.CC_STAT_AREA]) >= 9
               for index in range(1, count))


def _union_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    rgba[mask != 0] = (30, 170, 255, 88)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(outline)
    for contour in contours:
        points = [tuple(map(int, point[0])) for point in contour]
        if len(points) >= 2:
            draw.line(points + [points[0]], fill=(30, 170, 255, 255), width=3)
    return Image.alpha_composite(
        Image.alpha_composite(image.convert("RGBA"), Image.fromarray(rgba, "RGBA")),
        outline).convert("RGB")


def render_proposals(candidate_file: Path, semantic_report: Path,
                     output_dir: Path) -> dict:
    candidate_before = candidate_file.read_bytes()
    candidate_sha = hashlib.sha256(candidate_before).hexdigest()
    payload = json.loads(candidate_before.decode("utf-8"))
    semantic_bytes = semantic_report.read_bytes()
    semantic = json.loads(semantic_bytes.decode("utf-8"))
    if semantic.get("candidate_sha256") != candidate_sha:
        raise ValueError("semantic review is not bound to this candidate file")
    pair_report = Path(semantic["pair_report"])
    pair_bytes = pair_report.read_bytes()
    if hashlib.sha256(pair_bytes).hexdigest() != semantic.get("pair_report_sha256"):
        raise ValueError("pair report changed after semantic review")
    pairs = json.loads(pair_bytes.decode("utf-8"))
    pair_by_rank = {int(row["rank"]): row for row in pairs.get("rows") or []}

    panels = []
    rows = []
    font = ImageFont.load_default()
    for result in semantic.get("results") or []:
        decision = result.get("decision") or {}
        eligible = (
            result.get("geometry_gate") is True
            and decision.get("larger_candidate_is_visible_pallet") is True
            and decision.get("smaller_candidate_is_visible_pallet_part") is True
            and decision.get("visual_evidence_is_clear") is True
            and decision.get("same_physical_instance") == "yes"
            and decision.get("smaller_is_redundant_fragment") is True
            and decision.get("safe_to_consolidate") is True
            and float(decision.get("confidence", 0)) >= .9)
        if not eligible:
            continue
        rank = int(result["rank"])
        pair = pair_by_rank.get(rank)
        if pair is None:
            raise ValueError(f"rank {rank} missing from bound pair report")
        small = _shape_lookup(
            payload, result["topic"], int(result["frame"]), result["small_id"])
        large = _shape_lookup(
            payload, result["topic"], int(result["frame"]), result["large_id"])
        image_path = Path(pair["image_path"])
        image = Image.open(image_path).convert("RGB")
        full_crop = (0, 0, image.width, image.height)
        small_mask = _local_mask(small, full_crop)
        large_mask = _local_mask(large, full_crop)
        union = np.logical_or(small_mask, large_mask).astype(np.uint8)
        small_pixels = int(np.count_nonzero(small_mask))
        large_pixels = int(np.count_nonzero(large_mask))
        union_pixels = int(np.count_nonzero(union))
        expansion_pixels = union_pixels - large_pixels
        expansion_ratio = expansion_pixels / max(large_pixels, 1)
        large_components = _component_count(large_mask)
        union_components = _component_count(union)
        proposal_gate = (
            expansion_ratio <= .03
            and union_components == large_components
            and pair.get("conservative_duplicate_gate") is True)

        boxes = (_bbox(small), _bbox(large))
        left = math.floor(min(box[0] for box in boxes))
        top = math.floor(min(box[1] for box in boxes))
        right = math.ceil(max(box[2] for box in boxes))
        bottom = math.ceil(max(box[3] for box in boxes))
        pad = max(28, round(max(right - left, bottom - top) * .28))
        crop = (max(0, left - pad), max(0, top - pad),
                min(image.width, right + pad + 1),
                min(image.height, bottom + pad + 1))

        before_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        _draw_shape(before_layer, large, (38, 217, 133, 72), (38, 217, 133, 255))
        _draw_shape(before_layer, small, (255, 148, 32, 96), (255, 148, 32, 255))
        before = Image.alpha_composite(
            image.convert("RGBA"), before_layer).convert("RGB")
        proposed = _union_overlay(image, union)
        views = [
            _fit(image.crop(crop), (400, 300)),
            _fit(before.crop(crop), (400, 300)),
            _fit(proposed.crop(crop), (400, 300)),
        ]
        panel = Image.new("RGB", (1200, 350), "white")
        for index, view in enumerate(views):
            panel.paste(view, (index * 400, 50))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 6),
                  f"#{rank} {result['topic']} frame {result['frame']}  "
                  f"small={result['small_id']} large={result['large_id']}",
                  fill="black", font=font)
        draw.text((8, 27),
                  f"confidence={float(decision['confidence']):.3f}  "
                  f"expansion={expansion_ratio:.4%}  components "
                  f"{large_components}->{union_components}  "
                  f"proposal_gate={proposal_gate}", fill="black", font=font)
        draw.text((18, 55), "original", fill="white", font=font)
        draw.text((418, 55), "before: green + orange", fill="white", font=font)
        draw.text((818, 55), "proposal only: cyan union", fill="white", font=font)
        panels.append(panel)
        rows.append({
            "rank": rank, "topic": result["topic"], "frame": result["frame"],
            "small_id": result["small_id"], "large_id": result["large_id"],
            "confidence": float(decision["confidence"]),
            "small_pixels": small_pixels, "large_pixels": large_pixels,
            "union_pixels": union_pixels, "union_expansion_pixels": expansion_pixels,
            "union_expansion_ratio": round(expansion_ratio, 8),
            "large_component_count": large_components,
            "union_component_count": union_components,
            "proposal_gate": proposal_gate,
            "automatic_apply_allowed": False,
            "image_path": str(image_path.resolve()),
        })

    if not panels:
        raise ValueError("semantic report contains no eligible consolidation proposal")
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (1200, sum(panel.height for panel in panels)), "white")
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    sheet_path = output_dir / "pallet-consolidation-proposals.jpg"
    sheet.save(sheet_path, quality=90, optimize=True)
    if candidate_file.read_bytes() != candidate_before:
        raise RuntimeError("candidate source changed during proposal rendering")
    report = {
        "schema_version": "labelagent.pallet_consolidation_proposal.v1",
        "candidate_file": str(candidate_file.resolve()),
        "candidate_sha256": candidate_sha,
        "semantic_report": str(semantic_report.resolve()),
        "semantic_report_sha256": hashlib.sha256(semantic_bytes).hexdigest(),
        "pair_report": str(pair_report.resolve()),
        "pair_report_sha256": hashlib.sha256(pair_bytes).hexdigest(),
        "proposal_count": len(rows),
        "proposal_gate_pass_count": sum(row["proposal_gate"] for row in rows),
        "contact_sheet": str(sheet_path.resolve()),
        "contact_sheet_sha256": hashlib.sha256(sheet_path.read_bytes()).hexdigest(),
        "candidate_only": True,
        "assistant_review_only": True,
        "automatic_apply_allowed": False,
        "human_truth": False,
        "approved_for_training": False,
        "rows": rows,
    }
    report_path = output_dir / "pallet-consolidation-proposals.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path.resolve())
    report["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--semantic-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = render_proposals(args.candidates, args.semantic_report, args.output)
    print(json.dumps({key: report[key] for key in (
        "proposal_count", "proposal_gate_pass_count", "contact_sheet",
        "contact_sheet_sha256", "report_path", "report_sha256",
        "automatic_apply_allowed")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
