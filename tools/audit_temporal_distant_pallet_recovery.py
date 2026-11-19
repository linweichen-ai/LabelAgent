#!/usr/bin/env python3
"""Read-only adjacent-frame audit for partial distant pallet candidates.

The audit transports existing SAM pallet masks from nearby frames with
bidirectional optical flow.  A transported expansion is only reported when it
passes the frozen axis and neighbouring-instance guards and is independently
supported by another source frame.  Nothing is written to the workbench,
human truth, or the training set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.agent.edit_agent import _rasterize
from tools.review_distant_pallet_recovery import (
    _camera_name,
    _crop,
    _difference_overlay,
    _fit,
    _mapping,
    _mask_overlay,
    _mask_record,
    _public_metrics,
    match_expansion,
)
from tools.temporal_candidate_utils import (
    estimate_bidirectional_flow,
    prepare_flow_warp,
    warp_candidate,
)


FLOW_CONFIG = {
    "width": 480,
    "height": 384,
    "pyr_scale": .5,
    "levels": 3,
    "winsize": 25,
    "iterations": 3,
    "poly_n": 7,
    "poly_sigma": 1.5,
    "gaussian": True,
}

WARP_SETTINGS = {
    "minimum_inherited_source_score": .5,
    "minimum_area_pixels_full_resolution": 256,
    "area_ratio_bounds": [.5, 2.0],
    "minimum_consistent_foreground_fraction": .80,
    "minimum_polygon_raster_IoU": .95,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_iou(first: dict, second: dict, width: int, height: int) -> float:
    one = _rasterize(first, width, height).astype(bool)
    two = _rasterize(second, width, height).astype(bool)
    union = int(np.count_nonzero(one | two))
    return int(np.count_nonzero(one & two)) / union if union else 0.0


def select_temporal_consensus(proposals: list[dict], width: int, height: int,
                              *, minimum_iou: float = .45,
                              minimum_source_frames: int = 2
                              ) -> tuple[dict | None, dict]:
    """Choose one transported mask only when another frame supports it."""
    if not 0 < minimum_iou <= 1:
        raise ValueError("minimum consensus IoU must be in (0, 1]")
    if minimum_source_frames < 2:
        raise ValueError("temporal consensus requires at least two frames")
    candidates = []
    for index, row in enumerate(proposals):
        supporters = []
        pairwise = {}
        for other_index, other in enumerate(proposals):
            if index == other_index or row["source_frame"] == other["source_frame"]:
                continue
            overlap = mask_iou(row["shape"], other["shape"], width, height)
            pairwise[str(other["source_frame"])] = round(overlap, 6)
            if overlap >= minimum_iou:
                supporters.append(other["source_frame"])
        source_frames = {row["source_frame"], *supporters}
        if len(source_frames) < minimum_source_frames:
            continue
        metrics = row["metrics"]
        candidates.append((
            len(source_frames),
            float(metrics["baseline_coverage"]),
            float(metrics["proposal_to_baseline_area_ratio"]),
            float(row["shape"].get("score", 0) or 0),
            str(row["shape"].get("id")),
            row,
            sorted(source_frames),
            pairwise,
        ))
    if not candidates:
        return None, {
            "accepted": False,
            "reason": "no_two_frame_consensus",
            "proposal_count": len(proposals),
            "minimum_iou": minimum_iou,
            "minimum_source_frames": minimum_source_frames,
        }
    candidates.sort(key=lambda item: item[:5], reverse=True)
    selected = candidates[0]
    return selected[5], {
        "accepted": True,
        "reason": None,
        "proposal_count": len(proposals),
        "supporting_source_frames": selected[6],
        "supporting_source_frame_count": selected[0],
        "pairwise_iou": selected[7],
        "minimum_iou": minimum_iou,
        "minimum_source_frames": minimum_source_frames,
    }


def _ranked_targets(audit: dict, limit: int) -> list[dict]:
    rows = audit.get("results")
    if not isinstance(rows, list):
        raise ValueError("audit report has no results list")
    return sorted(
        (row for row in rows
         if row.get("size_band") in {"small", "tiny_skipped"}
         and float(row.get("change_ratio") or 0) > 0),
        key=lambda row: (-float(row.get("change_ratio") or 0),
                         str(row.get("topic")), int(row.get("frame", -1)),
                         str(row.get("shape_id"))))[:limit]


def audit(baseline_path: Path, geometry_audit_path: Path, image_root: Path,
          output: Path, *, limit: int = 16, width: int = 1920,
          height: int = 1536, offsets: tuple[int, ...] = (-2, -1, 1, 2),
          minimum_consensus_iou: float = .45) -> dict:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not offsets or 0 in offsets or len(offsets) != len(set(offsets)):
        raise ValueError("offsets must be unique non-zero frame offsets")
    inputs = [baseline_path, geometry_audit_path]
    before = {str(path.resolve()): _sha256(path) for path in inputs}
    baseline = _mapping(json.loads(baseline_path.read_text(encoding="utf-8")))
    geometry = json.loads(geometry_audit_path.read_text(encoding="utf-8"))
    targets = _ranked_targets(geometry, limit)
    inventories: dict[str, list[Path]] = {}
    image_cache: dict[tuple[str, int], np.ndarray] = {}

    def image(camera: str, frame: int) -> np.ndarray:
        key = (camera, frame)
        if key not in image_cache:
            paths = inventories[camera]
            if frame < 0 or frame >= len(paths):
                raise IndexError(key)
            value = np.asarray(Image.open(paths[frame]).convert("RGB"))
            if value.shape != (height, width, 3):
                raise ValueError(f"unexpected image dimensions: {paths[frame]}")
            image_cache[key] = value
        return image_cache[key]

    rows = []
    panels = []
    font = ImageFont.load_default()
    for rank, target in enumerate(targets, start=1):
        topic, frame = str(target["topic"]), int(target["frame"])
        camera = _camera_name(topic)
        if camera not in inventories:
            inventories[camera] = sorted((image_root / camera).glob("*.png"))
        target_shapes = baseline.get(topic, {}).get(str(frame), [])
        matches = [shape for shape in target_shapes
                   if str(shape.get("id")) == str(target.get("shape_id"))]
        if len(matches) != 1:
            raise ValueError(f"target shape identity is not unique: {topic} {frame}")
        target_shape = matches[0]
        other_targets = [shape for shape in target_shapes
                         if str(shape.get("id")) != str(target.get("shape_id"))]
        accepted = []
        neighbor_audit = []
        for offset in offsets:
            source_frame = frame + offset
            source_shapes = baseline.get(topic, {}).get(str(source_frame))
            if source_shapes is None or not 0 <= source_frame < len(inventories[camera]):
                neighbor_audit.append({
                    "source_frame": source_frame,
                    "offset": offset,
                    "status": "unavailable",
                })
                continue
            forward, backward = estimate_bidirectional_flow(
                image(camera, source_frame), image(camera, frame), FLOW_CONFIG)
            warp = prepare_flow_warp(
                forward, backward, width, height, error_threshold=1.5)
            transported = []
            warp_reasons: dict[str, int] = {}
            for index, source_shape in enumerate(source_shapes):
                if str(source_shape.get("label", "")).casefold() != "pallet":
                    continue
                proposal, warp_audit = warp_candidate(
                    source_shape, warp,
                    f"temporal-{camera}-{frame}-from-{source_frame}-{index}",
                    WARP_SETTINGS)
                reason = warp_audit.get("reason") or "accepted"
                warp_reasons[reason] = warp_reasons.get(reason, 0) + 1
                if proposal is not None:
                    transported.append(proposal)
            proposal, metrics = match_expansion(
                target_shape, transported, width, height,
                minimum_baseline_coverage=.5,
                minimum_area_ratio=1.25,
                maximum_minor_axis_ratio=1.5,
                minimum_axis_elongation=2.0,
                other_baselines=other_targets,
                maximum_other_baseline_fraction=.1)
            row = {
                "source_frame": source_frame,
                "offset": offset,
                "source_pallet_count": sum(
                    str(shape.get("label", "")).casefold() == "pallet"
                    for shape in source_shapes),
                "transported_count": len(transported),
                "warp_reasons": warp_reasons,
                "qualified_expansion": proposal is not None,
                "expansion_metrics": _public_metrics(metrics),
            }
            neighbor_audit.append(row)
            if proposal is not None:
                accepted.append({
                    "source_frame": source_frame,
                    "shape": proposal,
                    "metrics": metrics,
                })
        selected, consensus = select_temporal_consensus(
            accepted, width, height,
            minimum_iou=minimum_consensus_iou,
            minimum_source_frames=2)
        current = _mask_record(target_shape, width, height)["mask"]
        proposed = (_mask_record(selected["shape"], width, height)["mask"]
                    if selected is not None else current)
        crop = _crop(current, proposed)
        source_image = Image.fromarray(image(camera, frame), "RGB")
        views = (
            _fit(source_image.crop(crop), (360, 250)),
            _fit(_mask_overlay(source_image, current, (35, 205, 130)).crop(crop),
                 (360, 250)),
            _fit(_mask_overlay(source_image, proposed, (255, 165, 35)).crop(crop),
                 (360, 250)),
            _fit(_difference_overlay(source_image, current, proposed).crop(crop),
                 (360, 250)),
        )
        panel = Image.new("RGB", (1440, 312), "white")
        for x, view in zip((0, 360, 720, 1080), views):
            panel.paste(view, (x, 62))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 7), f"#{rank} {camera} frame {frame} {target['shape_id']}",
                  fill="black", font=font)
        draw.text((8, 27),
                  f"qualified neighbors={len(accepted)} consensus={consensus['accepted']}",
                  fill="black", font=font)
        if selected is not None:
            draw.text((730, 43),
                      f"source={selected['source_frame']} support="
                      f"{consensus['supporting_source_frames']}",
                      fill="black", font=font)
        draw.text((1090, 43), "green kept, yellow added, red missing",
                  fill="black", font=font)
        panels.append(panel)
        rows.append({
            "rank": rank,
            "topic": topic,
            "frame": str(frame),
            "shape_id": target.get("shape_id"),
            "baseline_area": int(current.sum()),
            "neighbor_audit": neighbor_audit,
            "qualified_neighbor_count": len(accepted),
            "consensus": consensus,
            "selected_source_frame": selected["source_frame"] if selected else None,
            "selected_metrics": (_public_metrics(selected["metrics"])
                                 if selected else None),
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
        })
        image_cache.clear()

    if not panels:
        raise ValueError("no distant pallet rows selected")
    output.mkdir(parents=True, exist_ok=False)
    sheet = Image.new("RGB", (1440, sum(panel.height for panel in panels)), "white")
    top = 0
    for panel in panels:
        sheet.paste(panel, (0, top))
        top += panel.height
    sheet_path = output / "temporal-distant-pallet-review.jpg"
    sheet.save(sheet_path, quality=90, optimize=True)
    if any(_sha256(Path(path)) != digest for path, digest in before.items()):
        raise RuntimeError("an input changed during read-only temporal audit")
    report = {
        "schema_version": "labelagent.temporal_distant_pallet_audit.v1",
        "input_sha256": before,
        "selection": "highest_geometry_change_small_candidates",
        "selected_count": len(rows),
        "offsets": list(offsets),
        "flow_config": FLOW_CONFIG,
        "warp_settings": WARP_SETTINGS,
        "minimum_consensus_iou": minimum_consensus_iou,
        "qualified_neighbor_expansion_count": sum(
            row["qualified_neighbor_count"] for row in rows),
        "two_frame_consensus_count": sum(
            bool(row["consensus"]["accepted"]) for row in rows),
        "new_model_requests": 0,
        "workbench_writes": 0,
        "contact_sheet": str(sheet_path.resolve()),
        "contact_sheet_sha256": _sha256(sheet_path),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "rows": rows,
    }
    report_path = output / "temporal-distant-pallet-review.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    report["report_path"] = str(report_path.resolve())
    report["report_sha256"] = _sha256(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--geometry-audit", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--minimum-consensus-iou", type=float, default=.45)
    args = parser.parse_args()
    report = audit(
        args.baseline, args.geometry_audit, args.image_root, args.output,
        limit=args.limit, width=args.width, height=args.height,
        minimum_consensus_iou=args.minimum_consensus_iou)
    print(json.dumps({key: report[key] for key in (
        "selected_count", "qualified_neighbor_expansion_count",
        "two_frame_consensus_count", "new_model_requests", "workbench_writes",
        "contact_sheet", "contact_sheet_sha256", "report_path",
        "report_sha256", "candidate_only", "human_truth",
        "approved_for_training")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
