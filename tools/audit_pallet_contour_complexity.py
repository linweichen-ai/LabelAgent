#!/usr/bin/env python3
"""Rank and render complex pallet contours without changing candidates.

The geometry cleanup intentionally skips most small and medium pallets.  This
tool provides the missing visual evidence: it ranks their existing outer
polygons by scale-normalized vertex density and renders deterministic source /
overlay crops for review.  Inputs are read-only and outputs remain candidate
evidence, never human truth or training approval.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from source.agent.edit_agent import _rasterize
from source.agent.workflow import refine_pallet_candidate_geometry


def _mapping(payload: dict) -> dict:
    nested = payload.get("candidates_by_topic")
    return nested if isinstance(nested, dict) else payload


def _parts(shape: dict) -> list[dict]:
    parts = shape.get("parts")
    return parts if isinstance(parts, list) and parts else [shape]


def contour_metrics(shape: dict) -> dict | None:
    parts = _parts(shape)
    points = [point for part in parts for point in (part.get("points") or [])]
    if len(points) < 3:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    width = max(xs) - min(xs) + 1
    height = max(ys) - min(ys) + 1
    outer_vertices = sum(len(part.get("points") or []) for part in parts)
    hole_vertices = sum(
        len(hole) for part in parts for hole in (part.get("holes") or []))
    bbox_perimeter = max(2 * (width + height), 1)
    return {
        "bbox": [round(min(xs), 3), round(min(ys), 3),
                 round(max(xs), 3), round(max(ys), 3)],
        "bbox_width": round(width, 3),
        "bbox_height": round(height, 3),
        "bbox_area": round(width * height, 3),
        "outer_vertices": outer_vertices,
        "hole_vertices": hole_vertices,
        "outer_vertex_per_bbox_perimeter": round(
            outer_vertices / bbox_perimeter, 6),
    }


def rank_candidates(payload: dict, *, minimum_bbox_area: float = 1000,
                    limit: int = 16, audit_results: list[dict] | None = None,
                    size_band: str | None = None,
                    only_hole_fills: bool = False) -> list[dict]:
    audit_by_key = {
        (str(result.get("topic")), int(result.get("frame", -1)),
         str(result.get("shape_id"))): result
        for result in (audit_results or [])
    }
    rows = []
    for topic, frames in sorted(_mapping(payload).items()):
        for frame, shapes in frames.items():
            for shape in shapes:
                if str(shape.get("label", "")).casefold() != "pallet":
                    continue
                metrics = contour_metrics(shape)
                if metrics is None or metrics["bbox_area"] < minimum_bbox_area:
                    continue
                audit = audit_by_key.get((str(topic), int(frame), str(shape.get("id"))))
                if audit_results is not None and audit is None:
                    continue
                if size_band and (audit or {}).get("size_band") != size_band:
                    continue
                if only_hole_fills and int(
                        (audit or {}).get("filled_hole_count") or 0) <= 0:
                    continue
                rows.append({
                    "topic": topic,
                    "frame": int(frame),
                    "shape_id": shape.get("id"),
                    **metrics,
                    "_audit_result": audit,
                    "_shape": shape,
                })
    if audit_results is None:
        rows.sort(key=lambda row: (
            row["outer_vertex_per_bbox_perimeter"], row["outer_vertices"],
            row["bbox_area"], row["topic"], -row["frame"],
            str(row["shape_id"])), reverse=True)
    else:
        rows.sort(key=lambda row: (
            float(row["_audit_result"].get("change_ratio") or 0),
            float(row["_audit_result"].get(
                "boundary_vertex_reduction_ratio") or 0),
            row["outer_vertices"], row["topic"], -row["frame"],
            str(row["shape_id"])), reverse=True)
    return rows[:max(0, limit)]


def _camera_name(topic: str) -> str:
    match = re.search(r"/(camera_\d+)(?:/|$)", str(topic))
    if not match:
        raise ValueError(f"cannot map topic to camera directory: {topic}")
    return match.group(1)


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, (19, 27, 34))
    panel.paste(fitted, ((size[0] - fitted.width) // 2,
                         (size[1] - fitted.height) // 2))
    return panel


def _crop_box(row: dict, image: Image.Image) -> tuple[int, int, int, int]:
    left, top, right, bottom = row["bbox"]
    pad = max(24, round(max(right - left, bottom - top) * .18))
    return (max(0, int(left) - pad), max(0, int(top) - pad),
            min(image.width, int(right) + pad + 1),
            min(image.height, int(bottom) + pad + 1))


def _overlay(image: Image.Image, shape: dict) -> Image.Image:
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for part in _parts(shape):
        points = [tuple(map(float, point)) for point in (part.get("points") or [])]
        if len(points) >= 3:
            draw.polygon(points, fill=(38, 217, 133, 72),
                         outline=(255, 118, 32, 255), width=5)
        for hole in part.get("holes") or []:
            hole_points = [tuple(map(float, point)) for point in hole]
            if len(hole_points) >= 3:
                draw.line(hole_points + [hole_points[0]], fill=(255, 72, 72, 255),
                          width=4)
    return Image.alpha_composite(base, layer).convert("RGB")


def render_review(candidate_file: Path, image_root: Path, output_dir: Path, *,
                  minimum_bbox_area: float = 1000, limit: int = 16,
                  audit_report: Path | None = None,
                  size_band: str | None = None,
                  only_hole_fills: bool = False) -> dict:
    source_before = candidate_file.read_bytes()
    payload = json.loads(source_before.decode("utf-8"))
    audit_results = None
    if audit_report is not None:
        audit_payload = json.loads(audit_report.read_text(encoding="utf-8"))
        audit_results = audit_payload.get("results")
        if not isinstance(audit_results, list):
            raise ValueError(f"audit report has no results list: {audit_report}")
    selected = rank_candidates(
        payload, minimum_bbox_area=minimum_bbox_area, limit=limit,
        audit_results=audit_results, size_band=size_band,
        only_hole_fills=only_hole_fills)
    if not selected:
        raise ValueError("no pallet candidates satisfy the contour review scope")

    inventories: dict[str, list[Path]] = {}
    panels = []
    public_rows = []
    font = ImageFont.load_default()
    for rank, row in enumerate(selected, start=1):
        camera = _camera_name(row["topic"])
        if camera not in inventories:
            files = sorted((image_root / camera).glob("*.png"))
            if not files:
                raise ValueError(f"no PNG images for {camera}: {image_root / camera}")
            inventories[camera] = files
        files = inventories[camera]
        if row["frame"] < 0 or row["frame"] >= len(files):
            raise ValueError(f"frame {row['frame']} outside image inventory for {camera}")
        image_path = files[row["frame"]]
        image = Image.open(image_path).convert("RGB")
        crop = _crop_box(row, image)
        original = _fit(image.crop(crop), (420, 280))
        overlay = _fit(_overlay(image, row["_shape"]).crop(crop), (420, 280))
        image_rgb = np.asarray(image)
        proposed, audit = refine_pallet_candidate_geometry(
            copy.deepcopy(row["_shape"]), image.width, image.height,
            image_rgb=image_rgb)
        before_mask = _rasterize(
            row["_shape"], image.width, image.height).astype(np.uint8)
        after_mask = _rasterize(
            proposed, image.width, image.height).astype(np.uint8)
        changed_pixels = int(np.count_nonzero(before_mask != after_mask))
        before_pixels = int(np.count_nonzero(before_mask))
        change_fraction = changed_pixels / max(before_pixels, 1)
        proposed_metrics = contour_metrics(proposed) or {}
        expected_changed_pixels = (
            int(row["_audit_result"]["changed_pixels"])
            if row.get("_audit_result") is not None else None)
        audit_replay_match = (
            expected_changed_pixels is None
            or expected_changed_pixels == changed_pixels)
        if not audit_replay_match:
            raise RuntimeError(
                "rendered proposal does not match the image-guided audit for "
                f"{row['topic']} frame {row['frame']} shape {row['shape_id']}: "
                f"expected {expected_changed_pixels} changed pixels, got "
                f"{changed_pixels}")
        proposed_overlay = _fit(_overlay(image, proposed).crop(crop), (420, 280))
        header = 48
        panel = Image.new("RGB", (1260, 280 + header), "white")
        panel.paste(original, (0, header))
        panel.paste(overlay, (420, header))
        panel.paste(proposed_overlay, (840, header))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 7),
                  f"#{rank} {camera} frame {row['frame']}  {row['shape_id']}",
                  fill="black", font=font)
        draw.text((8, 25),
                  f"outer vertices={row['outer_vertices']}  density="
                  f"{row['outer_vertex_per_bbox_perimeter']:.4f}  "
                  f"bbox={row['bbox_width']:.0f}x{row['bbox_height']:.0f}",
                  fill="black", font=font)
        draw.text((430, 25), "before: orange=outer contour; red=holes",
                  fill="black", font=font)
        draw.text((850, 25),
                  f"proposed: changed={changed_pixels} ({change_fraction:.2%}); "
                  f"vertices={proposed_metrics.get('outer_vertices', 0)}",
                  fill="black", font=font)
        panels.append(panel)
        public_rows.append({
            key: value for key, value in row.items() if not key.startswith("_")
        } | {
            "rank": rank,
            "image_path": str(image_path.resolve()),
            "proposed": audit is not None,
            "proposed_shape_id": proposed.get("id"),
            "changed_pixels": changed_pixels,
            "expected_changed_pixels": expected_changed_pixels,
            "audit_replay_match": audit_replay_match,
            "image_guidance_used": True,
            "change_fraction_of_candidate": round(change_fraction, 8),
            "outer_vertices_after": proposed_metrics.get("outer_vertices"),
            "outer_vertex_density_after": proposed_metrics.get(
                "outer_vertex_per_bbox_perimeter"),
            "audit": audit,
        })

    sheet = Image.new("RGB", (1260, sum(panel.height for panel in panels)), "white")
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height

    output_dir.mkdir(parents=True, exist_ok=True)
    selection_slug = "top-change" if audit_results is not None else "top-complexity"
    if only_hole_fills:
        selection_slug = "hole-fills"
    if size_band:
        selection_slug += f"-{size_band}"
    sheet_path = output_dir / f"pallet-{selection_slug}-before-after.jpg"
    sheet.save(sheet_path, quality=90, optimize=True)
    if candidate_file.read_bytes() != source_before:
        raise RuntimeError("candidate source changed during read-only review")
    report = {
        "schema_version": "labelagent.pallet_contour_complexity_review.v2",
        "candidate_file": str(candidate_file.resolve()),
        "candidate_sha256": hashlib.sha256(source_before).hexdigest(),
        "image_root": str(image_root.resolve()),
        "minimum_bbox_area": minimum_bbox_area,
        "selected_count": len(public_rows),
        "adjusted_count": sum(row["proposed"] for row in public_rows),
        "maximum_change_fraction": max(
            row["change_fraction_of_candidate"] for row in public_rows),
        "selection": ("audit_rows_with_filled_holes" if only_hole_fills
                      else "highest_cleanup_change_ratio"
                      if audit_results is not None
                      else "highest_scale_normalized_outer_vertex_density"),
        "audit_report": (str(audit_report.resolve()) if audit_report else None),
        "size_band": size_band,
        "only_hole_fills": only_hole_fills,
        "image_guidance_used": True,
        "audit_replay_match": all(
            row["audit_replay_match"] for row in public_rows),
        "contact_sheet": str(sheet_path.resolve()),
        "contact_sheet_sha256": hashlib.sha256(sheet_path.read_bytes()).hexdigest(),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "rows": public_rows,
    }
    report_path = output_dir / f"pallet-{selection_slug}-review.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path.resolve())
    report["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-bbox-area", type=float, default=1000)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--size-band", choices=("tiny_skipped", "small", "medium", "large"))
    parser.add_argument(
        "--only-hole-fills", action="store_true",
        help="Review only candidates whose full audit filled at least one hole.")
    args = parser.parse_args()
    report = render_review(
        args.candidates, args.image_root, args.output,
        minimum_bbox_area=args.minimum_bbox_area, limit=args.limit,
        audit_report=args.audit_report, size_band=args.size_band,
        only_hole_fills=args.only_hole_fills)
    print(json.dumps({key: report[key] for key in (
        "selected_count", "selection", "contact_sheet", "contact_sheet_sha256",
        "report_path", "report_sha256", "candidate_only", "human_truth",
        "approved_for_training")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
