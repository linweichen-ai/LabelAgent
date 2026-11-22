#!/usr/bin/env python3
"""Build a read-only pixel-structure review for pallet mask candidates.

The report is an assistant triage aid only.  It verifies frozen identities and
hashes, measures fragmentation/holes/boundary complexity, and renders zoomed
review panels.  It never writes human truth or approves training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_pallet_box3d_overlap_pairs import index_manifest_files


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _component_sizes(mask: np.ndarray) -> list[int]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    return [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)]


def _hole_sizes(mask: np.ndarray) -> list[int]:
    inverted = (~mask).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8)
    border = set(np.concatenate((labels[0], labels[-1], labels[:, 0],
                                 labels[:, -1])).tolist())
    return [int(stats[index, cv2.CC_STAT_AREA])
            for index in range(1, count) if index not in border]


def structure_metrics(mask: np.ndarray, *, small_island_pixels: int = 100) -> dict:
    if mask.ndim != 2:
        raise ValueError("candidate mask must be two-dimensional")
    mask = mask.astype(bool)
    area = int(mask.sum())
    if area <= 0:
        raise ValueError("candidate mask must be non-empty")
    rows, columns = np.where(mask)
    left, right = int(columns.min()), int(columns.max())
    top, bottom = int(rows.min()), int(rows.max())
    bbox_area = (right - left + 1) * (bottom - top + 1)
    components = sorted(_component_sizes(mask), reverse=True)
    small_area = sum(size for size in components if size < small_island_pixels)
    holes = _hole_sizes(mask)
    vertical = int(np.not_equal(mask[1:], mask[:-1]).sum())
    horizontal = int(np.not_equal(mask[:, 1:], mask[:, :-1]).sum())
    perimeter = (vertical + horizontal + int(mask[0].sum())
                 + int(mask[-1].sum()) + int(mask[:, 0].sum())
                 + int(mask[:, -1].sum()))
    contour_rows, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = np.concatenate(contour_rows, axis=0) if contour_rows else None
    hull_area = float(cv2.contourArea(cv2.convexHull(points))) if points is not None else 0.0
    return {
        "area_pixels": area,
        "bbox_xyxy": [left, top, right, bottom],
        "bbox_width": right - left + 1,
        "bbox_height": bottom - top + 1,
        "bbox_fill_ratio": round(area / bbox_area, 6),
        "component_count": len(components),
        "largest_component_ratio": round(components[0] / area, 6),
        "small_island_count": sum(size < small_island_pixels for size in components),
        "small_island_area_ratio": round(small_area / area, 6),
        "hole_count": len(holes),
        "hole_area_ratio": round(sum(holes) / area, 6),
        "grid_perimeter_pixels": perimeter,
        "perimeter_area_ratio": round(perimeter / area, 6),
        "convex_fill_ratio": round(area / hull_area, 6) if hull_area > 0 else None,
    }


def structural_alerts(metrics: dict) -> list[str]:
    alerts = []
    if (metrics["component_count"] > 1
            and metrics["small_island_area_ratio"] > .01):
        alerts.append("fragmented_components")
    if metrics["hole_area_ratio"] > .08:
        alerts.append("large_internal_holes")
    if metrics["bbox_fill_ratio"] < .08:
        alerts.append("very_sparse_bbox")
    if metrics["perimeter_area_ratio"] > .8:
        alerts.append("high_boundary_complexity")
    return alerts


def _fit(image: Image.Image, size=(420, 280)) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, (16, 24, 31))
    panel.paste(fitted, ((size[0] - fitted.width) // 2,
                         (size[1] - fitted.height) // 2))
    return panel


def _crop_box(metrics: dict, width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = metrics["bbox_xyxy"]
    pad = max(24, round(max(right - left + 1, bottom - top + 1) * .25))
    return (max(0, left - pad), max(0, top - pad),
            min(width, right + pad + 1), min(height, bottom + pad + 1))


def _overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = np.asarray(image.convert("RGB")).copy()
    tint = base.copy()
    tint[mask] = (32, 220, 138)
    blended = cv2.addWeighted(base, .68, tint, .32, 0)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (255, 147, 34), 3,
                     lineType=cv2.LINE_AA)
    return Image.fromarray(blended)


def _mask_view(mask: np.ndarray) -> Image.Image:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask] = (238, 244, 241)
    return Image.fromarray(rgb)


def build(bundle_path: Path, manifest_path: Path, split_path: Path,
          output: Path) -> dict:
    if output.exists():
        raise ValueError("output already exists")
    bundle_bytes = bundle_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    split_bytes = split_path.read_bytes()
    bundle = json.loads(bundle_bytes.decode("utf-8"))
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    split = json.loads(split_bytes.decode("utf-8"))
    if (bundle.get("schema_version")
            != "labelagent.pallet_visibility_candidate_bundle.v1"
            or bundle.get("production_deployed") is not False
            or bundle.get("human_truth_writes") != 0
            or bundle.get("training_data_writes") != 0):
        raise ValueError("requires a frozen review-only candidate bundle")
    if digest(manifest_path) != bundle.get("image_manifest_sha256"):
        raise ValueError("image manifest hash differs from candidate bundle")
    if digest(split_path) != bundle.get("split_report_sha256"):
        raise ValueError("split report hash differs from candidate bundle")
    if manifest.get("evaluation_only") is not True:
        raise ValueError("requires an evaluation-only image manifest")

    images = index_manifest_files(manifest["files"])
    root = Path(manifest["local_root"])
    split_metrics = {}
    for image_row in split.get("rows", []):
        for metric in image_row.get("depth_split_metrics", []):
            key = (str(image_row["sequence"]), int(image_row["camera"]),
                   str(image_row["frame_id"]), str(metric["obj_id"]))
            split_metrics[key] = metric

    output.mkdir(parents=True)
    panels_dir = output / "panels"
    panels_dir.mkdir()
    rows = []
    panels = []
    font = ImageFont.load_default()
    for candidate in bundle["rows"]:
        key = (str(candidate["sequence"]), int(candidate["camera"]),
               str(candidate["frame_id"]), str(candidate["obj_id"]))
        split_metric = split_metrics.get(key)
        if split_metric is None:
            raise ValueError(f"candidate has no split metric: {key}")
        effective_metric = candidate.get("recovery_split_metrics") or split_metric
        base = {
            "sequence": key[0], "camera": key[1], "frame_id": key[2],
            "obj_id": candidate["obj_id"], "route": candidate["route"],
            "review_required": True, "human_truth": False,
            "approved_for_training": False,
            "split_metrics": effective_metric,
        }
        if candidate.get("recovery_split_metrics"):
            base.update({
                "baseline_split_metrics": split_metric,
                "recovery_policy": candidate.get("recovery_policy"),
                "recovery_positive_point_count": candidate.get(
                    "recovery_positive_point_count"),
                "recovery_negative_point_count": candidate.get(
                    "recovery_negative_point_count"),
            })
        if not candidate.get("mask_path"):
            rows.append({**base, "mask_available": False,
                         "assistant_triage": "suppressed_no_mask"})
            continue
        mask_path = bundle_path.parent / candidate["mask_path"]
        if digest(mask_path) != candidate["mask_sha256"]:
            raise ValueError(f"candidate mask hash mismatch: {key}")
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        metrics = structure_metrics(mask)
        if metrics["area_pixels"] != int(candidate["mask_pixels"]):
            raise ValueError(f"candidate pixel count mismatch: {key}")
        image_item = images[(key[0], key[1], key[2])]
        image_path = root / image_item["relative_path"]
        if digest(image_path) != image_item["sha256"]:
            raise ValueError(f"held-out image changed: {key}")
        image = Image.open(image_path).convert("RGB")
        if mask.shape != (image.height, image.width):
            raise ValueError(f"candidate mask dimensions differ from image: {key}")
        alerts = structural_alerts(metrics)
        priority = ("high" if alerts or
                    float(effective_metric.get("source_mask_retention") or 0) < .5 or
                    float(effective_metric.get("projected_box_coverage") or 0) < .15
                    else "medium")
        crop = _crop_box(metrics, image.width, image.height)
        views = [_fit(view.crop(crop)) for view in (
            image, _overlay(image, mask), _mask_view(mask))]
        panel = Image.new("RGB", (1260, 330), "white")
        for index, view in enumerate(views):
            panel.paste(view, (index * 420, 50))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 8),
                  f"{key[0]} frame {key[2]} camera {key[1]} obj {key[3]}",
                  fill="black", font=font)
        draw.text((8, 27),
                  f"area={metrics['area_pixels']} components={metrics['component_count']} "
                  f"holes={metrics['hole_count']} perimeter/area="
                  f"{metrics['perimeter_area_ratio']:.3f} priority={priority}",
                  fill="black", font=font)
        name = f"{key[0]}-frame{key[2]}-camera{key[1]}-obj{key[3]}.jpg"
        panel_path = panels_dir / name
        panel.save(panel_path, quality=92, optimize=True)
        panels.append(panel)
        rows.append({
            **base, "mask_available": True,
            "mask_path": str(mask_path.resolve()),
            "mask_sha256": candidate["mask_sha256"],
            "structure_metrics": metrics, "structural_alerts": alerts,
            "assistant_review_priority": priority,
            "assistant_triage": "visual_review_required",
            "panel_path": str(panel_path.resolve()),
            "panel_sha256": digest(panel_path),
        })

    sheet = Image.new("RGB", (1260, sum(panel.height for panel in panels)), "white")
    top = 0
    for panel in panels:
        sheet.paste(panel, (0, top))
        top += panel.height
    sheet_path = output / "pallet-candidate-pixel-review.jpg"
    sheet.save(sheet_path, quality=92, optimize=True)
    if (bundle_path.read_bytes() != bundle_bytes
            or manifest_path.read_bytes() != manifest_bytes
            or split_path.read_bytes() != split_bytes):
        raise RuntimeError("frozen input changed during audit")
    report = {
        "schema_version": "labelagent.pallet_candidate_pixel_audit.v1",
        "status": "assistant_pixel_structure_review_not_human_truth",
        "bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "split_report_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "candidate_count": len(bundle["rows"]),
        "mask_count": sum(row["mask_available"] for row in rows),
        "suppressed_count": sum(not row["mask_available"] for row in rows),
        "high_priority_count": sum(
            row.get("assistant_review_priority") == "high" for row in rows),
        "structural_alert_count": sum(bool(row.get("structural_alerts")) for row in rows),
        "human_truth": False, "approved_for_training": False,
        "production_deployed": False, "workbench_writes": 0,
        "training_data_writes": 0,
        "contact_sheet": str(sheet_path.resolve()),
        "contact_sheet_sha256": digest(sheet_path),
        "rows": rows,
    }
    report_path = output / "results.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    report["report_path"] = str(report_path.resolve())
    report["report_sha256"] = digest(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build(args.bundle, args.manifest, args.split_report, args.output)
    print(json.dumps({key: report[key] for key in (
        "candidate_count", "mask_count", "suppressed_count",
        "high_priority_count", "structural_alert_count", "human_truth",
        "approved_for_training", "workbench_writes", "contact_sheet",
        "contact_sheet_sha256", "report_path", "report_sha256")},
        ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
