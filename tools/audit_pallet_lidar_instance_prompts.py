#!/usr/bin/env python3
"""Audit 3D-instance-aware LiDAR prompts for difficult pallet candidates.

The tool is deliberately prediction-only.  It reads raw point clouds,
calibration, source images and frozen machine candidates, but an audit hook
denies access to reference labels.  It never calls SAM and never writes to the
LabelAgent workbench.  Its output is an offline prompt proposal for review.
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
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.agent.edit_agent import _rasterize
from tools.audit_temporal_distant_pallet_recovery import _ranked_targets
from tools.evaluate_pallet_lidar_point_prompts import (
    _camera_name,
    _mapping,
    _sample_spread,
    select_lidar_prompt_points,
)
from tools.evaluate_projected_support import project_visible
from tools.lidar_candidate_evidence import (
    POINT_RECIPE,
    ReferenceReadAudit,
    fit_support_plane,
    read_xyz,
    sensor_cameras,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _xy_components(points: np.ndarray, radius: float) -> np.ndarray:
    """Return deterministic connected-component labels in the LiDAR XY plane."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or radius <= 0:
        raise ValueError("expected XYZ points and a positive cluster radius")
    if not len(values):
        return np.zeros(0, dtype=np.int32)
    parent = np.arange(len(values), dtype=np.int32)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for left, right in sorted(cKDTree(values[:, :2]).query_pairs(radius)):
        a, b = root(int(left)), root(int(right))
        if a != b:
            parent[max(a, b)] = min(a, b)
    roots = np.array([root(index) for index in range(len(values))])
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int32)


def select_instance_lidar_prompts(
    candidate_mask: np.ndarray,
    existing_pallet_union: np.ndarray,
    pixels: np.ndarray,
    heights: np.ndarray,
    xyz: np.ndarray,
    *,
    low_height: tuple[float, float] = (0.04, 0.45),
    floor_band: float = 0.04,
    tall_height: float = 0.65,
    seed_radius: int = 20,
    cluster_radius_m: float = 0.18,
    minimum_seed_points: int = 2,
    maximum_component_span_m: float = 1.8,
    maximum_positive_points: int = 4,
    maximum_floor_negative_points: int = 4,
    maximum_tall_negative_points: int = 2,
) -> dict:
    """Choose prompt points from one unambiguous low-height 3D component.

    Image-space dilation is used only to associate the frozen candidate with a
    seed.  Component membership is decided in LiDAR XY, which prevents two
    depth-separated objects that project next to each other from being merged.
    """
    mask = np.asarray(candidate_mask, dtype=bool)
    occupied = np.asarray(existing_pallet_union, dtype=bool)
    pixels = np.asarray(pixels, dtype=np.int32)
    heights = np.asarray(heights, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    if mask.ndim != 2 or occupied.shape != mask.shape or not mask.any():
        raise ValueError("candidate and pallet union must be aligned and nonempty")
    if (
        pixels.ndim != 2
        or pixels.shape[1] != 2
        or xyz.ndim != 2
        or xyz.shape[1] != 3
        or len(pixels) != len(heights)
        or len(pixels) != len(xyz)
    ):
        raise ValueError("projected pixels, heights and XYZ points must align")
    if cluster_radius_m <= 0 or minimum_seed_points < 1:
        raise ValueError("invalid instance clustering recipe")

    height, width = mask.shape
    finite = np.isfinite(heights) & np.isfinite(xyz).all(axis=1)
    in_bounds = (
        finite
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    pixels, heights, xyz = pixels[in_bounds], heights[in_bounds], xyz[in_bounds]
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    span = max(x1 - x0 + 1, y1 - y0 + 1)
    padding = max(48, min(192, span * 2))
    roi = (
        (pixels[:, 0] >= max(0, x0 - padding))
        & (pixels[:, 0] <= min(width - 1, x1 + padding))
        & (pixels[:, 1] >= max(0, y0 - padding))
        & (pixels[:, 1] <= min(height - 1, y1 + padding))
    )
    low = (heights >= low_height[0]) & (heights <= low_height[1])
    floor = np.abs(heights) <= floor_band
    tall = heights >= tall_height
    seed_region = cv2.dilate(
        mask.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (seed_radius * 2 + 1, seed_radius * 2 + 1)
        ),
    ) > 0
    low_indices = np.flatnonzero(low & roi)
    if not len(low_indices):
        return {"ready": False, "reason": "no_low_height_points_in_roi"}
    labels = _xy_components(xyz[low_indices], cluster_radius_m)
    seed_local = seed_region[
        pixels[low_indices, 1], pixels[low_indices, 0]
    ]
    seed_labels = labels[seed_local]
    if not len(seed_labels):
        return {
            "ready": False,
            "reason": "no_low_height_seed_near_candidate",
            "low_points_in_roi": int(len(low_indices)),
        }
    unique, counts = np.unique(seed_labels, return_counts=True)
    ordering = np.argsort(-counts, kind="stable")
    winner = int(unique[ordering[0]])
    winner_count = int(counts[ordering[0]])
    runner_up_count = int(counts[ordering[1]]) if len(ordering) > 1 else 0
    if winner_count < minimum_seed_points:
        return {
            "ready": False,
            "reason": "insufficient_3d_seed_points",
            "winning_seed_points": winner_count,
            "runner_up_seed_points": runner_up_count,
        }
    if runner_up_count == winner_count:
        return {
            "ready": False,
            "reason": "ambiguous_low_height_components",
            "winning_seed_points": winner_count,
            "runner_up_seed_points": runner_up_count,
        }

    component_indices = low_indices[labels == winner]
    component_xyz = xyz[component_indices]
    component_pixels = pixels[component_indices]
    xy_span = np.ptp(component_xyz[:, :2], axis=0)
    if float(np.max(xy_span)) > maximum_component_span_m:
        return {
            "ready": False,
            "reason": "low_height_component_too_large",
            "component_xy_span_m": xy_span.tolist(),
        }
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2], dtype=float)
    positives = _sample_spread(
        component_pixels, maximum_positive_points, anchor=center
    )
    if not positives:
        return {"ready": False, "reason": "empty_3d_component_projection"}

    image_exclusion = cv2.dilate(
        occupied.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33)),
    ) > 0
    tree = cKDTree(component_xyz[:, :2])
    distances, _ = tree.query(xyz[:, :2], k=1)
    floor_pool = (
        floor
        & roi
        & (distances >= 0.20)
        & ~image_exclusion[pixels[:, 1], pixels[:, 0]]
    )
    floor_points = _sample_spread(
        pixels[floor_pool], maximum_floor_negative_points, anchor=center
    )
    if len(floor_points) < 2:
        return {
            "ready": False,
            "reason": "insufficient_ground_negative_points",
            "positive_points": len(positives),
            "floor_negative_points": len(floor_points),
        }

    # Tall points inside the candidate's horizontal neighborhood prevent SAM
    # from expanding a thin pallet into a wall, rack or stacked cargo behind it.
    horizontal = (
        (pixels[:, 0] >= max(0, x0 - 32))
        & (pixels[:, 0] <= min(width - 1, x1 + 32))
        & (pixels[:, 1] <= min(height - 1, y1 + 32))
    )
    tall_pool = (
        tall
        & roi
        & horizontal
        # Tall background immediately above a thin pallet is the most useful
        # anti-leak signal.  Exclude actual pallet pixels, but do not apply the
        # wider ground-ring dilation here or that signal disappears.
        & ~occupied[pixels[:, 1], pixels[:, 0]]
    )
    tall_points = _sample_spread(
        pixels[tall_pool], maximum_tall_negative_points, anchor=center
    )
    points = (
        [{"x": x, "y": y, "label": 1, "source": "low_3d_component"}
         for x, y in positives]
        + [{"x": x, "y": y, "label": 0, "source": "ground_ring"}
           for x, y in floor_points]
        + [{"x": x, "y": y, "label": 0, "source": "tall_background"}
           for x, y in tall_points]
    )
    return {
        "ready": True,
        "reason": None,
        "points": points,
        "positive_points": len(positives),
        "floor_negative_points": len(floor_points),
        "tall_negative_points": len(tall_points),
        "low_points_in_roi": int(len(low_indices)),
        "component_points": int(len(component_indices)),
        "component_xy_span_m": xy_span.tolist(),
        "winning_seed_points": winner_count,
        "runner_up_seed_points": runner_up_count,
        "component_pixel_bbox": [
            int(component_pixels[:, 0].min()),
            int(component_pixels[:, 1].min()),
            int(component_pixels[:, 0].max()),
            int(component_pixels[:, 1].max()),
        ],
        "roi_bbox": [
            max(0, x0 - padding),
            max(0, y0 - padding),
            min(width - 1, x1 + padding),
            min(height - 1, y1 + padding),
        ],
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
    }


def _panel(image: Image.Image, mask: np.ndarray, points: list[dict], title: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        values = [(int(x), int(y)) for [[x, y]] in contour]
        if len(values) >= 2:
            draw.line(values + [values[0]], fill=(0, 210, 255), width=4)
    for point in points:
        source = point.get("source")
        color = ((20, 230, 120) if point["label"] == 1 else
                 (255, 80, 80) if source == "ground_ring" else (255, 180, 0))
        x, y = int(point["x"]), int(point["y"])
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline="black", width=2)
    image.thumbnail((720, 520), Image.Resampling.LANCZOS)
    output = Image.new("RGB", (720, 570), "white")
    output.paste(image, ((720 - image.width) // 2, 40))
    ImageDraw.Draw(output).text((8, 8), title, fill="black", font=ImageFont.load_default())
    return output


def audit(dataset: Path, baseline_path: Path, geometry_path: Path,
          output: Path, *, limit: int = 4) -> dict:
    if output.exists():
        raise ValueError("output directory already exists")
    dataset = dataset.resolve()
    manifest_path = dataset / "import_manifest.json"
    inputs = [manifest_path, baseline_path.resolve(), geometry_path.resolve()]
    before = {str(path): _sha256(path) for path in inputs}
    state_path = Path(r"D:\Apps\LabelAgent\work\state.json")
    state_before = _sha256(state_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline = _mapping(json.loads(baseline_path.read_text(encoding="utf-8")))
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    targets = _ranked_targets(geometry, limit)
    camera_numbers = sorted({
        int(_camera_name(row["topic"]).split("_")[1]) for row in targets
    })
    reference_audit = ReferenceReadAudit(dataset)
    reference_audit.start()
    cameras, calibration_paths = sensor_cameras(dataset, manifest, camera_numbers)
    inventories = {
        camera: sorted((dataset / "source" / "cameras" / f"camera_{camera}").glob("*.png"))
        for camera in camera_numbers
    }
    output.mkdir(parents=True)
    rows, panels = [], []
    for rank, target in enumerate(targets, start=1):
        topic, frame = str(target["topic"]), int(target["frame"])
        camera = int(_camera_name(topic).split("_")[1])
        shapes = baseline[topic][str(frame)]
        matches = [shape for shape in shapes if str(shape.get("id")) == str(target["shape_id"])]
        if len(matches) != 1:
            raise ValueError(f"target identity is not unique: {topic} {frame}")
        current = _rasterize(matches[0], 1920, 1536).astype(np.uint8)
        union = np.zeros_like(current)
        for shape in shapes:
            if str(shape.get("label", "")).casefold() == "pallet":
                union |= _rasterize(shape, 1920, 1536).astype(np.uint8)
        xyz, lidar_path = read_xyz(dataset, manifest, frame)
        plane = fit_support_plane(xyz)
        if plane.get("usable"):
            indices, pixels = project_visible(
                xyz, cameras[camera], **{
                    key: POINT_RECIPE[key] for key in ("cell_size", "depth_slack", "near")
                }
            )
            visible_xyz = xyz[indices]
            heights = visible_xyz @ np.asarray(plane["plane"][:3]) + float(plane["plane"][3])
            legacy = select_lidar_prompt_points(current, union, pixels, heights)
            prompt = select_instance_lidar_prompts(current, union, pixels, heights, visible_xyz)
        else:
            legacy = {"ready": False, "reason": "plane_unavailable"}
            prompt = {"ready": False, "reason": "plane_unavailable"}
        image_path = inventories[camera][frame]
        row = {
            "rank": rank,
            "topic": topic,
            "frame": frame,
            "shape_id": target["shape_id"],
            "image_path": str(image_path),
            "image_sha256": _sha256(image_path),
            "lidar_path": lidar_path,
            "plane": plane,
            "legacy_prompt": legacy,
            "instance_prompt": prompt,
            "sam_called": False,
            "workbench_writes": 0,
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
        }
        panels.append(_panel(
            Image.open(image_path).convert("RGB"), current,
            prompt.get("points") or [],
            f"#{rank} camera_{camera} frame {frame} legacy={legacy['ready']} 3d={prompt['ready']}"
        ))
        rows.append(row)
        (output / f"camera_{camera}-frame-{frame}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    reference_audit.stop()
    sheet = Image.new("RGB", (1440, 1140), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 2) * 720, (index // 2) * 570))
    sheet_path = output / "instance-lidar-prompt-review.jpg"
    sheet.save(sheet_path, quality=90, optimize=True)
    if any(_sha256(Path(path)) != digest for path, digest in before.items()):
        raise RuntimeError("an input changed during LiDAR prompt audit")
    report = {
        "schema_version": "labelagent.pallet_lidar_instance_prompt_audit.v1",
        "status": "offline_candidate_prompt_only",
        "input_sha256": before,
        "calibration_paths": calibration_paths,
        "selected_count": len(rows),
        "legacy_ready_count": sum(row["legacy_prompt"]["ready"] for row in rows),
        "instance_ready_count": sum(row["instance_prompt"]["ready"] for row in rows),
        "sam_calls": 0,
        "reference_label_paths_opened": reference_audit.denied,
        "reference_labels_used": False,
        "workbench_writes": 0,
        "production_state_before_sha256": state_before,
        "production_state_after_sha256": _sha256(state_path),
        "contact_sheet": str(sheet_path.resolve()),
        "contact_sheet_sha256": _sha256(sheet_path),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "production_deployed": False,
        "rows": rows,
    }
    report_path = output / "results.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path.resolve())
    report["report_sha256"] = _sha256(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--geometry-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()
    report = audit(args.dataset, args.baseline, args.geometry_audit, args.output, limit=args.limit)
    print(json.dumps({key: report[key] for key in (
        "selected_count", "legacy_ready_count", "instance_ready_count", "sam_calls",
        "reference_labels_used", "workbench_writes", "production_state_before_sha256",
        "production_state_after_sha256", "contact_sheet", "contact_sheet_sha256",
        "report_path", "report_sha256",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
