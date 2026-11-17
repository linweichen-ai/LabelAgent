"""Evaluate label-free LiDAR ramp hints on a bounded multi-camera dev set.

Assistant-selected points are loaded only after all geometry hints have been
generated.  They are diagnostic candidates rather than human truth and never
participate in fitting, clustering, ranking, or prompt generation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from tools.slope_lidar_geometry import (
    camera_parameters,
    cluster_tilted_samples,
    component_camera_hints,
    fit_support_plane,
    project_visible,
    read_xyz_pcd,
    select_traversable_components,
    tilted_planar_samples,
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_camera(calibration: Path, camera: int) -> dict:
    extrinsics = yaml.safe_load((calibration / "lidar2camera.yaml").read_text(encoding="utf-8"))
    text = (calibration / f"camera{camera}_calib.yaml").read_text(encoding="utf-8")
    intrinsic = yaml.safe_load("\n".join(line for line in text.splitlines() if not line.startswith("%YAML:")))
    return camera_parameters(intrinsic, extrinsics[f"lidar_to_camera_{camera}"])


def validate_lidar_receipts(manifest_path: Path, lidar_root: Path, frames: list[str]) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {(str(row["sequence"]), str(row["frame_id"])): row for row in manifest.get("files", [])}
    receipts = []
    for frame in frames:
        row = rows.get(("008", frame))
        path = lidar_root / f"{frame}.pcd"
        if row is None or not path.is_file():
            raise ValueError(f"Missing bound LiDAR source for {frame}")
        actual = digest(path)
        if actual != row.get("sha256") or path.stat().st_size != row.get("size_bytes") or not row.get("etag_md5_match"):
            raise ValueError(f"LiDAR receipt mismatch for {frame}")
        receipts.append({
            "frame_id": frame,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
            "oss_uri": row.get("oss_uri"),
            "etag": row.get("etag"),
            "etag_md5_match": True,
        })
    return receipts


def serial_component(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in {"sample_indices", "points", "centroid"}} | {
        "centroid": np.asarray(row["centroid"]).round(6).tolist()
    }


def distance_to_hints(point: dict, hints: list[dict]) -> float | None:
    if not hints:
        return None
    pixels = [pixel for hint in hints for pixel in hint.get("points", [{"x": hint["x"], "y": hint["y"]}])]
    return float(min(np.hypot(pixel["x"] - point["x"], pixel["y"] - point["y"]) for pixel in pixels))


def render_overlay(image_path: Path, output_path: Path, hints: list[dict], review_points: list[dict]) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image {image_path}")
    for rank, hint in enumerate(hints, 1):
        x0, y0, x1, y1 = hint["pixel_bbox"]
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 165, 255), 2)
        cv2.circle(image, (hint["x"], hint["y"]), 6, (0, 0, 255), -1)
        cv2.putText(image, f"G{rank}", (hint["x"] + 7, hint["y"] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        for point in hint.get("points", [])[1:]:
            cv2.circle(image, (point["x"], point["y"]), 4, (255, 0, 255), -1)
    for point in review_points:
        color = (0, 255, 0) if point["label"] == 1 else (255, 0, 0)
        cv2.drawMarker(image, (point["x"], point["y"]), color, cv2.MARKER_CROSS, 18, 2)
    if not cv2.imwrite(str(output_path), image):
        raise ValueError(f"Cannot write overlay {output_path}")


def evaluate(args) -> dict:
    data_root = Path(args.data_root).resolve()
    lidar_root = Path(args.lidar_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Output directory already exists")
    output.mkdir(parents=True)
    visuals = output / "visuals"
    visuals.mkdir()

    review = json.loads(Path(args.assistant_points).read_text(encoding="utf-8-sig"))
    if review.get("human_truth") is not False or review.get("human_confirmed") is not False:
        raise ValueError("This evaluator expects explicitly non-human diagnostic points")
    by_sample = {row["sample_id"]: row["points"] for row in review["items"]}
    frames = sorted({sample.split("_")[1] for sample in by_sample})
    if len(frames) != 2:
        raise ValueError("The bounded dev gate must contain exactly two synchronized frames")
    receipts = validate_lidar_receipts(Path(args.lidar_manifest), lidar_root, frames)
    calibration = data_root / "calib_640x512"
    cameras = {camera: load_camera(calibration, camera) for camera in range(4)}

    rows = []
    frame_rows = []
    generated = {}
    for frame in frames:
        points = read_xyz_pcd(lidar_root / f"{frame}.pcd")
        support = fit_support_plane(points)
        if not support.get("usable"):
            raise ValueError(f"Support plane unavailable for {frame}: {support}")
        samples = tilted_planar_samples(points, support)
        all_components = cluster_tilted_samples(samples, support)
        components = select_traversable_components(all_components, args.maximum_median_tilt)
        frame_rows.append({
            "frame_id": frame,
            "raw_xyz_points": len(points),
            "support_plane": support,
            "tilted_voxel_samples": len(samples["selected_points"]),
            "all_components": [serial_component(row) for row in all_components],
            "selected_traversable_components": [serial_component(row) for row in components],
        })
        for camera in range(4):
            sample_id = f"008_{frame}_camera_{camera}"
            hints = component_camera_hints(
                points,
                components,
                cameras[camera],
                maximum_hints=args.maximum_hints,
                maximum_points_per_hint=args.maximum_points_per_hint,
            )
            generated[sample_id] = hints
            review_points = by_sample.get(sample_id, [])
            positive = next((point for point in review_points if point["label"] == 1), None)
            distance = distance_to_hints(positive, hints) if positive is not None else None
            image_path = data_root / "cameras_640x512" / f"camera_{camera}" / f"{frame}.png"
            render_overlay(image_path, visuals / f"{sample_id}.png", hints, review_points)
            rows.append({
                "sample_id": sample_id,
                "assistant_visible_slope_candidate": positive is not None,
                "generated_hint_count": len(hints),
                "nearest_hint_distance_pixels": distance,
                "covered_at_16px": distance is not None and distance <= 16,
                "covered_at_24px": distance is not None and distance <= 24,
                "covered_at_32px": distance is not None and distance <= 32,
                "hints": hints,
                "image_path": str(image_path),
                "image_sha256": digest(image_path),
            })

    positives = [row for row in rows if row["assistant_visible_slope_candidate"]]
    negatives = [row for row in rows if not row["assistant_visible_slope_candidate"]]
    summary = {
        "assistant_candidate_positives": len(positives),
        "assistant_candidate_negatives": len(negatives),
        "positive_views_with_any_geometry_hint": sum(row["generated_hint_count"] > 0 for row in positives),
        "negative_views_with_any_geometry_hint": sum(row["generated_hint_count"] > 0 for row in negatives),
        "positive_points_covered_at_16px": sum(row["covered_at_16px"] for row in positives),
        "positive_points_covered_at_24px": sum(row["covered_at_24px"] for row in positives),
        "positive_points_covered_at_32px": sum(row["covered_at_32px"] for row in positives),
        "total_generated_hints": sum(row["generated_hint_count"] for row in rows),
    }
    report = {
        "schema_version": "labelagent.slope-lidar-geometry.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_only_not_deployed",
        "method": "label-free dominant support plane plus locally tilted planar LiDAR components",
        "maximum_component_median_tilt_degrees": args.maximum_median_tilt,
        "maximum_positive_points_per_component_hint": args.maximum_points_per_hint,
        "prediction_reads_assistant_points": False,
        "assistant_points_used_after_generation_for_diagnostic_evaluation_only": True,
        "human_truth": False,
        "human_confirmed": False,
        "training_authorized": False,
        "production_deployed": False,
        "color_rule_used": False,
        "labels_read": False,
        "lidar_receipts": receipts,
        "calibration_files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in sorted(calibration.glob("*.yaml"))
        ],
        "frames": frame_rows,
        "images": rows,
        "summary": summary,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")), flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--lidar-root", required=True)
    parser.add_argument("--lidar-manifest", required=True)
    parser.add_argument("--assistant-points", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-hints", type=int, default=6)
    parser.add_argument("--maximum-median-tilt", type=float, default=10.0)
    parser.add_argument("--maximum-points-per-hint", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.maximum_hints <= 20:
        parser.error("--maximum-hints must be between 1 and 20")
    if not 5.0 <= args.maximum_median_tilt <= 20.0:
        parser.error("--maximum-median-tilt must be between 5 and 20")
    if not 1 <= args.maximum_points_per_hint <= 8:
        parser.error("--maximum-points-per-hint must be between 1 and 8")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
