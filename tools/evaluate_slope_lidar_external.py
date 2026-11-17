"""Run the frozen, label-free cross-dataset LiDAR-to-SAM slope gate.

The input manifest must have been created before any image or model output was
inspected.  Every bound source is hashed again before geometry or SAM runs.
This evaluator never opens labels and produces candidates for later review,
not human truth or a production decision.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import yaml

from tools.evaluate_slope_lidar_sam import (
    geometry_attempt_rank,
    geometry_mask_metrics,
    polygon_mask,
    prompt_points,
    ring_constrained_points,
)
from tools.slope_lidar_geometry import (
    camera_parameters,
    cluster_tilted_samples,
    component_camera_hints,
    fit_support_plane,
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


def request_json(url: str, payload: dict | None = None, timeout: float = 90) -> dict:
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise ValueError("Service response is not an object")
    return result


def verify_record(record: dict) -> dict:
    if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
        raise ValueError("Frozen file record has an unexpected schema")
    path = Path(record["path"])
    if not path.is_file():
        raise ValueError(f"Frozen source is missing: {path}")
    size = path.stat().st_size
    actual = digest(path)
    if size != record["bytes"] or actual != record["sha256"]:
        raise ValueError(f"Frozen source changed after selection: {path}")
    return {"path": str(path), "bytes": size, "sha256": actual}


def verify_manifest(path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    boundaries = manifest.get("boundaries") or {}
    if (
        manifest.get("schema_version") != "labelagent.slope-geometry-external-freeze.v1"
        or manifest.get("status") != "frozen_before_visual_or_model_inspection"
        or boundaries.get("images_inspected_before_freeze") is not False
        or boundaries.get("model_outputs_inspected_before_freeze") is not False
        or boundaries.get("labels_read") is not False
        or boundaries.get("human_truth") is not False
        or boundaries.get("training_authorized") is not False
        or boundaries.get("production_deployed") is not False
    ):
        raise ValueError("Manifest does not preserve the unseen candidate boundary")
    frames = manifest.get("frames")
    selection = manifest.get("selection") or {}
    if not isinstance(frames, list) or len(frames) != selection.get("selected_frames"):
        raise ValueError("Frozen frame count does not match selection")
    frame_ids = [str(row.get("frame_id")) for row in frames]
    if frame_ids != selection.get("frame_ids") or len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Frozen frame order or identity changed")

    verified = [verify_record(record) for record in manifest.get("calibration", [])]
    if len(verified) != 5:
        raise ValueError("Expected exactly five frozen calibration files")
    for frame in frames:
        images = frame.get("images") or {}
        if set(images) != {f"camera_{camera}" for camera in range(4)}:
            raise ValueError("Each frozen frame must bind exactly four cameras")
        verified.append(verify_record(frame["lidar"]))
        verified.extend(verify_record(images[f"camera_{camera}"]) for camera in range(4))
    return manifest, verified


def load_cameras(manifest: dict) -> dict[int, dict]:
    paths = {Path(record["path"]).name: Path(record["path"]) for record in manifest["calibration"]}
    expected = {"lidar2camera.yaml", *(f"camera{camera}_calib.yaml" for camera in range(4))}
    if set(paths) != expected:
        raise ValueError("Frozen calibration filenames are incomplete")
    extrinsics = yaml.safe_load(paths["lidar2camera.yaml"].read_text(encoding="utf-8"))
    cameras = {}
    for camera in range(4):
        text = paths[f"camera{camera}_calib.yaml"].read_text(encoding="utf-8")
        intrinsic = yaml.safe_load("\n".join(line for line in text.splitlines() if not line.startswith("%YAML:")))
        cameras[camera] = camera_parameters(intrinsic, extrinsics[f"lidar_to_camera_{camera}"])
        if (cameras[camera]["width"], cameras[camera]["height"]) != (640, 512):
            raise ValueError("External gate is fixed to the 640x512 BEV camera stream")
    return cameras


def serial_component(row: dict) -> dict:
    values = {key: value for key, value in row.items() if key not in {"sample_indices", "points", "centroid"}}
    values["centroid"] = np.asarray(row["centroid"]).round(6).tolist()
    return values


def run_sam_attempt(
    sam_url: str,
    image_bytes: bytes,
    image_shape: tuple[int, int],
    hints: list[dict],
    policy: str,
    negative_margin: int,
    timeout: float,
) -> dict:
    height, width = image_shape
    points = (
        prompt_points(hints)
        if policy == "all_positive"
        else ring_constrained_points(hints, width, height, margin=negative_margin)
    )
    payload = {"image": base64.b64encode(image_bytes).decode("ascii"), "points": points}
    started = time.monotonic()
    result = request_json(sam_url.rstrip("/") + "/segment_points", payload, timeout=timeout)
    elapsed = time.monotonic() - started
    mask = polygon_mask(result, width, height)
    return {
        "policy": policy,
        "points": points,
        "result": result,
        "mask": mask,
        "warnings": result.get("warnings") or [],
        "elapsed_seconds": elapsed,
        "metrics": geometry_mask_metrics(mask, hints),
    }


def render_candidate(
    source: np.ndarray,
    output: Path,
    sample_id: str,
    hints: list[dict],
    attempt: dict | None,
) -> np.ndarray:
    image = source.copy()
    if attempt is not None:
        mask = attempt["mask"]
        color = np.zeros_like(image)
        color[mask != 0] = (0, 180, 0)
        image = cv2.addWeighted(image, 1.0, color, 0.42, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 255), 2)
        for point in attempt["points"]:
            point_color = (0, 0, 255) if point["label"] == 1 else (255, 0, 0)
            cv2.circle(image, (point["x"], point["y"]), 4, point_color, -1)
    for hint in hints:
        x0, y0, x1, y1 = hint["pixel_bbox"]
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 165, 255), 1)
    state = "NO GEOMETRY HINT" if not hints else attempt["policy"].upper()
    cv2.rectangle(image, (0, 0), (640, 33), (0, 0, 0), -1)
    cv2.putText(image, f"{sample_id} | {state}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    if not cv2.imwrite(str(output), image):
        raise ValueError(f"Cannot write overlay {output}")
    return image


def save_contact_sheet(images: list[np.ndarray], output: Path, columns: int = 4) -> None:
    if not images or len(images) % columns:
        raise ValueError("Contact sheet requires a nonempty complete grid")
    tiles = [cv2.resize(image, (320, 256), interpolation=cv2.INTER_AREA) for image in images]
    rows = [np.hstack(tiles[index:index + columns]) for index in range(0, len(tiles), columns)]
    sheet = np.vstack(rows)
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise ValueError("Cannot write contact sheet")


def evaluate(args) -> dict:
    manifest_path = Path(args.manifest).resolve()
    manifest, verified = verify_manifest(manifest_path)
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Output directory already exists")
    output.mkdir(parents=True)
    visuals = output / "visuals"
    visuals.mkdir()
    cameras = load_cameras(manifest)
    health_before = request_json(args.sam_url.rstrip("/") + "/health")
    if health_before.get("status") != "ok" or health_before.get("model_loaded") is not True:
        raise ValueError("Formal SAM service is not loaded")

    frame_rows = []
    image_rows = []
    contact_images = []
    for frame in manifest["frames"]:
        frame_id = str(frame["frame_id"])
        points = read_xyz_pcd(frame["lidar"]["path"])
        support = fit_support_plane(points)
        all_components = []
        selected = []
        tilted_count = 0
        if support.get("usable"):
            samples = tilted_planar_samples(points, support)
            tilted_count = len(samples["selected_points"])
            all_components = cluster_tilted_samples(samples, support)
            selected = select_traversable_components(all_components, args.maximum_median_tilt)
        frame_rows.append({
            "frame_id": frame_id,
            "raw_xyz_points": len(points),
            "support_plane": support,
            "tilted_voxel_samples": tilted_count,
            "all_components": [serial_component(row) for row in all_components],
            "selected_traversable_components": [serial_component(row) for row in selected],
        })
        for camera in range(4):
            sample_id = f"{manifest['sequence']}_{frame_id}_camera_{camera}"
            image_record = frame["images"][f"camera_{camera}"]
            image_path = Path(image_record["path"])
            image_bytes = image_path.read_bytes()
            decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None or decoded.shape[:2] != (512, 640):
                raise ValueError(f"Frozen image cannot be decoded as 640x512: {image_path}")
            hints = component_camera_hints(
                points,
                selected,
                cameras[camera],
                maximum_hints=args.maximum_hints,
                maximum_points_per_hint=args.maximum_points_per_hint,
            ) if support.get("usable") else []
            attempts = []
            failures = []
            for policy in (["component_center_ring_negatives", "all_positive"] if hints else []):
                try:
                    attempts.append(run_sam_attempt(
                        args.sam_url, image_bytes, decoded.shape[:2], hints, policy,
                        args.negative_margin, args.timeout,
                    ))
                except (ValueError, urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
                    failures.append({
                        "policy": policy,
                        "type": type(error).__name__,
                        "detail": str(error)[:1000],
                    })
            selected_attempt = max(attempts, key=geometry_attempt_rank) if attempts else None
            overlay = visuals / f"{sample_id}.jpg"
            contact_images.append(render_candidate(decoded, overlay, sample_id, hints, selected_attempt))
            row = {
                "sample_id": sample_id,
                "frame_id": frame_id,
                "camera": camera,
                "image_path": str(image_path),
                "image_sha256": image_record["sha256"],
                "support_plane_usable": bool(support.get("usable")),
                "geometry_hint_count": len(hints),
                "hints": hints,
                "sam_attempts": [
                    {
                        "policy": attempt["policy"],
                        "points": attempt["points"],
                        "warnings": attempt["warnings"],
                        "elapsed_seconds": round(attempt["elapsed_seconds"], 4),
                        "metrics": attempt["metrics"],
                    }
                    for attempt in attempts
                ],
                "sam_failures": failures,
                "selected_policy": selected_attempt["policy"] if selected_attempt else None,
                "selected_metrics": selected_attempt["metrics"] if selected_attempt else None,
                "selected_warnings": selected_attempt["warnings"] if selected_attempt else [],
                "selected_mask_pixels": int(selected_attempt["mask"].sum()) if selected_attempt else 0,
                "selected_mask_fraction": float(selected_attempt["mask"].mean()) if selected_attempt else 0.0,
                "overlay": str(overlay),
            }
            image_rows.append(row)
            print(json.dumps({
                "sample_id": sample_id,
                "hints": len(hints),
                "selected_policy": row["selected_policy"],
                "geometry_gate": (row["selected_metrics"] or {}).get("geometry_gate_passed"),
                "failures": len(failures),
            }, separators=(",", ":")), flush=True)

    contact_sheet = output / "contact-sheet.jpg"
    save_contact_sheet(contact_images, contact_sheet)
    health_after = request_json(args.sam_url.rstrip("/") + "/health")
    selected_rows = [row for row in image_rows if row["selected_policy"] is not None]
    report = {
        "schema_version": "labelagent.slope-lidar-external.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_only_post_prediction_visual_review_required",
        "dataset": manifest["dataset"],
        "sequence": manifest["sequence"],
        "frozen_manifest_sha256": digest(manifest_path),
        "frozen_sources_verified": len(verified),
        "labels_read": False,
        "assistant_points_read": False,
        "human_truth": False,
        "human_confirmed": False,
        "training_authorized": False,
        "production_deployed": False,
        "color_rule_used": False,
        "selection_reads_visual_or_semantic_review": False,
        "fixed_parameters": {
            "maximum_component_median_tilt_degrees": args.maximum_median_tilt,
            "maximum_hints_per_view": args.maximum_hints,
            "maximum_points_per_hint": args.maximum_points_per_hint,
            "negative_ring_margin_pixels": args.negative_margin,
            "sam_policies": ["component_center_ring_negatives", "all_positive"],
        },
        "health_before": health_before,
        "health_after": health_after,
        "summary": {
            "frozen_frames": len(frame_rows),
            "frozen_views": len(image_rows),
            "support_planes_usable": sum(row["support_plane"].get("usable") is True for row in frame_rows),
            "views_with_geometry_hints": sum(row["geometry_hint_count"] > 0 for row in image_rows),
            "views_sent_to_sam": len(selected_rows),
            "selected_masks_geometry_gate_passed": sum(row["selected_metrics"]["geometry_gate_passed"] for row in selected_rows),
            "selected_masks_with_warnings": sum(bool(row["selected_warnings"]) for row in selected_rows),
            "sam_attempt_failures": sum(len(row["sam_failures"]) for row in image_rows),
            "sam_attempts": sum(len(row["sam_attempts"]) for row in image_rows),
            "sam_elapsed_seconds": round(sum(
                attempt["elapsed_seconds"] for row in image_rows for attempt in row["sam_attempts"]
            ), 4),
        },
        "frames": frame_rows,
        "images": image_rows,
        "contact_sheet": str(contact_sheet),
        "evidence_note": "All outputs are label-free candidates. Visual review happens after prediction and remains separate from human truth.",
    }
    report_path = output / "results.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], separators=(",", ":")), flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sam-url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--maximum-hints", type=int, default=6)
    parser.add_argument("--maximum-median-tilt", type=float, default=10.0)
    parser.add_argument("--maximum-points-per-hint", type=int, default=5)
    parser.add_argument("--negative-margin", type=int, default=80)
    args = parser.parse_args()
    if not args.sam_url.startswith("http://127.0.0.1:"):
        parser.error("SAM URL must be loopback")
    if not 1 <= args.maximum_hints <= 20:
        parser.error("--maximum-hints must be between 1 and 20")
    if not 5.0 <= args.maximum_median_tilt <= 20.0:
        parser.error("--maximum-median-tilt must be between 5 and 20")
    if not 1 <= args.maximum_points_per_hint <= 8:
        parser.error("--maximum-points-per-hint must be between 1 and 8")
    if not 8 <= args.negative_margin <= 128:
        parser.error("--negative-margin must be between 8 and 128")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
