#!/usr/bin/env python3
"""Find offline temporal local-coordinate pallet point-prompt opportunities.

This label-conditioned geometry diagnostic only counts possible prompt points.
It does not read images, call SAM, write truth/training/workbench/OSS data, or
deploy a behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.compare_boxes3d import Box
from tools.evaluate_pallet_box3d_roi_sam import load_camera
from tools.evaluate_pallet_multiview_prompt_opportunities import (
    MINIMUM_EXCLUSIVE_POINTS, VISIBILITY_GRIDS, digest, hash_indices,
    plan_targets,
)
from tools.evaluate_projected_support import class_membership, project_visible
from tools.slope_lidar_geometry import read_xyz_pcd


REPORT_SCHEMA = "labelagent.pallet_temporal_local_point_opportunities.v1"
EXPECTED_WINDOW_FRAMES = 11
VOXEL_SIZE_METERS = 0.02
MINIMUM_NONCENTER_CONTRIBUTING_FRAMES = 2


def _frame_sort_key(frame_id: str) -> tuple[int, int | str]:
    return (0, int(frame_id)) if frame_id.isdecimal() else (1, frame_id)


def _safe_sequence_root(window_root: Path, sequence: str) -> Path:
    root = window_root.resolve()
    candidate = (root / sequence).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"sequence escapes frozen window root: {sequence}") from error
    return candidate


def _read_json_bound(path: Path) -> tuple[dict, str]:
    before = digest(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if digest(path) != before:
        raise RuntimeError(f"label changed while being parsed: {path}")
    return payload, before


def _read_pcd_bound(path: Path) -> tuple[np.ndarray, str]:
    before = digest(path)
    points = read_xyz_pcd(path)
    if digest(path) != before:
        raise RuntimeError(f"PCD changed while being parsed: {path}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("PCD did not produce XYZ points")
    return points, before


def load_sequence_window(window_root: Path, sequence: str
                         ) -> tuple[list[dict], dict[str, str], dict[str, int]]:
    """Read each of the exact 11 label/PCD pairs once with drift guards."""
    sequence_root = _safe_sequence_root(window_root, sequence)
    labels_root, pcd_root = sequence_root / "labels", sequence_root / "pcd"
    if not labels_root.is_dir() or not pcd_root.is_dir():
        raise ValueError(f"frozen sequence window is incomplete: {sequence}")
    labels = {path.stem: path.resolve() for path in labels_root.glob("*.json")
              if path.is_file()}
    pcds = {path.stem: path.resolve() for path in pcd_root.glob("*.pcd")
            if path.is_file()}
    if (set(labels) != set(pcds)
            or len(labels) != EXPECTED_WINDOW_FRAMES):
        raise ValueError(
            f"requires exactly {EXPECTED_WINDOW_FRAMES} matching label/PCD frames: {sequence}")
    frames, frozen, reads = [], {}, {}
    for frame_id in sorted(labels, key=_frame_sort_key):
        label, label_sha = _read_json_bound(labels[frame_id])
        if str(label.get("frame_id")) != frame_id:
            raise ValueError("label frame identity differs from filename")
        objects = label.get("objects")
        if not isinstance(objects, list):
            raise ValueError("label objects are absent")
        points, pcd_sha = _read_pcd_bound(pcds[frame_id])
        frozen[str(labels[frame_id])] = label_sha
        frozen[str(pcds[frame_id])] = pcd_sha
        reads[frame_id] = reads.get(frame_id, 0) + 1
        frames.append({"frame_id": frame_id, "objects": objects, "points": points})
    return frames, frozen, reads


def select_target_object(objects: list[dict], target_obj_id: str) -> dict:
    matches = [obj for obj in objects
               if obj.get("obj_type") == "Pallet"
               and str(obj.get("obj_id")) == str(target_obj_id)]
    if len(matches) != 1:
        raise ValueError("target pallet is missing or duplicated in temporal frame")
    if matches[0].get("ignore"):
        raise ValueError("ignored target pallet cannot provide prompt evidence")
    # Box performs finite, positive-scale and rotation validation.
    Box(matches[0])
    return matches[0]


def remap_exclusive_target_points(points: np.ndarray, objects: list[dict],
                                  target_obj_id: str,
                                  center_target: dict) -> tuple[np.ndarray, dict]:
    """Map exclusive target points through normalized box-local coordinates.

    For row-vector coordinates, source local coordinate is
      l_f = (p_f - c_f) R_f
    normalized coordinate is u = l_f / s_f, and the center-frame point is
      p_c = (u * s_c) R_c^T + c_c.
    """
    source_target = select_target_object(objects, target_obj_id)
    source_box, center_box = Box(source_target), Box(center_target)
    bits, owner, boxes = class_membership(points, objects)
    matches = [index for index, item in enumerate(boxes)
               if item["bit"] == 1
               and str(item["obj_id"]) == str(target_obj_id)]
    if len(matches) != 1:
        raise ValueError("target owner is missing or duplicated after membership build")
    exclusive = (owner == matches[0] + 1) & (bits == 1)
    local = (points[exclusive] - source_box.center) @ source_box.rotation
    normalized = local / source_box.size
    if (not np.isfinite(normalized).all()
            or np.any(np.abs(normalized) > 0.50000002)):
        raise RuntimeError("exclusive point left normalized source box")
    mapped = ((normalized * center_box.size) @ center_box.rotation.T
              + center_box.center)
    return mapped, {
        "source_exclusive_point_count": int(exclusive.sum()),
        "source_ambiguous_point_count": int((owner == -1).sum()),
        "source_target_attr_canonical": json.dumps(
            source_target.get("obj_attr"), ensure_ascii=False,
            sort_keys=True, separators=(",", ":")),
        "source_target_scale": source_box.size.tolist(),
    }


def _unique_pixel_count(pixels: np.ndarray) -> int:
    if not len(pixels):
        return 0
    return len({(int(x), int(y)) for x, y in pixels})


def voxelize_mapped_points(parts: list[np.ndarray], frame_indices: list[int],
                           voxel_size: float = VOXEL_SIZE_METERS
                           ) -> tuple[np.ndarray, list[tuple[int, ...]], np.ndarray]:
    """Deduplicate mapped points in frozen center-global 2 cm voxels.

    The representative is the arithmetic centroid of all mapped samples in a
    voxel.  Voxels are emitted in lexicographic integer-key order, making the
    resulting index axis independent of frame/input traversal order.
    """
    if voxel_size <= 0 or len(parts) != len(frame_indices):
        raise ValueError("invalid temporal voxelization inputs")
    buckets: dict[tuple[int, int, int], dict[str, object]] = {}
    for points, frame_index in zip(parts, frame_indices):
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError("mapped points must be finite XYZ")
        keys = np.floor(values / voxel_size).astype(np.int64)
        for point, key in zip(values, keys):
            identity = tuple(map(int, key))
            bucket = buckets.setdefault(identity, {"points": [], "frames": set()})
            bucket["points"].append(point)
            bucket["frames"].add(int(frame_index))
    ordered = sorted(buckets)
    unique = (np.asarray([
        np.mean(np.asarray(buckets[key]["points"]), axis=0) for key in ordered],
        dtype=float).reshape((-1, 3)) if ordered else np.empty((0, 3), dtype=float))
    origins = [tuple(sorted(buckets[key]["frames"])) for key in ordered]
    keys = np.asarray(ordered, dtype="<i8").reshape((-1, 3))
    return unique, origins, keys


def analyze_target(frames: list[dict], center_frame_id: str, camera: dict,
                   target_obj_id: str, *, grids=VISIBILITY_GRIDS,
                   minimum_points: int = MINIMUM_EXCLUSIVE_POINTS) -> dict:
    if (len(frames) != EXPECTED_WINDOW_FRAMES
            or len({row["frame_id"] for row in frames}) != len(frames)):
        raise ValueError("temporal target analysis requires 11 distinct frames")
    center_rows = [row for row in frames if row["frame_id"] == center_frame_id]
    if len(center_rows) != 1:
        raise ValueError("center frame is absent or duplicated in temporal window")
    if (len(grids) < 2 or len(set(grids)) != len(grids)
            or minimum_points < 1):
        raise ValueError("invalid temporal multigrid parameters")
    center_target = select_target_object(center_rows[0]["objects"], target_obj_id)
    center_scene_points = center_rows[0]["points"]
    mapped_parts, frame_rows = [], []
    for frame_index, row in enumerate(frames):
        mapped, metrics = remap_exclusive_target_points(
            row["points"], row["objects"], target_obj_id, center_target)
        mapped_parts.append(mapped)
        frame_rows.append({
            "frame_id": row["frame_id"],
            "is_center_frame": row["frame_id"] == center_frame_id,
            **metrics,
        })
    raw_aggregate = (np.concatenate(mapped_parts, axis=0)
                     if mapped_parts else np.empty((0, 3), dtype=float))
    attrs = [row["source_target_attr_canonical"] for row in frame_rows]
    scales = np.asarray([row["source_target_scale"] for row in frame_rows],
                        dtype=float)
    attr_consistent = len(set(attrs)) == 1
    scale_max_range = float(np.ptp(scales, axis=0).max())
    aggregate, origin_sets, voxel_keys = voxelize_mapped_points(
        mapped_parts, list(range(len(mapped_parts))))
    center_index = next(index for index, row in enumerate(frames)
                        if row["frame_id"] == center_frame_id)
    grid_rows, cohorts, pixel_maps = [], [], []
    for cell_size, depth_slack in grids:
        raw_offset = len(center_scene_points)
        raw_source, raw_pixels = project_visible(
            np.concatenate((center_scene_points, raw_aggregate), axis=0),
            camera, int(cell_size), float(depth_slack))
        raw_keep = np.asarray(raw_source) >= raw_offset
        raw_source = np.asarray(raw_source, dtype=np.int64)[raw_keep] - raw_offset
        raw_pixels = np.asarray(raw_pixels)[raw_keep]
        offset = len(center_scene_points)
        source, pixels = project_visible(
            np.concatenate((center_scene_points, aggregate), axis=0),
            camera, int(cell_size), float(depth_slack))
        keep = np.asarray(source) >= offset
        source = np.asarray(source, dtype=np.int64)[keep] - offset
        pixels = np.asarray(pixels)[keep]
        order = np.argsort(source)
        source, pixels = source[order], np.asarray(pixels)[order]
        cohorts.append(source)
        pixel_maps.append({int(index): tuple(map(int, pixel))
                           for index, pixel in zip(source, pixels)})
        temporal = np.asarray([
            center_index not in origin_sets[int(index)]
            and any(origin != center_index for origin in origin_sets[int(index)])
            for index in source], dtype=bool)
        grid_rows.append({
            "cell_size_pixels": int(cell_size),
            "depth_slack_m": float(depth_slack),
            "visible_raw_sample_index_count": len(raw_source),
            "visible_raw_sample_unique_pixel_count": _unique_pixel_count(raw_pixels),
            "visible_unique_voxel_index_count": len(source),
            "visible_unique_voxel_index_sha256": hash_indices(source),
            "visible_unique_pixel_count": _unique_pixel_count(pixels),
            "visible_temporal_new_voxel_count": int(temporal.sum()),
            "visible_temporal_new_unique_pixel_count": _unique_pixel_count(
                pixels[temporal]),
        })
    shared = np.asarray(sorted(set(cohorts[0]).intersection(
        *(set(values) for values in cohorts[1:]))), dtype=np.int64)
    canonical_pixels = np.asarray([pixel_maps[0][int(index)] for index in shared],
                                  dtype=np.int64).reshape((-1, 2))
    temporal_mask = np.asarray([
        center_index not in origin_sets[int(index)]
        and any(origin != center_index for origin in origin_sets[int(index)])
        for index in shared], dtype=bool)
    temporal_shared = shared[temporal_mask]
    temporal_pixels = canonical_pixels[temporal_mask]
    contributing_indices = sorted({origin
        for index in temporal_shared
        for origin in origin_sets[int(index)] if origin != center_index})
    contributing_frames = [frames[index]["frame_id"]
                           for index in contributing_indices]
    center_shared_count = sum(
        center_index in origin_sets[int(index)] for index in shared)
    shared_unique_pixels = _unique_pixel_count(canonical_pixels)
    temporal_unique_pixels = _unique_pixel_count(temporal_pixels)
    gate = bool(
        all(row["visible_unique_voxel_index_count"] >= minimum_points
            and row["visible_unique_pixel_count"] >= minimum_points
            for row in grid_rows)
        and len(shared) >= minimum_points
        and shared_unique_pixels >= minimum_points
        and len(temporal_shared) >= minimum_points
        and temporal_unique_pixels >= minimum_points
        and len(contributing_frames) >= MINIMUM_NONCENTER_CONTRIBUTING_FRAMES
        and attr_consistent
        and scale_max_range <= 1e-6)
    return {
        "evidence_status": (
            "label_conditioned_temporal_local_point_prompt_opportunity"
            if gate else "insufficient"),
        "frame_count": len(frames),
        "aggregate_raw_exclusive_point_count": len(raw_aggregate),
        "aggregate_unique_voxel_count": len(aggregate),
        "aggregate_unique_voxel_key_sha256": hashlib.sha256(
            voxel_keys.tobytes()).hexdigest(),
        "voxel_size_meters": VOXEL_SIZE_METERS,
        "shared_across_grids_unique_voxel_index_count": len(shared),
        "shared_across_grids_unique_voxel_index_sha256": hash_indices(shared),
        "shared_across_grids_unique_pixel_count": shared_unique_pixels,
        "center_shared_index_count": center_shared_count,
        "noncenter_shared_index_count": len(temporal_shared),
        "noncenter_shared_unique_pixel_count": temporal_unique_pixels,
        "noncenter_contributing_frame_count": len(contributing_frames),
        "noncenter_contributing_frame_ids": contributing_frames,
        "target_attr_consistent": attr_consistent,
        "target_scale_max_range": scale_max_range,
        "grids": grid_rows,
        "frames": frame_rows,
    }


def _calibration_inputs(calibration: Path, cameras: set[int]) -> dict[str, str]:
    paths = [calibration / "lidar2camera.yaml"] + [
        calibration / f"camera{camera}_calib.yaml" for camera in sorted(cameras)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"calibration inputs missing: {missing}")
    return {str(path.resolve()): digest(path) for path in paths}


def _verify_frozen(frozen: dict[str, str]) -> None:
    drifted = [path for path, expected in frozen.items()
               if not Path(path).is_file() or digest(Path(path)) != expected]
    if drifted:
        raise RuntimeError(f"frozen temporal-local input changed: {drifted}")


def evaluate(overlap_report: Path, window_root: Path, calibration: Path,
             output: Path) -> dict:
    if output.exists():
        raise ValueError("output already exists")
    overlap_bytes = overlap_report.read_bytes()
    overlap = json.loads(overlap_bytes.decode("utf-8"))
    jobs = plan_targets(overlap)
    cameras_needed = {int(job["primary_camera"]) for job in jobs}
    frozen = {str(overlap_report.resolve()): hashlib.sha256(overlap_bytes).hexdigest()}
    calibration_hashes = _calibration_inputs(calibration, cameras_needed)
    frozen.update(calibration_hashes)
    cameras = {number: load_camera(calibration, number)
               for number in sorted(cameras_needed)}
    _verify_frozen(calibration_hashes)
    windows, read_audit = {}, {}
    for sequence in sorted({job["sequence"] for job in jobs}):
        frames, hashes, reads = load_sequence_window(window_root, sequence)
        windows[sequence] = frames
        read_audit[sequence] = reads
        frozen.update(hashes)
    rows = []
    for job in jobs:
        result = analyze_target(
            windows[job["sequence"]], job["frame_id"],
            cameras[job["primary_camera"]], job["target_obj_id"])
        rows.append({**job, **result, "candidate_only": True,
                     "requires_review": True})
    _verify_frozen(frozen)
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "offline_temporal_local_point_prompt_opportunities_not_deployed",
        "scope": "prompt_opportunity_not_segmentation_support_or_pass",
        "overlap_report_sha256": hashlib.sha256(overlap_bytes).hexdigest(),
        "input_sha256": frozen,
        "mathematical_definition": (
            "u=((p_f-c_f)@R_f)/s_f; p_center=(u*s_center)@R_center.T+c_center; "
            "row vectors, per-axis scale normalization"),
        "voxel_definition": (
            "floor(center_global_xyz/0.02m), lexicographic voxel-key order, "
            "arithmetic-centroid representative; gate uses unique voxels only"),
        "visibility_occlusion_reference": (
            "each temporal aggregate is projected together with the complete "
            "center-frame scene PCD; a nearer center-scene return can suppress it"),
        "visibility_grids": [
            {"cell_size_pixels": cell, "depth_slack_m": slack}
            for cell, slack in VISIBILITY_GRIDS],
        "opportunity_gate": (
            "all grids have >=3 unique voxels and unique pixels; >=3 identical "
            "voxel indices and unique pixels are shared across grids; >=3 shared "
            "temporal-new voxels and unique pixels originate from >=2 noncenter "
            "frames; target attribute and 3D-box scale remain consistent"),
        "tool_sha256": digest(Path(__file__).resolve()),
        "window_frame_count": EXPECTED_WINDOW_FRAMES,
        "unresolved_target_count": len(rows),
        "opportunity_count": sum(
            row["evidence_status"] ==
            "label_conditioned_temporal_local_point_prompt_opportunity"
            for row in rows),
        "insufficient_count": sum(row["evidence_status"] == "insufficient"
                                  for row in rows),
        "pcd_read_counts": read_audit,
        "total_pcd_reads": sum(sum(values.values())
                               for values in read_audit.values()),
        "rows": rows,
        "candidate_only": True,
        "conditioned_on_human_3d_labels": True,
        "images_read": 0,
        "sam_calls": 0,
        "gpu_requests": 0,
        "not_segmentation_support": True,
        "not_a_pass_decision": True,
        "human_pixel_truth": False,
        "approved_for_training": False,
        "human_truth_writes": 0,
        "training_data_writes": 0,
        "workbench_writes": 0,
        "oss_writes": 0,
        "production_deployed": False,
        "readback_verified": True,
    }
    output.mkdir(parents=True)
    try:
        report_path = output / "results.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        raw = report_path.read_bytes()
        if json.loads(raw.decode("utf-8")) != report:
            raise RuntimeError("temporal local point report readback differs")
        _verify_frozen(frozen)
        report["report_path"] = str(report_path.resolve())
        report["report_sha256"] = hashlib.sha256(raw).hexdigest()
        return report
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap-report", required=True, type=Path)
    parser.add_argument("--window-root", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    report = evaluate(arguments.overlap_report, arguments.window_root,
                      arguments.calibration, arguments.output)
    keys = ("unresolved_target_count", "opportunity_count",
            "insufficient_count", "total_pcd_reads", "images_read",
            "sam_calls", "gpu_requests", "human_truth_writes",
            "training_data_writes", "workbench_writes", "oss_writes",
            "production_deployed", "readback_verified", "report_path",
            "report_sha256")
    print(json.dumps({key: report[key] for key in keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
