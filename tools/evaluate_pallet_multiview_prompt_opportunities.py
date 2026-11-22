#!/usr/bin/env python3
"""Find label-conditioned cross-view pallet prompt opportunities offline.

The result is a review opportunity only.  It is not segmentation support, a
pass/fail decision, pixel truth, training data, a workbench write, or a
deployed behavior.
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

from tools.evaluate_pallet_box3d_roi_sam import load_camera
from tools.evaluate_projected_support import class_membership, project_visible
from tools.slope_lidar_geometry import read_xyz_pcd


OVERLAP_SCHEMA = "labelagent.pallet_box3d_overlap_pairs.v1"
REPORT_SCHEMA = "labelagent.pallet_multiview_prompt_opportunities.v1"
VISIBILITY_GRIDS = ((4, 0.10), (8, 0.15))
MINIMUM_EXCLUSIVE_POINTS = 3


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def hash_indices(indices: np.ndarray) -> str:
    """Hash a canonical sorted little-endian shared-XYZ index vector."""
    canonical = np.asarray(sorted(set(map(int, indices))), dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def validate_overlap_report(report: dict) -> None:
    if (report.get("schema_version") != OVERLAP_SCHEMA
            or report.get("status") != "offline_instance_diagnostic_not_deployed"
            or report.get("human_pixel_truth") is not False
            or report.get("approved_for_training") is not False
            or report.get("production_deployed") is not False
            or report.get("workbench_writes") != 0
            or not isinstance(report.get("input_sha256"), dict)
            or not report["input_sha256"]
            or not isinstance(report.get("rows"), list)):
        raise ValueError("requires a frozen non-deployed overlap split report")


def plan_targets(report: dict) -> list[dict]:
    """Select only explicitly unresolved targets and preserve their primary view."""
    validate_overlap_report(report)
    result, identities = [], set()
    for row in report["rows"]:
        sequence, frame_id = str(row["sequence"]), str(row["frame_id"])
        primary = int(row["camera"])
        if primary not in range(4):
            raise ValueError("primary camera must be in [0, 3]")
        metrics = row.get("depth_split_metrics")
        if not isinstance(metrics, list):
            raise ValueError("overlap row has no depth split metrics")
        for metric in metrics:
            if not isinstance(metric.get("handled_candidate"), bool):
                raise ValueError("handled_candidate must be an explicit boolean")
            if metric["handled_candidate"] is not False:
                continue
            if metric.get("obj_id") is None:
                raise ValueError("unresolved target obj_id is absent")
            target_id = str(metric["obj_id"])
            identity = (sequence, frame_id, primary, target_id)
            if identity in identities:
                raise ValueError(f"duplicate unresolved target: {identity}")
            identities.add(identity)
            result.append({
                "sequence": sequence,
                "frame_id": frame_id,
                "primary_camera": primary,
                "target_obj_id": target_id,
            })
    return result


def _safe_sequence_root(window_root: Path, sequence: str) -> Path:
    root = window_root.resolve()
    candidate = (root / sequence).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"sequence escapes frozen window root: {sequence}") from error
    return candidate


def load_center_frame(window_root: Path, sequence: str, frame_id: str
                      ) -> tuple[np.ndarray, list[dict], dict[str, str]]:
    sequence_root = _safe_sequence_root(window_root, sequence)
    labels_root, pcd_root = sequence_root / "labels", sequence_root / "pcd"
    label_path = (labels_root / f"{frame_id}.json").resolve()
    pcd_path = (pcd_root / f"{frame_id}.pcd").resolve()
    try:
        label_path.relative_to(labels_root.resolve())
        pcd_path.relative_to(pcd_root.resolve())
    except ValueError as error:
        raise ValueError("frame identity escapes frozen sequence") from error
    if not label_path.is_file() or not pcd_path.is_file():
        raise ValueError(f"center frame inputs are incomplete: {sequence}/{frame_id}")
    # Bind the exact bytes that will be parsed.  A second digest after parsing
    # closes the race where either file could otherwise change between use and
    # evidence capture.
    frozen = {
        str(label_path): digest(label_path),
        str(pcd_path): digest(pcd_path),
    }
    label = json.loads(label_path.read_text(encoding="utf-8"))
    if str(label.get("frame_id")) != frame_id:
        raise ValueError("label frame identity differs from requested center frame")
    objects = label.get("objects")
    if not isinstance(objects, list):
        raise ValueError("center label objects are absent")
    xyz = read_xyz_pcd(pcd_path)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("center PCD did not produce XYZ points")
    drifted = [path for path, expected in frozen.items()
               if not Path(path).is_file() or digest(Path(path)) != expected]
    if drifted:
        raise RuntimeError(
            f"center frame input changed while being parsed: {drifted}")
    return xyz, objects, frozen


def _target_owner(points: np.ndarray, objects: list[dict], target_obj_id: str
                  ) -> tuple[np.ndarray, int, int]:
    bits, owner, boxes = class_membership(points, objects)
    matches = [index for index, box in enumerate(boxes)
               if box["bit"] == 1 and str(box["obj_id"]) == target_obj_id]
    if len(matches) != 1:
        raise ValueError("target pallet identity is absent or ambiguous in center labels")
    owner_id = matches[0] + 1
    exclusive = (owner == owner_id) & (bits == 1)
    return exclusive, int(exclusive.sum()), int((owner == -1).sum())


def analyze_target(points: np.ndarray, objects: list[dict], cameras: dict[int, dict],
                   primary_camera: int, target_obj_id: str, *,
                   grids=VISIBILITY_GRIDS,
                   minimum_points: int = MINIMUM_EXCLUSIVE_POINTS) -> dict:
    """Evaluate robust visibility without treating it as segmentation support."""
    if set(cameras) != set(range(4)) or primary_camera not in range(4):
        raise ValueError("all four calibrated cameras and one primary camera are required")
    if (len(grids) < 2 or len(set(grids)) != len(grids)
            or minimum_points < 1
            or any(int(cell) < 1 or float(slack) < 0 for cell, slack in grids)):
        raise ValueError("invalid multigrid visibility parameters")
    exclusive, exclusive_count, ambiguous_count = _target_owner(
        points, objects, str(target_obj_id))
    camera_rows, eligible = [], []
    for camera_number in range(4):
        grid_rows, cohorts = [], []
        for cell_size, depth_slack in grids:
            source_indices, _ = project_visible(
                points, cameras[camera_number], int(cell_size), float(depth_slack))
            target_indices = np.asarray(
                sorted(set(map(int, source_indices[exclusive[source_indices]]))),
                dtype=np.int64)
            cohorts.append(target_indices)
            grid_rows.append({
                "cell_size_pixels": int(cell_size),
                "depth_slack_m": float(depth_slack),
                "exclusive_target_visible_raw_index_count": len(target_indices),
                "exclusive_target_visible_raw_index_sha256": hash_indices(target_indices),
            })
        shared = (np.asarray(sorted(set(cohorts[0]).intersection(
            *(set(values) for values in cohorts[1:]))), dtype=np.int64)
            if cohorts else np.empty(0, dtype=np.int64))
        minimum = min(row["exclusive_target_visible_raw_index_count"]
                      for row in grid_rows)
        robust = (camera_number != primary_camera
                  and all(row["exclusive_target_visible_raw_index_count"] >= minimum_points
                          for row in grid_rows)
                  and len(shared) >= minimum_points)
        camera_row = {
            "camera": camera_number,
            "is_primary_camera": camera_number == primary_camera,
            "grids": grid_rows,
            "minimum_grid_exclusive_target_visible_raw_index_count": minimum,
            "shared_across_grids_raw_index_count": len(shared),
            "shared_across_grids_raw_index_sha256": hash_indices(shared),
            "qualifies_as_nonprimary_prompt_opportunity": robust,
        }
        camera_rows.append(camera_row)
        if robust:
            eligible.append(camera_row)
    # Frozen ordering: maximum worst-grid count, then lowest camera number.
    eligible.sort(key=lambda row: (
        -row["minimum_grid_exclusive_target_visible_raw_index_count"],
        row["camera"]))
    best = eligible[0] if eligible else None
    return {
        "evidence_status": (
            "label_conditioned_cross_view_prompt_opportunity"
            if best else "insufficient"),
        "best_camera": best["camera"] if best else None,
        "best_camera_minimum_grid_point_count": (
            best["minimum_grid_exclusive_target_visible_raw_index_count"]
            if best else None),
        "exclusive_target_raw_index_count": exclusive_count,
        "ambiguous_3d_box_raw_index_count": ambiguous_count,
        "cameras": camera_rows,
    }


def _calibration_inputs(calibration: Path) -> dict[str, str]:
    paths = [calibration / "lidar2camera.yaml"] + [
        calibration / f"camera{camera}_calib.yaml" for camera in range(4)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"calibration inputs missing: {missing}")
    return {str(path.resolve()): digest(path) for path in paths}


def _verify_frozen(frozen: dict[str, str]) -> None:
    drifted = [path for path, expected in frozen.items()
               if not Path(path).is_file() or digest(Path(path)) != expected]
    if drifted:
        raise RuntimeError(f"frozen multiview opportunity input changed: {drifted}")


def evaluate(overlap_report: Path, window_root: Path, calibration: Path,
             output: Path) -> dict:
    if output.exists():
        raise ValueError("output already exists")
    overlap_bytes = overlap_report.read_bytes()
    overlap = json.loads(overlap_bytes.decode("utf-8"))
    jobs = plan_targets(overlap)
    frozen = {str(overlap_report.resolve()): hashlib.sha256(overlap_bytes).hexdigest()}
    frozen.update(_calibration_inputs(calibration))
    cameras = {camera: load_camera(calibration, camera) for camera in range(4)}

    frame_cache: dict[tuple[str, str], tuple[np.ndarray, list[dict]]] = {}
    rows = []
    for job in jobs:
        key = (job["sequence"], job["frame_id"])
        if key not in frame_cache:
            points, objects, hashes = load_center_frame(window_root, *key)
            frame_cache[key] = (points, objects)
            frozen.update(hashes)
        result = analyze_target(*frame_cache[key], cameras,
                                job["primary_camera"], job["target_obj_id"])
        rows.append({**job, **result,
                     "candidate_only": True,
                     "requires_review": True})

    _verify_frozen(frozen)
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "offline_label_conditioned_prompt_opportunities_not_deployed",
        "scope": "cross_view_prompt_opportunity_not_segmentation_support_or_pass",
        "overlap_report_sha256": hashlib.sha256(overlap_bytes).hexdigest(),
        "input_sha256": frozen,
        "visibility_grids": [
            {"cell_size_pixels": cell, "depth_slack_m": slack}
            for cell, slack in VISIBILITY_GRIDS],
        "shared_index_definition": (
            "zero-based position in the one finite-nonzero shared XYZ array "
            "read once from the center-frame PCD; not the original PCD record ordinal"),
        "opportunity_gate": (
            "nonprimary camera has at least 3 exclusive target indices in every "
            "visibility grid and at least 3 identical indices shared across grids"),
        "best_camera_selection": (
            "maximum minimum-grid point count, then lowest camera number"),
        "minimum_exclusive_target_points_per_grid": MINIMUM_EXCLUSIVE_POINTS,
        "unresolved_target_count": len(jobs),
        "opportunity_count": sum(
            row["evidence_status"] ==
            "label_conditioned_cross_view_prompt_opportunity" for row in rows),
        "insufficient_count": sum(row["evidence_status"] == "insufficient"
                                  for row in rows),
        "center_frame_pcd_read_count": len(frame_cache),
        "rows": rows,
        "candidate_only": True,
        "conditioned_on_human_3d_labels": True,
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
            raise RuntimeError("multiview opportunity report readback differs")
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
            "insufficient_count", "center_frame_pcd_read_count",
            "human_truth_writes", "training_data_writes", "workbench_writes",
            "oss_writes", "production_deployed", "readback_verified",
            "report_path", "report_sha256")
    print(json.dumps({key: report[key] for key in keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
