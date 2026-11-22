#!/usr/bin/env python3
"""Audit full point evidence on frozen Round 700 final split masks.

The Round 700 decision remains immutable.  This tool recomputes the complete
temporal positive and center-conflict negative pixel sets from frozen inputs,
then asks whether each already-produced final split mask satisfies those point
gates.  It never invokes SAM and never writes truth, training data, or the
workbench.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.evaluate_pallet_temporal_local_point_opportunities as temporal_local
import tools.evaluate_pallet_temporal_local_point_sam as temporal_sam
from tools.evaluate_pallet_box3d_overlap_pairs import projected_conflicts
from tools.evaluate_pallet_box3d_roi_sam import digest, load_camera, localization_gate, project_box
from tools.evaluate_pallet_lidar_conflict_points import point_fraction
from tools.evaluate_projected_support import project_visible


REPORT_SCHEMA = "labelagent.pallet_temporal_split_point_evidence_audit.v1"
SOURCE_SCHEMA = "labelagent.pallet_temporal_local_point_sam.v1"
SUPPORT_SCHEMAS = {
    "opportunity": "labelagent.pallet_temporal_local_point_opportunities.v1",
    "overlap": "labelagent.pallet_box3d_overlap_pairs.v1",
    "baseline": "labelagent.pallet_lidar_conflict_points.v1",
}


def bound_json(path: Path) -> tuple[dict, str]:
    before = digest(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if digest(path) != before:
        raise RuntimeError(f"JSON input changed while being parsed: {path}")
    return payload, before


def bind_file(path: Path, inputs: dict[str, str], expected: str | None = None) -> str:
    if not path.is_file():
        raise ValueError(f"frozen input is missing: {path}")
    value = digest(path)
    if expected is not None and value != expected:
        raise ValueError(f"frozen input hash differs: {path}")
    key = str(path.resolve())
    prior = inputs.setdefault(key, value)
    if prior != value:
        raise ValueError(f"conflicting frozen input identity: {path}")
    return value


def verify_frozen(inputs: dict[str, str]) -> None:
    changed = [path for path, expected in inputs.items()
               if not Path(path).is_file() or digest(Path(path)) != expected]
    if changed:
        raise RuntimeError(f"split point audit input changed: {changed}")


def read_mask(path: Path, expected: str, inputs: dict[str, str]) -> np.ndarray:
    bind_file(path, inputs, expected)
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if digest(path) != expected:
        raise RuntimeError(f"final split mask changed while being decoded: {path}")
    if value is None:
        raise ValueError(f"final split mask is unreadable: {path}")
    return value > 0


def validate_source(report: dict) -> None:
    if (report.get("schema_version") != SOURCE_SCHEMA
            or report.get("status") !=
            "offline_temporal_local_point_sam_candidate_not_deployed"
            or report.get("candidate_only") is not True
            or report.get("human_pixel_truth") is not False
            or report.get("approved_for_training") is not False
            or report.get("production_deployed") is not False
            or report.get("human_truth_writes") != 0
            or report.get("training_data_writes") != 0
            or report.get("workbench_writes") != 0
            or report.get("oss_writes") != 0
            or report.get("evaluated_opportunity_count") != 7
            or not isinstance(report.get("input_sha256"), dict)):
        raise ValueError("requires a frozen candidate-only Round 700 report")


def index_rows(rows: list[dict]) -> dict[tuple[str, str, int, str], dict]:
    result = {}
    for row in rows:
        key = (str(row["sequence"]), str(row["frame_id"]), int(row["camera"]),
               str(row["target_obj_id"]))
        if key in result:
            raise ValueError(f"duplicate Round 700 row identity: {key}")
        result[key] = row
    if len(result) != 7:
        raise ValueError("Round 700 row count differs from frozen seven")
    return result


def discover_support_reports(source_inputs: dict[str, str],
                             inputs: dict[str, str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for raw_path, expected in sorted(source_inputs.items()):
        path = Path(raw_path)
        if path.name != "results.json" or not path.is_file():
            continue
        bind_file(path, inputs, expected)
        payload, actual = bound_json(path)
        if actual != expected:
            raise ValueError(f"support report hash differs: {path}")
        for name, schema in SUPPORT_SCHEMAS.items():
            if payload.get("schema_version") != schema:
                continue
            if name in found:
                raise ValueError(f"duplicate bound {name} support report")
            found[name] = {**payload, "_audit_bound_sha256": actual,
                           "_audit_bound_path": str(path.resolve())}
    if set(found) != set(SUPPORT_SCHEMAS):
        raise ValueError("Round 700 input map lacks one exact support report")
    return found


def validate_report_batch(source_rows: dict, support: dict,
                          source_opportunity_tool_sha: str | None = None) -> None:
    if (support["opportunity"].get("overlap_report_sha256")
            != support["overlap"].get("_audit_bound_sha256")):
        raise ValueError("opportunity and overlap support reports are cross-batch")
    if (source_opportunity_tool_sha is not None
            and support["opportunity"].get("tool_sha256")
            != source_opportunity_tool_sha):
        raise ValueError("opportunity support tool receipt differs from Round 700")
    # The source report records the exact historical tool hash, while this
    # checkout intentionally contains later visibility hardening.  Validate the
    # complete report structure with the current fail-closed validator only
    # after preserving and checking that historical receipt above.
    opportunity_for_validation = dict(support["opportunity"])
    opportunity_for_validation["tool_sha256"] = digest(
        Path(temporal_local.__file__))
    selected = temporal_sam.validate_reports(
        opportunity_for_validation, support["overlap"], support["baseline"])
    selected_ids = {(row["sequence"], row["frame_id"], row["camera"],
                     row["target_obj_id"]) for row in selected}
    if selected_ids != set(source_rows):
        raise ValueError("Round 700 rows differ from their bound support batch")


def recompute_round700_temporal_positive_pixels(
        frames: list[dict], center_frame: str, camera: dict, target_id: str
        ) -> tuple[np.ndarray, dict]:
    """Reproduce the frozen pre-scene-occlusion Round 700 point protocol.

    This is intentionally local to the historical audit.  New candidate runs
    must use the corrected scene-aware implementation in ``temporal_sam``.
    """
    center_indices = [index for index, row in enumerate(frames)
                      if row["frame_id"] == center_frame]
    if len(center_indices) != 1:
        raise ValueError("Round 700 center frame is missing or duplicated")
    center_index = center_indices[0]
    center_target = temporal_local.select_target_object(
        frames[center_index]["objects"], target_id)
    parts, metrics = [], []
    for row in frames:
        mapped, row_metrics = temporal_local.remap_exclusive_target_points(
            row["points"], row["objects"], target_id, center_target)
        parts.append(mapped)
        metrics.append(row_metrics)
    unique, origin_sets, _ = temporal_local.voxelize_mapped_points(
        parts, list(range(len(parts))))
    cohorts, pixel_maps, grid_pixel_counts = [], [], []
    for cell_size, depth_slack in temporal_local.VISIBILITY_GRIDS:
        indices, pixels = project_visible(
            unique, camera, cell_size, depth_slack, near=.2)
        indices = np.asarray(indices, dtype=np.int64)
        pixels = np.asarray(pixels, dtype=np.int32).reshape((-1, 2))
        cohorts.append(indices)
        pixel_maps.append({int(index): tuple(map(int, pixel))
                           for index, pixel in zip(indices, pixels)})
        grid_pixel_counts.append(len(np.unique(pixels, axis=0)))
    shared = sorted(set(cohorts[0]).intersection(
        *(set(values) for values in cohorts[1:])))
    temporal_new = [index for index in shared
                    if center_index not in origin_sets[index]
                    and any(value != center_index for value in origin_sets[index])]
    pixels = np.asarray([pixel_maps[0][index] for index in temporal_new],
                        dtype=np.int32).reshape((-1, 2))
    pixels = np.unique(pixels, axis=0)
    contributing = {origin for index in temporal_new
                    for origin in origin_sets[index] if origin != center_index}
    attrs = [row["source_target_attr_canonical"] for row in metrics]
    scales = np.asarray([row["source_target_scale"] for row in metrics],
                        dtype=float)
    minimum = temporal_local.MINIMUM_EXCLUSIVE_POINTS
    gate = bool(
        all(len(indices) >= minimum and pixel_count >= minimum
            for indices, pixel_count in zip(cohorts, grid_pixel_counts))
        and len(shared) >= minimum
        and len(pixels) >= minimum
        and len(temporal_new) >= minimum
        and len(contributing) >=
        temporal_local.MINIMUM_NONCENTER_CONTRIBUTING_FRAMES
        and len(set(attrs)) == 1
        and float(np.ptp(scales, axis=0).max()) <= 1e-6)
    recomputed = {
        "evidence_status": (
            "label_conditioned_temporal_local_point_prompt_opportunity"
            if gate else "insufficient"),
        "noncenter_shared_unique_pixel_count": len(pixels),
        "noncenter_contributing_frame_count": len(contributing),
        "shared_across_grids_unique_voxel_index_count": len(shared),
        "aggregate_unique_voxel_count": len(unique),
    }
    return pixels, recomputed


def discover_window_root(source_inputs: dict[str, str], sequences: set[str]) -> Path:
    roots = set()
    for raw_path in source_inputs:
        path = Path(raw_path)
        # Frozen window layout is root/sequence/{labels,pcd}/frame.ext.
        if (path.parent.name in {"labels", "pcd"}
                and path.parent.parent.name in sequences):
            roots.add(path.parent.parent.parent.resolve())
    valid = [root for root in roots if all(
        (root / sequence / "labels").is_dir()
        and (root / sequence / "pcd").is_dir() for sequence in sequences)]
    if len(valid) != 1:
        raise ValueError("frozen temporal window root is missing or ambiguous")
    return valid[0]


def discover_calibration(source_inputs: dict[str, str], camera_number: int
                         ) -> tuple[Path, dict[str, str]]:
    camera_name = f"camera{camera_number}_calib.yaml"
    candidates = [Path(path) for path in source_inputs if Path(path).name == camera_name]
    if len(candidates) != 1:
        raise ValueError("frozen camera calibration is missing or ambiguous")
    camera_path = candidates[0]
    lidar_path = camera_path.parent / "lidar2camera.yaml"
    required = {str(camera_path.resolve()): source_inputs.get(str(camera_path.resolve())),
                str(lidar_path.resolve()): source_inputs.get(str(lidar_path.resolve()))}
    if any(value is None for value in required.values()):
        raise ValueError("frozen calibration input map is incomplete")
    return camera_path.parent, required


def component_structure(mask: np.ndarray) -> dict:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    areas = sorted((int(stats[index, cv2.CC_STAT_AREA])
                    for index in range(1, count)), reverse=True)
    total = int(mask.sum())
    return {
        "foreground_pixels": total,
        "component_count_8_connected": len(areas),
        "largest_component_pixels": areas[0] if areas else 0,
        "largest_component_fraction": areas[0] / total if total else 0.0,
        "small_component_lt_8_count": sum(area < 8 for area in areas),
        "small_component_lt_8_pixels": sum(area for area in areas if area < 8),
    }


def audit_attempt(attempt: dict, final_mask: np.ndarray,
                  positive_pixels: np.ndarray, negative_pixels: np.ndarray) -> dict:
    split_positive = point_fraction(final_mask, positive_pixels, True)
    split_negative = point_fraction(final_mask, negative_pixels, False)
    raw_positive = float(attempt["positive_coverage_all_temporal_points"])
    raw_negative = float(attempt["negative_clear_all_center_conflict_points"])
    localized = localization_gate(attempt["metrics"])
    split_group_pass = bool(attempt.get("split_metrics")) and all(
        row.get("candidate_pass") is True for row in attempt["split_metrics"])
    separation_pass = bool(attempt.get("separations")) and all(
        row.get("separated_at_0_15") is True for row in attempt["separations"])
    point_pass = split_positive >= .8 and split_negative >= .8
    evidence_candidate = bool(
        point_pass and localized and split_group_pass and separation_pass)
    return {
        "raw_positive_coverage": raw_positive,
        "raw_negative_clear": raw_negative,
        "raw_point_gate_pass": bool(raw_positive >= .8 and raw_negative >= .8),
        "raw_localization_pass": bool(localized),
        "round700_attempt_accepted_immutable": bool(attempt.get("accepted")),
        "reported_split_group_pass": split_group_pass,
        "reported_separation_pass": separation_pass,
        "reported_split_metrics": attempt["split_metrics"],
        "reported_separations": attempt["separations"],
        "final_split_positive_coverage": split_positive,
        "final_split_negative_clear": split_negative,
        "final_minus_raw_positive_coverage": split_positive - raw_positive,
        "final_minus_raw_negative_clear": split_negative - raw_negative,
        "final_split_point_gate_pass": point_pass,
        "post_split_evidence_candidate": evidence_candidate,
        "rescues_raw_negative_clear_failure": bool(
            evidence_candidate and raw_negative < .8),
        "structure": component_structure(final_mask),
    }


def evaluate(source_report_path: Path, output_path: Path) -> dict:
    if output_path.exists():
        raise ValueError("output already exists")
    source, source_sha = bound_json(source_report_path)
    validate_source(source)
    source_rows = index_rows(source.get("rows", []))
    source_inputs = source["input_sha256"]
    inputs = {str(source_report_path.resolve()): source_sha}
    bind_file(Path(__file__), inputs)
    runner_path = Path(temporal_sam.__file__)
    source_runner_sha = source_inputs.get(str(runner_path.resolve()))
    if source_runner_sha is None:
        raise ValueError("Round 700 input map lacks its source runner receipt")
    # The current runner may contain later protocol-only hardening.  Bind the
    # exact helper implementation used by this audit and preserve the original
    # Round 700 runner hash separately; recomputed opportunity summaries below
    # must still agree with the frozen source rows.
    bind_file(runner_path, inputs)
    opportunity_tool_path = Path(temporal_local.__file__)
    opportunity_tool_sha = source_inputs.get(str(opportunity_tool_path.resolve()))
    if opportunity_tool_sha is None:
        raise ValueError("Round 700 input map lacks temporal point tool")
    bind_file(opportunity_tool_path, inputs)
    support = discover_support_reports(source_inputs, inputs)
    validate_report_batch(source_rows, support, opportunity_tool_sha)
    sequences = {key[0] for key in source_rows}
    window_root = discover_window_root(source_inputs, sequences)
    windows, window_hashes = {}, {}
    for sequence in sorted(sequences):
        frames, hashes, _ = temporal_local.load_sequence_window(window_root, sequence)
        if any(source_inputs.get(path) != value for path, value in hashes.items()):
            raise ValueError("temporal window differs from Round 700 input map")
        windows[sequence], window_hashes[sequence] = frames, hashes
        for path, value in hashes.items():
            bind_file(Path(path), inputs, value)
    cameras = {}
    for camera_number in sorted({key[2] for key in source_rows}):
        calibration, receipts = discover_calibration(source_inputs, camera_number)
        for path, value in receipts.items():
            bind_file(Path(path), inputs, value)
        cameras[camera_number] = load_camera(calibration, camera_number)
        if any(digest(Path(path)) != value for path, value in receipts.items()):
            raise RuntimeError("calibration changed while being parsed")
    rows = []
    for identity, source_row in sorted(source_rows.items()):
        sequence, frame_id, camera_number, target_id = identity
        frame_rows = [row for row in windows[sequence]
                      if str(row["frame_id"]) == frame_id]
        if len(frame_rows) != 1:
            raise ValueError("center frame is missing or duplicated in frozen window")
        center = frame_rows[0]
        camera = cameras[camera_number]
        positives, recomputed = recompute_round700_temporal_positive_pixels(
            windows[sequence], frame_id, camera, target_id)
        temporal_sam.validate_recomputed_opportunity(
            recomputed, source_row["recomputed_opportunity"])
        objects = center["objects"]
        projected = [value for obj in objects
                     if obj.get("obj_type") == "Pallet" and not obj.get("ignore")
                     if (value := project_box(obj, camera)) is not None]
        targets = [row for row in projected if str(row["obj_id"]) == target_id]
        if len(targets) != 1:
            raise ValueError("target projection is missing or duplicated")
        conflicts = [row[0] for row in projected_conflicts(targets[0], projected)]
        if not conflicts:
            raise ValueError("target has no projected conflict")
        visible_indices, visible_pixels = project_visible(
            center["points"], camera, cell_size=4, depth_slack=.12, near=.2)
        negatives = temporal_sam.exclusive_conflict_negative_pixels(
            center["points"], visible_indices, visible_pixels, objects,
            {str(row["obj_id"]) for row in conflicts})
        attempts = []
        policies = set()
        for attempt in source_row.get("attempts", []):
            policy = str(attempt.get("policy", ""))
            if not policy or policy in policies:
                raise ValueError("attempt policy is missing or duplicated")
            policies.add(policy)
            if "error" in attempt:
                attempts.append({"policy": policy, "source_error": attempt["error"],
                                 "post_split_evidence_candidate": False})
                continue
            path = Path(attempt.get("mask_path", ""))
            sha = str(attempt.get("mask_sha256", ""))
            if len(sha) != 64:
                raise ValueError("final split mask receipt is incomplete")
            mask = read_mask(path, sha, inputs)
            if mask.shape != targets[0]["mask"].shape:
                raise ValueError("final split mask dimensions differ")
            temporal_sam.validate_pixels(positives, mask.shape[1], mask.shape[0],
                                         "temporal positive")
            temporal_sam.validate_pixels(negatives, mask.shape[1], mask.shape[0],
                                         "center conflict negative")
            attempts.append({
                "policy": policy, "mask_path": str(path.resolve()),
                "mask_sha256": sha,
                **audit_attempt(attempt, mask, positives, negatives),
            })
        rows.append({
            "sequence": sequence, "frame_id": frame_id,
            "camera": camera_number, "target_obj_id": target_id,
            "recomputed_temporal_positive_count": len(np.unique(positives, axis=0)),
            "recomputed_center_conflict_negative_count": len(
                np.unique(negatives, axis=0)),
            "round700_recovered_immutable": bool(source_row.get("recovered")),
            "attempts": attempts,
            "post_split_evidence_candidate_count": sum(
                row.get("post_split_evidence_candidate", False) for row in attempts),
            "rescued_raw_negative_clear_failure_count": sum(
                row.get("rescues_raw_negative_clear_failure", False)
                for row in attempts),
        })
    verify_frozen(inputs)
    obj10_rows = [row for row in rows if row["target_obj_id"] == "10"]
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "offline_post_split_point_evidence_candidate_not_deployed",
        "source_report_sha256": source_sha,
        "round700_source_runner_sha256": source_runner_sha,
        "audit_runtime_runner_sha256": digest(runner_path),
        "runtime_runner_may_include_later_protocol_hardening": True,
        "round700_opportunity_tool_sha256": opportunity_tool_sha,
        "audit_runtime_opportunity_tool_sha256": digest(opportunity_tool_path),
        "historical_positive_protocol": "pre_scene_occlusion_round700",
        "tool_sha256": digest(Path(__file__)),
        "input_sha256": inputs,
        "evaluated_target_count": len(rows),
        "evaluated_final_split_mask_count": sum(
            "mask_sha256" in attempt for row in rows for attempt in row["attempts"]),
        "post_split_evidence_candidate_count": sum(
            row["post_split_evidence_candidate_count"] for row in rows),
        "rescued_raw_negative_clear_failure_count": sum(
            row["rescued_raw_negative_clear_failure_count"] for row in rows),
        "obj10": obj10_rows,
        "rows": rows,
        "round700_acceptance_or_recovered_modified": False,
        "candidate_only": True, "conditioned_on_human_3d_labels": True,
        "comparison_is_independent_validation": False,
        "human_pixel_truth": False, "approved_for_training": False,
        "human_truth_writes": 0, "training_data_writes": 0,
        "workbench_writes": 0, "oss_writes": 0,
        "production_deployed": False, "readback_verified": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    raw = output_path.read_bytes()
    if json.loads(raw.decode("utf-8")) != report:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("split point audit report readback differs")
    try:
        verify_frozen(inputs)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    report["report_path"] = str(output_path.resolve())
    report["report_sha256"] = hashlib.sha256(raw).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate(args.source_report, args.output)
    keys = ("evaluated_target_count", "evaluated_final_split_mask_count",
            "post_split_evidence_candidate_count",
            "rescued_raw_negative_clear_failure_count", "human_truth_writes",
            "training_data_writes", "workbench_writes", "oss_writes",
            "report_path", "report_sha256")
    print(json.dumps({key: report[key] for key in keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
