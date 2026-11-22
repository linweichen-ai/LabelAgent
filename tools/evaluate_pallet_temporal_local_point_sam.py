#!/usr/bin/env python3
"""Controlled SAM evaluation of frozen temporal-local pallet point prompts.

Only opportunity rows are evaluated.  Temporal positives are recomputed from
the bound 11-frame labels/PCDs; report coordinates are never trusted.  This is
candidate-only and never writes truth, training data, the workbench, or OSS.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.evaluate_pallet_temporal_local_point_opportunities as temporal_local
from tools.evaluate_pallet_box3d_overlap_pairs import (
    depth_split_masks, index_manifest_files, mask_separation,
    projected_conflicts,
)
from tools.evaluate_pallet_box3d_roi_sam import (
    association_metrics, digest, load_camera, project_box,
)
from tools.evaluate_pallet_lidar_centered_roi_sam import _crop_box, _result_mask
from tools.evaluate_pallet_lidar_conflict_points import (
    accepted_candidate, downsample_mask, indexed_prompt_group, point_fraction,
    prompt_groups, render_panel, spread_points,
)
from tools.evaluate_pallet_lidar_visibility import index_pcd_files
from tools.evaluate_pallet_multiview_prompt_opportunities import plan_targets
from tools.evaluate_projected_support import class_membership, project_visible
from tools.evaluate_slope_lidar_sam import request_json


REPORT_SCHEMA = "labelagent.pallet_temporal_local_point_sam.v1"
OPPORTUNITY_SCHEMA = "labelagent.pallet_temporal_local_point_opportunities.v1"
BASELINE_SCHEMA = "labelagent.pallet_lidar_conflict_points.v1"
EXPECTED_OPPORTUNITIES = 7
ROI_SCALE = 2


def _bound_json(path: Path) -> tuple[dict, str]:
    before = digest(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if digest(path) != before:
        raise RuntimeError(f"JSON input changed while being parsed: {path}")
    return payload, before


def _safe_child(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"input path escapes frozen root: {relative}") from error
    return path


def resolve_manifest_root(manifest_path: Path, local_root: str) -> Path:
    value = Path(local_root)
    if value.is_absolute():
        if not value.is_dir():
            raise ValueError("manifest local_root is missing")
        return value.resolve()
    candidates = []
    for candidate in (value.resolve(), (manifest_path.parent / value).resolve()):
        if candidate.is_dir() and candidate not in candidates:
            candidates.append(candidate)
    if len(candidates) != 1:
        raise ValueError("manifest local_root is missing or ambiguous")
    return candidates[0]


def _bind_file(path: Path, inputs: dict[str, str], expected: str | None = None) -> str:
    if not path.is_file():
        raise ValueError(f"frozen input is missing: {path}")
    value = digest(path)
    if expected is not None and value != expected:
        raise ValueError(f"frozen input hash differs: {path}")
    resolved = str(path.resolve())
    previous = inputs.setdefault(resolved, value)
    if previous != value:
        raise ValueError(f"conflicting frozen input identity: {path}")
    return value


def _verify_frozen(inputs: dict[str, str]) -> None:
    drifted = [path for path, expected in inputs.items()
               if not Path(path).is_file() or digest(Path(path)) != expected]
    if drifted:
        raise RuntimeError(f"temporal point SAM input changed: {drifted}")


def validate_loopback_sam_url(value: str) -> None:
    parsed = urlparse(value)
    if (parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None or parsed.password is not None):
        raise ValueError("SAM URL must use a loopback host")


def validate_reports(opportunity: dict, overlap: dict, baseline: dict) -> list[dict]:
    if (opportunity.get("schema_version") != OPPORTUNITY_SCHEMA
            or opportunity.get("status") !=
            "offline_temporal_local_point_prompt_opportunities_not_deployed"
            or opportunity.get("candidate_only") is not True
            or opportunity.get("human_pixel_truth") is not False
            or opportunity.get("approved_for_training") is not False
            or opportunity.get("production_deployed") is not False
            or opportunity.get("human_truth_writes") != 0
            or opportunity.get("training_data_writes") != 0
            or opportunity.get("workbench_writes") != 0
            or opportunity.get("oss_writes") != 0
            or not isinstance(opportunity.get("input_sha256"), dict)
            or not opportunity["input_sha256"]
            or opportunity.get("tool_sha256") != digest(Path(temporal_local.__file__))):
        raise ValueError("requires a frozen candidate-only temporal opportunity report")
    overlap_jobs = plan_targets(overlap)
    unresolved = {(job["sequence"], job["frame_id"], job["primary_camera"],
                   job["target_obj_id"]) for job in overlap_jobs}
    rows = [row for row in opportunity.get("rows", [])
            if row.get("evidence_status") ==
            "label_conditioned_temporal_local_point_prompt_opportunity"]
    if (len(rows) != EXPECTED_OPPORTUNITIES
            or opportunity.get("opportunity_count") != len(rows)):
        raise ValueError("requires exactly seven frozen opportunity rows")
    identities, selected = set(), []
    for row in rows:
        identity = (str(row["sequence"]), str(row["frame_id"]),
                    int(row["primary_camera"]), str(row["target_obj_id"]))
        if identity in identities or identity not in unresolved:
            raise ValueError("opportunity row identity is duplicate or not unresolved")
        identities.add(identity)
        selected.append({
            "sequence": identity[0], "frame_id": identity[1],
            "camera": identity[2], "target_obj_id": identity[3],
            "opportunity_row": row,
        })
    if (baseline.get("schema_version") != BASELINE_SCHEMA
            or baseline.get("status") != "offline_candidate_diagnostic_not_deployed"
            or baseline.get("human_pixel_truth") is not False
            or baseline.get("approved_for_training") is not False
            or baseline.get("production_deployed") is not False
            or baseline.get("workbench_writes") != 0
            or baseline.get("training_data_writes") != 0
            or baseline.get("oss_writes") != 0
            or not isinstance(baseline.get("input_sha256"), dict)
            or not baseline["input_sha256"]):
        raise ValueError("requires a frozen non-deployed round677 baseline")
    return selected


def index_baseline_rows(baseline: dict) -> dict[tuple[str, str, int, str], dict]:
    result = {}
    for row in baseline.get("rows", []):
        key = (str(row["sequence"]), str(row["frame_id"]), int(row["camera"]),
               str(row["target_obj_id"]))
        if key in result:
            raise ValueError("duplicate baseline identity")
        result[key] = row
    return result


def recompute_temporal_positive_pixels(frames: list[dict], center_frame: str,
                                       camera: dict, target_id: str
                                       ) -> tuple[np.ndarray, dict]:
    """Recompute stable temporal-new pixels without trusting report coordinates."""
    summary = temporal_local.analyze_target(
        frames, center_frame, camera, target_id)
    center_index = next(index for index, row in enumerate(frames)
                        if row["frame_id"] == center_frame)
    center_target = temporal_local.select_target_object(
        frames[center_index]["objects"], target_id)
    parts = [temporal_local.remap_exclusive_target_points(
        row["points"], row["objects"], target_id, center_target)[0]
        for row in frames]
    unique, origin_sets, _ = temporal_local.voxelize_mapped_points(
        parts, list(range(len(parts))))
    cohorts, pixel_maps = [], []
    for cell_size, depth_slack in temporal_local.VISIBILITY_GRIDS:
        indices, pixels = project_visible(
            unique, camera, cell_size, depth_slack, near=.2)
        indices = np.asarray(indices, dtype=np.int64)
        cohorts.append(indices)
        pixel_maps.append({int(index): tuple(map(int, pixel))
                           for index, pixel in zip(indices, pixels)})
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
    recomputed = {
        "evidence_status": summary["evidence_status"],
        "noncenter_shared_unique_pixel_count": len(pixels),
        "noncenter_contributing_frame_count": len(contributing),
        "shared_across_grids_unique_voxel_index_count": len(shared),
        "aggregate_unique_voxel_count": len(unique),
    }
    return pixels, recomputed


def validate_recomputed_opportunity(recomputed: dict, reported: dict) -> None:
    keys = ("evidence_status", "noncenter_shared_unique_pixel_count",
            "noncenter_contributing_frame_count",
            "shared_across_grids_unique_voxel_index_count",
            "aggregate_unique_voxel_count")
    if any(recomputed.get(key) != reported.get(key) for key in keys):
        raise ValueError("recomputed temporal opportunity differs from frozen report")
    if recomputed["evidence_status"] != (
            "label_conditioned_temporal_local_point_prompt_opportunity"):
        raise ValueError("temporal opportunity no longer passes its frozen gate")


def exclusive_conflict_negative_pixels(points: np.ndarray,
                                       visible_indices: np.ndarray,
                                       visible_pixels: np.ndarray,
                                       objects: list[dict],
                                       conflict_ids: set[str]) -> np.ndarray:
    bits, owner, boxes = class_membership(points, objects)
    conflict_owners = {index + 1 for index, row in enumerate(boxes)
                       if row["bit"] == 1 and str(row["obj_id"]) in conflict_ids}
    if len(conflict_owners) != len(conflict_ids):
        raise ValueError("conflict pallet identity is missing or duplicated")
    visible_indices = np.asarray(visible_indices, dtype=np.int64)
    selected = np.asarray([
        owner[index] in conflict_owners and bits[index] == 1
        for index in visible_indices], dtype=bool)
    return np.unique(np.asarray(visible_pixels, dtype=np.int32)[selected], axis=0)


def validate_pixels(points: np.ndarray, width: int, height: int, name: str) -> None:
    values = np.asarray(points)
    if (values.ndim != 2 or values.shape[1] != 2 or not len(values)
            or values.dtype.kind not in "iu"
            or np.any(values[:, 0] < 0) or np.any(values[:, 0] >= width)
            or np.any(values[:, 1] < 0) or np.any(values[:, 1] >= height)):
        raise ValueError(f"{name} points are empty, malformed, or outside the image")


def disjoint_prompt_negatives(positive_pixels: np.ndarray,
                              negative_pixels: np.ndarray
                              ) -> tuple[np.ndarray, int]:
    positives = set(map(tuple, np.asarray(positive_pixels, dtype=np.int32).tolist()))
    negatives = np.asarray(negative_pixels, dtype=np.int32).reshape((-1, 2))
    overlap = positives & set(map(tuple, negatives.tolist()))
    prompt_pool = np.asarray([
        point for point in negatives if tuple(map(int, point)) not in overlap],
        dtype=np.int32).reshape((-1, 2))
    return prompt_pool, len(overlap)


def normalized_batch_items(groups: list[tuple[str, list]], payload: dict) -> list[dict]:
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError("SAM batch response has no items list")
    indexed = {}
    for item in items:
        if not isinstance(item, dict) or "index" not in item:
            raise ValueError("SAM batch item has no index")
        if type(item["index"]) is not int:
            raise ValueError("SAM batch item index must be an integer")
        index = item["index"]
        indexed_prompt_group(groups, index)
        if index in indexed:
            raise ValueError("SAM returned a duplicate prompt group index")
        indexed[index] = item
    return [indexed.get(index, {"index": index,
                                "error": "missing_response_item"})
            for index in range(len(groups))]


def _baseline_delta(current: bool, baseline: bool) -> str:
    if current and not baseline:
        return "newly_recovered_vs_round677_same_source"
    if current and baseline:
        return "recovered_in_both_same_source_runs"
    if not current and baseline:
        return "regressed_vs_round677_same_source"
    return "not_recovered_in_either_same_source_run"


def evaluate(opportunity_report_path: Path, split_report_path: Path,
             baseline_report_path: Path, manifest_path: Path,
             pcd_manifest_path: Path, window_root: Path, labels_root: Path,
             calibration: Path, pcd_root: Path, output: Path, sam_url: str,
             timeout: float = 120) -> dict:
    if output.exists():
        raise ValueError("output directory already exists")
    validate_loopback_sam_url(sam_url)
    opportunity, opportunity_sha = _bound_json(opportunity_report_path)
    overlap, overlap_sha = _bound_json(split_report_path)
    baseline, baseline_sha = _bound_json(baseline_report_path)
    manifest, manifest_sha = _bound_json(manifest_path)
    pcd_manifest, pcd_manifest_sha = _bound_json(pcd_manifest_path)
    selected = validate_reports(opportunity, overlap, baseline)
    if opportunity.get("overlap_report_sha256") != overlap_sha:
        raise ValueError("opportunity report is not bound to this overlap report")
    if (manifest.get("evaluation_only") is not True
            or manifest.get("training_return_allowed") is not False
            or manifest.get("candidate_write_allowed") is not False
            or manifest.get("human_truth_write_allowed") is not False
            or pcd_manifest.get("schema_version") != "labelagent.pcd_manifest.v1"):
        raise ValueError("requires frozen evaluation-only image and PCD manifests")
    inputs = {
        str(opportunity_report_path.resolve()): opportunity_sha,
        str(split_report_path.resolve()): overlap_sha,
        str(baseline_report_path.resolve()): baseline_sha,
        str(manifest_path.resolve()): manifest_sha,
        str(pcd_manifest_path.resolve()): pcd_manifest_sha,
    }
    baseline_hashes = set(baseline["input_sha256"].values())
    if not {overlap_sha, manifest_sha, pcd_manifest_sha}.issubset(baseline_hashes):
        raise ValueError("round677 baseline is not bound to the same frozen sources")
    _bind_file(Path(__file__), inputs)
    _bind_file(Path(temporal_local.__file__), inputs)
    cameras_needed = {row["camera"] for row in selected}
    opportunity_inputs = opportunity["input_sha256"]
    opportunity_overlap_path = str(split_report_path.resolve())
    if opportunity_inputs.get(opportunity_overlap_path) != overlap_sha:
        raise ValueError("opportunity input map is not bound to this overlap path")
    calibration_paths = [calibration / "lidar2camera.yaml"]
    for camera_number in sorted(cameras_needed):
        calibration_paths.append(calibration / f"camera{camera_number}_calib.yaml")
    for path in calibration_paths:
        expected = opportunity_inputs.get(str(path.resolve()))
        if expected is None:
            raise ValueError("opportunity input map is missing calibration")
        _bind_file(path, inputs, expected)
    cameras = {number: load_camera(calibration, number)
               for number in sorted(cameras_needed)}
    _verify_frozen({str(path.resolve()): opportunity_inputs[str(path.resolve())]
                    for path in calibration_paths})
    windows, window_hashes = {}, {}
    for sequence in sorted({row["sequence"] for row in selected}):
        frames, hashes, _ = temporal_local.load_sequence_window(
            window_root, sequence)
        if any(opportunity_inputs.get(path) != value
               for path, value in hashes.items()):
            raise ValueError("temporal window differs from opportunity input map")
        windows[sequence] = frames
        window_hashes[sequence] = hashes
        inputs.update(hashes)
    files = index_manifest_files(manifest["files"])
    pcd_files = index_pcd_files(pcd_manifest["files"])
    image_root = resolve_manifest_root(manifest_path, str(manifest["local_root"]))
    labels_root, pcd_root = labels_root.resolve(), pcd_root.resolve()
    baseline_rows = index_baseline_rows(baseline)
    contexts = []
    for row in selected:
        sequence, frame, camera_number, target_id = (
            row["sequence"], row["frame_id"], row["camera"], row["target_obj_id"])
        identity = (sequence, frame, camera_number, target_id)
        if identity not in baseline_rows:
            raise ValueError("round677 baseline identity is missing")
        item = files.get((sequence, camera_number, frame))
        pcd_item = pcd_files.get((sequence, frame))
        if item is None or pcd_item is None:
            raise ValueError("frozen image or PCD manifest identity is missing")
        image_path = _safe_child(image_root, item["relative_path"])
        pcd_path = _safe_child(pcd_root, pcd_item["filename"])
        image_sha = _bind_file(image_path, inputs, item["sha256"])
        pcd_sha = _bind_file(pcd_path, inputs, pcd_item["sha256"])
        label_path = _safe_child(labels_root / sequence, f"{frame}.json")
        label, label_sha = _bound_json(label_path)
        inputs[str(label_path.resolve())] = label_sha
        if str(label.get("frame_id")) != frame:
            raise ValueError("center label frame identity differs")
        frame_rows = [value for value in windows[sequence]
                      if value["frame_id"] == frame]
        if len(frame_rows) != 1:
            raise ValueError("center frame is absent from temporal window")
        center_window_hashes = window_hashes[sequence]
        temporal_label_sha = next((value for path, value in center_window_hashes.items()
            if Path(path).parent.name == "labels" and Path(path).stem == frame), None)
        temporal_pcd_sha = next((value for path, value in center_window_hashes.items()
            if Path(path).parent.name == "pcd" and Path(path).stem == frame), None)
        if label_sha != temporal_label_sha or pcd_sha != temporal_pcd_sha:
            raise ValueError("center label/PCD differs from temporal frozen window")
        objects = label.get("objects")
        if not isinstance(objects, list):
            raise ValueError("center label objects are absent")
        camera = cameras[camera_number]
        recomputed_pixels, recomputed = recompute_temporal_positive_pixels(
            windows[sequence], frame, camera, target_id)
        validate_recomputed_opportunity(recomputed, row["opportunity_row"])
        image_before = digest(image_path)
        image = Image.open(image_path).convert("RGB")
        image.load()
        if digest(image_path) != image_before or image_before != image_sha:
            raise RuntimeError("image changed while being decoded")
        width, height = image.size
        validate_pixels(recomputed_pixels, width, height, "temporal positive")
        projected = [value for obj in objects
                     if obj.get("obj_type") == "Pallet" and not obj.get("ignore")
                     if (value := project_box(obj, camera)) is not None]
        targets = [value for value in projected if str(value["obj_id"]) == target_id]
        if len(targets) != 1:
            raise ValueError("target projection is missing or duplicated")
        target = targets[0]
        conflicts = [value[0] for value in projected_conflicts(target, projected)]
        if not conflicts:
            raise ValueError("opportunity target has no projected conflict")
        selected_projected = [target, *conflicts]
        center_points = frame_rows[0]["points"]
        visible_indices, visible_pixels = project_visible(
            center_points, camera, cell_size=4, depth_slack=.12, near=.2)
        conflict_ids = {str(value["obj_id"]) for value in conflicts}
        negative_pixels = exclusive_conflict_negative_pixels(
            center_points, visible_indices, visible_pixels, objects, conflict_ids)
        validate_pixels(negative_pixels, width, height, "center conflict negative")
        bbox = [min(value["bbox"][0] for value in selected_projected),
                min(value["bbox"][1] for value in selected_projected),
                max(value["bbox"][2] for value in selected_projected),
                max(value["bbox"][3] for value in selected_projected)]
        roi = _crop_box(bbox, width, height, minimum_side=240, padding=24)
        positive_in_roi = np.asarray([point for point in recomputed_pixels
            if roi[0] <= point[0] < roi[2] and roi[1] <= point[1] < roi[3]],
            dtype=np.int32).reshape((-1, 2))
        negative_in_roi = np.asarray([point for point in negative_pixels
            if roi[0] <= point[0] < roi[2] and roi[1] <= point[1] < roi[3]],
            dtype=np.int32).reshape((-1, 2))
        if len(positive_in_roi) != len(recomputed_pixels):
            raise ValueError("temporal positive point falls outside the frozen ROI")
        positives = spread_points(positive_in_roi, 3)
        # A projected pixel can be supported by a temporal target point and a
        # center-frame occluder point.  Never send contradictory labels for the
        # same prompt pixel; retain the overlap in both full acceptance sets so
        # it still makes acceptance appropriately harder.
        negative_prompt_pool, overlap_count = disjoint_prompt_negatives(
            positive_in_roi, negative_in_roi)
        negatives = spread_points(negative_prompt_pool, 4)
        groups = prompt_groups(positives, negatives, roi, ROI_SCALE,
                               positive_only_fallback=True,
                               sparse_negative_fallback=True,
                               pairwise_negative_fallback=False)
        required = {"one_positive_all_negative", "three_positive_all_negative",
                    "three_positive_two_negative", "three_positive_only"}
        if not positives or not negatives or not required.issubset(
                {name for name, _ in groups}):
            raise ValueError("empty or incomplete frozen prompt strategy set")
        other_masks = {}
        for value in selected_projected:
            receipt = next(split_row for split_row in overlap["rows"]
                           if str(split_row["sequence"]) == sequence
                           and str(split_row["frame_id"]) == frame
                           and int(split_row["camera"]) == camera_number)[
                               "depth_split_masks"][str(value["obj_id"])]
            mask_path = Path(receipt["path"])
            _bind_file(mask_path, inputs, receipt["sha256"])
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if digest(mask_path) != receipt["sha256"]:
                raise RuntimeError("frozen split mask changed while being decoded")
            if mask is None or mask.shape != (height, width):
                raise ValueError("frozen split mask dimensions differ")
            if str(value["obj_id"]) != target_id:
                other_masks[value["obj_id"]] = mask > 0
        contexts.append({
            "identity": identity, "image": image, "target": target,
            "conflicts": conflicts, "selected": selected_projected,
            "positives_all": recomputed_pixels,
            "negatives_all": negative_pixels,
            "positives": positives, "negatives": negatives,
            "groups": groups, "roi": roi, "other_masks": other_masks,
            "positive_negative_pixel_overlap_count": overlap_count,
            "baseline_recovered": bool(baseline_rows[identity].get("recovered")),
            "recomputed": recomputed,
        })
    _verify_frozen(inputs)
    health_before = request_json(sam_url.rstrip("/") + "/health", timeout=15)
    if health_before.get("status") != "ok" or health_before.get("model_loaded") is not True:
        raise ValueError("SAM service is not ready")
    output.mkdir(parents=True); masks_dir = output / "masks"; masks_dir.mkdir()
    panels, rows = [], []
    try:
        for context in contexts:
            sequence, frame, camera_number, target_id = context["identity"]
            image, roi = context["image"], context["roi"]
            crop = image.crop(tuple(roi)).resize(
                ((roi[2] - roi[0]) * ROI_SCALE,
                 (roi[3] - roi[1]) * ROI_SCALE), Image.Resampling.LANCZOS)
            stream = io.BytesIO(); crop.save(stream, "PNG")
            payload = request_json(
                sam_url.rstrip("/") + "/segment_point_groups_batch", {
                    "image": base64.b64encode(stream.getvalue()).decode("ascii"),
                    "prompt_groups": [points for _, points in context["groups"]],
                }, timeout=timeout)
            items = normalized_batch_items(context["groups"], payload)
            attempts = []
            for item in items:
                index = int(item["index"])
                policy, _ = indexed_prompt_group(context["groups"], index)
                if "result" not in item:
                    attempts.append({"policy": policy,
                                     "error": item.get("error", "missing result"),
                                     "accepted": False})
                    continue
                value = item["result"]
                model_local = _result_mask(value, crop.width, crop.height)
                local = downsample_mask(
                    model_local, roi[2] - roi[0], roi[3] - roi[1])
                proposal = np.zeros((image.height, image.width), dtype=bool)
                proposal[roi[1]:roi[3], roi[0]:roi[2]] = local
                metrics = association_metrics(
                    proposal, context["target"]["mask"],
                    [row["mask"] for row in context["conflicts"]],
                    context["target"]["center"])
                positive_coverage = point_fraction(
                    proposal, context["positives_all"], True)
                negative_clear = point_fraction(
                    proposal, context["negatives_all"], False)
                candidate_masks = {
                    context["target"]["obj_id"]: proposal,
                    **context["other_masks"],
                }
                split_masks, split_metrics = depth_split_masks(
                    context["selected"], candidate_masks)
                separations = [mask_separation(
                    split_masks[context["target"]["obj_id"]],
                    split_masks[row["obj_id"]])
                    for row in context["conflicts"]]
                accepted = accepted_candidate(
                    metrics, positive_coverage, negative_clear,
                    split_metrics, separations)
                stem = f"s{sequence}-f{frame[-8:]}-c{camera_number}-o{target_id}-p{index}"
                raw_path = masks_dir / f"{stem}-raw.png"
                mask_path = masks_dir / f"{stem}.png"
                Image.fromarray(proposal.astype(np.uint8) * 255).save(raw_path)
                target_split = split_masks[context["target"]["obj_id"]]
                Image.fromarray(target_split.astype(np.uint8) * 255).save(mask_path)
                attempt = {
                    "policy": policy, "sam_score": value.get("score"),
                    "positive_coverage_all_temporal_points": positive_coverage,
                    "negative_clear_all_center_conflict_points": negative_clear,
                    "metrics": metrics, "split_metrics": split_metrics,
                    "separations": separations, "accepted": accepted,
                    "split_target_pixels": int(target_split.sum()),
                    "raw_mask_path": str(raw_path.resolve()),
                    "raw_mask_sha256": digest(raw_path),
                    "mask_path": str(mask_path.resolve()),
                    "mask_sha256": digest(mask_path),
                }
                attempts.append(attempt)
                panels.append(render_panel(
                    image, proposal, target_split, context["positives"],
                    context["negatives"], roi,
                    f"temporal local {sequence}/{target_id} {policy} accepted={accepted}"))
            accepted_attempts = [row for row in attempts if row.get("accepted")]
            accepted_attempts.sort(key=lambda row: (
                row["negative_clear_all_center_conflict_points"],
                row["positive_coverage_all_temporal_points"],
                -row["split_target_pixels"], row["policy"]), reverse=True)
            chosen = accepted_attempts[0] if accepted_attempts else None
            recovered = chosen is not None
            rows.append({
                "sequence": sequence, "frame_id": frame,
                "camera": camera_number, "target_obj_id": target_id,
                "roi_scale": ROI_SCALE,
                "temporal_validation_positive_count": len(context["positives_all"]),
                "center_conflict_validation_negative_count": len(context["negatives_all"]),
                "positive_negative_validation_pixel_overlap_count":
                    context["positive_negative_pixel_overlap_count"],
                "selected_positive_points": context["positives"],
                "selected_negative_points": context["negatives"],
                "acceptance_point_scope": (
                    "all_temporal_positive_and_all_center_conflict_negative_pixels"),
                "prompt_policies": [name for name, _ in context["groups"]],
                "attempts": attempts, "recovered": recovered,
                "selected_policy": (chosen or {}).get("policy"),
                "selected_mask_path": (chosen or {}).get("mask_path"),
                "selected_mask_sha256": (chosen or {}).get("mask_sha256"),
                "round677_center_lidar_baseline_recovered":
                    context["baseline_recovered"],
                "same_source_round677_difference": _baseline_delta(
                    recovered, context["baseline_recovered"]),
                "comparison_is_independent_validation": False,
                "recomputed_opportunity": context["recomputed"],
            })
        sheet = Image.new("RGB", (960, max(1, sum(panel.height for panel in panels))),
                          "white")
        top = 0
        for panel in panels:
            sheet.paste(panel, (0, top)); top += panel.height
        sheet_path = output / "temporal-local-point-sam-review.jpg"
        sheet.save(sheet_path, quality=90)
        health_after = request_json(sam_url.rstrip("/") + "/health", timeout=15)
        _verify_frozen(inputs)
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": "offline_temporal_local_point_sam_candidate_not_deployed",
            "roi_scale": ROI_SCALE,
            "strategy_set": (
                "original_three_plus_positive_only_plus_each_sparse_one_negative"),
            "acceptance_point_scope": (
                "all_temporal_positive_and_all_center_conflict_negative_pixels"),
            "input_sha256": inputs,
            "evaluated_opportunity_count": len(rows),
            "recovered_candidate_count": sum(row["recovered"] for row in rows),
            "round677_baseline_recovered_count": sum(
                row["round677_center_lidar_baseline_recovered"] for row in rows),
            "newly_recovered_vs_round677_same_source_count": sum(
                row["same_source_round677_difference"] ==
                "newly_recovered_vs_round677_same_source" for row in rows),
            "same_source_comparison_is_independent_validation": False,
            "sam_attempt_count": sum(len(row["attempts"]) for row in rows),
            "health_before": health_before, "health_after": health_after,
            "contact_sheet": str(sheet_path.resolve()),
            "contact_sheet_sha256": digest(sheet_path),
            "rows": rows, "candidate_only": True,
            "conditioned_on_human_3d_labels": True,
            "human_pixel_truth": False, "approved_for_training": False,
            "human_truth_writes": 0, "training_data_writes": 0,
            "workbench_writes": 0, "oss_writes": 0,
            "production_deployed": False, "readback_verified": True,
        }
        report_path = output / "results.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        raw = report_path.read_bytes()
        if json.loads(raw.decode("utf-8")) != report:
            raise RuntimeError("temporal-local SAM report readback differs")
        _verify_frozen(inputs)
        report["report_path"] = str(report_path.resolve())
        report["report_sha256"] = hashlib.sha256(raw).hexdigest()
        return report
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opportunity-report", required=True, type=Path)
    parser.add_argument("--split-report", required=True, type=Path)
    parser.add_argument("--baseline-report", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pcd-manifest", required=True, type=Path)
    parser.add_argument("--window-root", required=True, type=Path)
    parser.add_argument("--labels-root", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--pcd-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sam-url", default="http://127.0.0.1:28766")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    report = evaluate(
        args.opportunity_report, args.split_report, args.baseline_report,
        args.manifest, args.pcd_manifest, args.window_root, args.labels_root,
        args.calibration, args.pcd_root, args.output, args.sam_url, args.timeout)
    keys = ("evaluated_opportunity_count", "recovered_candidate_count",
            "round677_baseline_recovered_count",
            "newly_recovered_vs_round677_same_source_count",
            "sam_attempt_count", "workbench_writes", "report_path",
            "report_sha256")
    print(json.dumps({key: report[key] for key in keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
