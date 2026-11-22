#!/usr/bin/env python3
"""Resolve ambiguous projected pallet boxes with per-instance ROI masks."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_pallet_box3d_roi_sam import (
    association_metrics, digest, load_camera, localization_gate,
    point_fallback_groups, project_box,
)
from tools.evaluate_pallet_lidar_centered_roi_sam import (
    _crop_box, _result_mask, select_associated_result,
)
from tools.evaluate_slope_lidar_sam import request_json


PROMPTS = {"generic": "pallet", "wooden": "wooden pallet",
           "plastic": "plastic pallet", "base": "pallet base"}


def select_source_rows(source: dict, selection: str) -> list[dict]:
    localized = [row for row in source["rows"] if row.get("localization_pass")]
    if selection == "ambiguous":
        return [row for row in localized if not row.get("instance_isolation_pass")]
    if selection == "all_localized":
        return localized
    raise ValueError(f"unknown selection: {selection}")


def index_manifest_files(rows: list[dict]) -> dict[tuple[str, int, str], dict]:
    """Index every frame independently; never collapse temporal samples."""
    result: dict[tuple[str, int, str], dict] = {}
    for row in rows:
        key = (str(row["sequence"]),
               int(str(row["camera"]).split("_")[-1]),
               str(row["frame_id"]))
        if key in result:
            raise ValueError(f"duplicate manifest image identity: {key}")
        result[key] = row
    return result


def projected_conflicts(target, projected, threshold=.25):
    result = []
    target_area = int(target["mask"].sum())
    for other in projected:
        if other is target:
            continue
        other_area = int(other["mask"].sum())
        overlap = int((target["mask"] & other["mask"]).sum())
        target_fraction = overlap / target_area if target_area else 0.0
        other_fraction = overlap / other_area if other_area else 0.0
        if max(target_fraction, other_fraction) >= threshold:
            result.append((other, target_fraction, other_fraction))
    return result


def projected_conflict_component(target, projected, threshold=.25):
    """Return the full transitive overlap component containing ``target``."""
    selected_ids = {str(target["obj_id"])}
    frontier = [target]
    while frontier:
        current = frontier.pop(0)
        for other, _, _ in projected_conflicts(current, projected, threshold):
            identity = str(other["obj_id"])
            if identity in selected_ids:
                continue
            selected_ids.add(identity)
            frontier.append(other)
    return [target, *(row for row in projected
                      if row is not target and str(row["obj_id"]) in selected_ids)]


def projected_conflict_edges(component, threshold=.25):
    """Return every material overlap edge in a component exactly once."""
    edges, seen = [], set()
    for first in component:
        for second, first_fraction, second_fraction in projected_conflicts(
                first, component, threshold):
            key = tuple(sorted((str(first["obj_id"]), str(second["obj_id"]))))
            if key in seen:
                continue
            seen.add(key)
            edges.append((first, second, first_fraction, second_fraction))
    return edges


def mask_separation(first, second):
    first_area, second_area = int(first.sum()), int(second.sum())
    intersection = int((first & second).sum())
    union = int((first | second).sum())
    return {
        "intersection_pixels": intersection,
        "iou": intersection / union if union else 1.0,
        "intersection_over_smaller": (
            intersection / min(first_area, second_area)
            if min(first_area, second_area) else 1.0),
        "separated_at_0_15": bool(
            min(first_area, second_area)
            and intersection / min(first_area, second_area) <= .15),
    }


def remove_tiny_islands(mask, relative_area=.01, minimum_area=8):
    """Remove only disconnected specks; never expand or fill the mask."""
    source = mask.astype(np.uint8)
    total = int(source.sum())
    if not total:
        return mask.copy(), {"removed_island_count": 0,
                             "removed_island_pixels": 0}
    count, labels, stats, _ = cv2.connectedComponentsWithStats(source, 8)
    if count <= 2:
        return mask.copy(), {"removed_island_count": 0,
                             "removed_island_pixels": 0}
    threshold = max(int(minimum_area), int(math.ceil(total * relative_area)))
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(component_areas)) + 1
    keep_labels = [
        label for label in range(1, count)
        if label == largest_label or int(stats[label, cv2.CC_STAT_AREA]) >= threshold
    ]
    refined = np.isin(labels, keep_labels)
    return refined, {
        "removed_island_count": int((count - 1) - len(keep_labels)),
        "removed_island_pixels": int(total - refined.sum()),
    }


def depth_split_masks(projected, masks):
    if not projected:
        return {}, []
    shape = next(iter(masks.values())).shape
    owner = np.full(shape, -1, dtype=np.int16)
    depth = np.full(shape, np.inf, dtype=float)
    for index, row in enumerate(projected):
        # Resolve only pixels predicted for this instance.  A nearer projected
        # box may win pixels shared by two source masks, but it must never
        # inherit pixels that exist only in another instance's mask.
        eligible = (masks[row["obj_id"]] & row["mask"]
                    & (float(row["camera_depth"]) < depth))
        owner[eligible] = index
        depth[eligible] = float(row["camera_depth"])
    split, metrics = {}, []
    for index, row in enumerate(projected):
        value, cleanup = remove_tiny_islands(owner == index)
        original = masks[row["obj_id"]]
        pixels, projected_pixels, original_pixels = (
            int(value.sum()), int(row["mask"].sum()), int(original.sum()))
        projected_coverage = pixels / projected_pixels if projected_pixels else 0.0
        source_retention = int((value & original).sum()) / original_pixels if original_pixels else 0.0
        nearer_overlaps = []
        for other in projected:
            if float(other["camera_depth"]) >= float(row["camera_depth"]):
                continue
            overlap = int((row["mask"] & other["mask"]).sum())
            nearer_overlaps.append(overlap / projected_pixels if projected_pixels else 0.0)
        maximum_nearer_overlap = max(nearer_overlaps, default=0.0)
        candidate_pass = bool(
            pixels >= 32 and projected_coverage >= .03 and source_retention >= .05)
        occluded_candidate = bool(
            not candidate_pass and projected_coverage < .01
            and source_retention < .01 and maximum_nearer_overlap >= .7)
        split[row["obj_id"]] = value
        metrics.append({
            "obj_id": row["obj_id"], "camera_depth": row["camera_depth"],
            "split_pixels": pixels, "projected_box_coverage": projected_coverage,
            "source_mask_retention": source_retention,
            **cleanup,
            "maximum_nearer_projected_box_overlap": maximum_nearer_overlap,
            "candidate_pass": candidate_pass,
            "occluded_unobservable_candidate": occluded_candidate,
            "handled_candidate": candidate_pass or occluded_candidate,
        })
    return split, metrics


def infer_target(image, target, projected, sam_url, timeout):
    width, height = image.size
    roi = _crop_box(target["bbox"], width, height, minimum_side=160, padding=16)
    crop = image.crop(tuple(roi)); buffer = io.BytesIO(); crop.save(buffer, "PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    started = time.monotonic()
    response = request_json(sam_url.rstrip("/") + "/segment_text_multi", {
        "image": encoded, "prompts": PROMPTS}, timeout=timeout)
    results = [{**result, "prompt_name": name}
               for name, payload in (response.get("prompts") or {}).items()
               for result in (payload.get("results") or [])]
    target_crop = target["mask"][roi[1]:roi[3], roi[0]:roi[2]]
    cx, cy = target["center"][0] - roi[0], target["center"][1] - roi[1]
    selected, local, _, distance, _ = select_associated_result(
        results, target_crop, cx, cy, crop.width, crop.height)
    proposal = np.zeros((height, width), dtype=bool)
    other_masks = [row["mask"] for row in projected if row is not target]
    metrics, selected_name, score = None, None, None
    if selected is not None:
        proposal[roi[1]:roi[3], roi[0]:roi[2]] = local
        metrics = association_metrics(proposal, target["mask"], other_masks,
                                      target["center"])
        selected_name, score = selected.get("prompt_name"), selected.get("score")
    fallback = None
    if metrics is None or not localization_gate(metrics):
        groups = point_fallback_groups(target, roi, crop.width, crop.height)
        payload = request_json(sam_url.rstrip("/") + "/segment_point_groups_batch", {
            "image": encoded, "prompt_groups": [points for _, points in groups]},
            timeout=timeout)
        candidates, attempts = [], []
        for item in payload.get("items") or []:
            index = int(item.get("index", 0)); policy = groups[index][0]
            if "result" not in item:
                attempts.append({"policy": policy, "error": item.get("error")}); continue
            value = item["result"]
            local_mask = _result_mask(value, crop.width, crop.height)
            mask = np.zeros((height, width), dtype=bool)
            mask[roi[1]:roi[3], roi[0]:roi[2]] = local_mask
            value_metrics = association_metrics(
                mask, target["mask"], other_masks, target["center"])
            passed = localization_gate(value_metrics)
            attempts.append({"policy": policy, "score": value.get("score"),
                             "metrics": value_metrics, "localization_pass": passed})
            candidates.append((passed,
                               float(value_metrics["proposal_inside_projected_box"] or 0),
                               float(value_metrics["projected_box_coverage"] or 0),
                               float(value.get("score", 0) or 0),
                               policy, value, mask, value_metrics))
        candidates.sort(key=lambda item: item[:4], reverse=True)
        if candidates and candidates[0][0]:
            selected_point = candidates[0]
            selected_name = f"point_{selected_point[4]}"
            score, proposal, metrics = selected_point[5].get("score"), selected_point[6], selected_point[7]
        fallback = {"attempts": attempts, "selected": selected_name}
    return proposal, {
        "obj_id": target["obj_id"], "obj_attr": target["obj_attr"],
        "projected_center": target["center"], "projected_bbox": target["bbox"],
        "selected_prompt": selected_name, "score": score, "metrics": metrics,
        "localization_pass": bool(metrics and localization_gate(metrics)),
        "fallback": fallback, "elapsed_seconds": round(time.monotonic() - started, 4),
    }


def render(image, projected, masks, output):
    canvas = np.asarray(image).copy()
    colors = [(35, 210, 125), (255, 160, 35), (90, 125, 255), (220, 75, 180)]
    for index, row in enumerate(projected):
        mask = masks.get(row["obj_id"])
        if mask is None:
            continue
        color = np.asarray(colors[index % len(colors)], dtype=np.uint8)
        canvas[mask] = (canvas[mask].astype(np.uint16) + color) // 2
        cx, cy = map(int, row["center"])
        cv2.putText(canvas, str(row["obj_id"]), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, tuple(int(v) for v in color), 2)
    Image.fromarray(canvas).save(output)


def evaluate(source_report: Path, manifest_path: Path, labels_root: Path,
             calibration: Path, output: Path, sam_url: str, timeout=120,
             selection="ambiguous"):
    if output.exists():
        raise ValueError("output directory already exists")
    source = json.loads(source_report.read_text(encoding="utf-8"))
    if (source.get("scope") != "3d_box_localization_agreement_not_pixel_boundary_accuracy"
            or source.get("production_deployed") is not False):
        raise ValueError("requires the frozen non-deployed 3D ROI report")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = index_manifest_files(manifest["files"])
    root = Path(manifest["local_root"])
    output.mkdir(parents=True); masks_dir = output / "masks"; masks_dir.mkdir()
    visuals = output / "visuals"; visuals.mkdir()
    rows = []
    selected_source_rows = select_source_rows(source, selection)
    skipped_no_conflict = 0
    for source_row in selected_source_rows:
        sequence, camera_number = str(source_row["sequence"]), int(source_row["camera"])
        frame = str(source_row["frame_id"])
        item = files[(sequence, camera_number, frame)]
        image_path = root / item["relative_path"]
        if digest(image_path) != item["sha256"]:
            raise ValueError("held-out image changed")
        image = Image.open(image_path).convert("RGB")
        label = json.loads((labels_root / sequence / f"{frame}.json").read_text(encoding="utf-8"))
        camera = load_camera(calibration, camera_number)
        projected = [value for obj in label["objects"]
                     if obj.get("obj_type") == "Pallet" and not obj.get("ignore")
                     if (value := project_box(obj, camera)) is not None]
        target = next(row for row in projected
                      if str(row["obj_id"]) == str(source_row["target_obj_id"]))
        selected = projected_conflict_component(target, projected)
        conflict_edges = projected_conflict_edges(selected)
        if not conflict_edges:
            skipped_no_conflict += 1
            continue
        masks, instances = {}, []
        for projected_row in selected:
            mask, instance = infer_target(image, projected_row, projected, sam_url, timeout)
            masks[projected_row["obj_id"]] = mask
            path = masks_dir / (
                f"{sequence}-frame{frame}-camera{camera_number}-"
                f"obj{projected_row['obj_id']}.png")
            Image.fromarray(mask.astype(np.uint8) * 255).save(path)
            instance.update(mask_path=str(path.resolve()), mask_sha256=digest(path))
            instances.append(instance)
        pairs = []
        for first, second, first_fraction, second_fraction in conflict_edges:
            metrics = mask_separation(
                masks[first["obj_id"]], masks[second["obj_id"]])
            pairs.append({
                "target_obj_id": first["obj_id"],
                "other_obj_id": second["obj_id"],
                "pair_scope": "transitive_component_edge",
                "projected_target_overlap": first_fraction,
                "projected_other_overlap": second_fraction,
                **metrics,
            })
        visual = visuals / f"{sequence}-frame{frame}-camera{camera_number}.png"
        render(image, selected, masks, visual)
        split_masks, split_metrics = depth_split_masks(selected, masks)
        split_paths = {}
        for projected_row in selected:
            split_path = masks_dir / (
                f"{sequence}-frame{frame}-camera{camera_number}-"
                f"obj{projected_row['obj_id']}-depth-split.png")
            Image.fromarray(split_masks[projected_row["obj_id"]].astype(np.uint8) * 255).save(split_path)
            split_paths[str(projected_row["obj_id"])] = {
                "path": str(split_path.resolve()), "sha256": digest(split_path)}
        split_visual = visuals / (
            f"{sequence}-frame{frame}-camera{camera_number}-depth-split.png")
        render(image, selected, split_masks, split_visual)
        split_pass = bool(split_metrics) and all(row["candidate_pass"] for row in split_metrics)
        handled = bool(split_metrics) and all(row["handled_candidate"] for row in split_metrics)
        rows.append({"sequence": sequence, "frame_id": frame, "camera": camera_number,
                     "target_obj_id": target["obj_id"],
                     "conflict_count": len(selected) - 1,
                     "conflict_pair_count": len(conflict_edges),
                     "conflict_group_mode": "transitive_component",
                     "all_instances_localized": all(row["localization_pass"] for row in instances),
                     "all_conflicts_separated": bool(pairs) and all(row["separated_at_0_15"] for row in pairs),
                     "instances": instances, "pairs": pairs,
                     "visual_path": str(visual.resolve()), "visual_sha256": digest(visual),
                     "depth_split_candidate_pass": split_pass,
                     "depth_or_occlusion_handled_candidate": handled,
                     "depth_split_metrics": split_metrics,
                     "depth_split_masks": split_paths,
                     "depth_split_visual_path": str(split_visual.resolve()),
                     "depth_split_visual_sha256": digest(split_visual)})
    report = {
        "schema_version": "labelagent.pallet_box3d_overlap_pairs.v1",
        "status": "offline_instance_diagnostic_not_deployed",
        "selection": selection,
        "selected_source_row_count": len(selected_source_rows),
        "skipped_no_conflict_count": skipped_no_conflict,
        "source_report_sha256": digest(source_report),
        "ambiguous_image_count": len(rows),
        "resolved_image_count": sum(row["all_conflicts_separated"] for row in rows),
        "unresolved_image_count": sum(not row["all_conflicts_separated"] for row in rows),
        "depth_split_candidate_pass_count": sum(row["depth_split_candidate_pass"] for row in rows),
        "depth_or_occlusion_handled_candidate_count": sum(
            row["depth_or_occlusion_handled_candidate"] for row in rows),
        "workbench_writes": 0, "human_pixel_truth": False,
        "approved_for_training": False, "production_deployed": False,
        "rows": rows,
    }
    path = output / "results.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(path.resolve()); report["report_sha256"] = digest(path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--labels-root", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sam-url", default="http://127.0.0.1:28766")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--selection", choices=("ambiguous", "all_localized"),
                        default="ambiguous",
                        help="rows to audit; rows without projected conflicts are skipped")
    args = parser.parse_args()
    report = evaluate(args.source_report, args.manifest, args.labels_root,
                      args.calibration, args.output, args.sam_url, args.timeout,
                      args.selection)
    print(json.dumps({key: report[key] for key in (
        "ambiguous_image_count", "resolved_image_count", "unresolved_image_count",
        "depth_split_candidate_pass_count",
        "depth_or_occlusion_handled_candidate_count",
        "workbench_writes", "report_path", "report_sha256")},
        ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
