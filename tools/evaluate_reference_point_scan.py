"""Evaluate candidate-only point scanning for missed reference-frame instances."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from source.agent.edit_agent import _rasterize
from source.agent.mcap_video import Video
from source.agent.workflow import chat, chat_batch, sam_text_multi
from tools.tiled_candidate_utils import _iou, _mask_record


CLASS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "class": {"type": "string", "enum": [
            "person", "pallet", "other_robot", "vehicle_or_forklift",
            "chair_or_furniture", "tool_or_equipment", "ego_robot_part",
            "floor_or_structure", "unknown_object",
        ]},
        "visible_evidence": {"type": "string", "minLength": 1, "maxLength": 120},
    },
    "required": ["class", "visible_evidence"],
}

CLASS_PROMPT = (
    "Classify only the thin-red-outlined object shown at four scales. Choose the most "
    "literal visible class. person means any human body or limb; pallet means passive "
    "load support or crate stack; ego_robot_part means foreground hardware attached to "
    "the camera vehicle; floor_or_structure includes racks, walls, beams and fixed "
    "equipment; vehicle_or_forklift means a manned forklift, tug or ordinary industrial "
    "vehicle; chair_or_furniture includes seats and wheeled chairs; tool_or_equipment "
    "includes a manual pallet jack, loose cart, machine part or hand-operated device. "
    "other_robot requires concrete visible evidence of a separate unmanned motorized "
    "autonomous chassis or AGV, not merely a low platform, seat, pallet, box, tool or "
    "forklift. A parked or currently unoccupied conventional forklift is still "
    "vehicle_or_forklift: a forklift mast, forks, operator compartment, seat, steering "
    "controls or manual pull handle are evidence against other_robot, even when no driver "
    "is visible. Use unknown_object when motorization or autonomy is not visibly supported. "
    "The fourth panel highlights the candidate mask in cyan. Judge the outlined candidate, "
    "not a nearby object in the wider context. Do not use overlay color as evidence."
)

EXEMPLAR_PROMPT = CLASS_PROMPT + (
    " The second image is a sheet of confirmed project-specific other_robot positive "
    "examples outlined in green. In this project, other_robot may be either the visible "
    "body of a separate autonomous industrial mobile robot or a clearly associated "
    "visible part such as its fork tip or attachment. Use the sheet to understand that "
    "project convention, but do not classify an object as other_robot merely because it "
    "has a similar color, is low to the floor, or is near a pallet. The first image is "
    "the candidate to classify; the second image is reference only."
)

SURFACE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": [
            "physical_object", "background_surface", "uncertain"]},
        "visible_evidence": {"type": "string", "minLength": 1, "maxLength": 120},
    },
    "required": ["decision", "visible_evidence"],
}

SURFACE_PROMPT = (
    "This image contains only the candidate pixels on a uniform gray background. Decide "
    "whether the visible pixels themselves form a coherent physical item, or are merely a "
    "patch of floor, wall, ceiling, glare or empty background. A small blurry box-like item "
    "still counts as physical_object. Do not decide its semantic class and do not infer from "
    "anything outside the isolated pixels. Use uncertain when evidence is insufficient."
)


def fixed_point_grid(width, height, columns=24, rows=12, top_fraction=2 / 3):
    """Return deterministic interior scan points without using class or color cues."""
    if (type(width) is not int or type(height) is not int or width < columns
            or height < rows or not 2 <= columns <= 64 or not 2 <= rows <= 64
            or not 0 < top_fraction <= 1):
        raise ValueError("invalid point grid")
    top = height * top_fraction
    points = [
        {"x": min(width - 1, round(column * width / columns)),
         "y": min(height - 1, round(row * height / rows))}
        for row in range(1, rows + 1)
        if row * height / rows <= top + 1e-9
        for column in range(1, columns)
    ]
    if not points:
        raise ValueError("point grid contains no scan points")
    return points


def point_batch(image, points, endpoint):
    body = {"image": base64.b64encode(image).decode("ascii"), "points": points}
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/segment_points_batch",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1200) as response:
        payload = json.load(response)
    if (not isinstance(payload, dict) or not isinstance(payload.get("items"), list)
            or payload.get("diagnostics", {}).get("image_encoded_once") is not True
            or payload.get("diagnostics", {}).get("point_count") != len(points)):
        raise ValueError("point batch response did not prove one-pass complete coverage")
    return payload


def sam_shapes(payload, label):
    shapes = []
    for index, result in enumerate(payload.get("results") or [], 1):
        parts = result.get("parts") or []
        if not parts:
            continue
        shapes.append({
            "id": index, "label": label,
            "points": parts[0]["points"], "holes": parts[0].get("holes", []),
            "parts": parts, "score": result.get("score"),
        })
    return shapes


def known_instance_masks(image, width, height, config):
    outputs, diagnostics = sam_text_multi(
        image, {"person": "person", "pallet": "pallet"}, config)
    masks = {}
    for label, payload in outputs.items():
        masks[label] = [
            _rasterize(shape, width, height).astype(bool)
            for shape in sam_shapes(payload, label)
        ]
    return masks, diagnostics


def known_coverage(mask, area, known_masks):
    return {
        label: max([float(np.count_nonzero(mask & known)) / area
                    for known in masks] or [0.0])
        for label, masks in known_masks.items()
    }


def bounded_mask_record(shape, width, height):
    """Rasterize only the polygon envelope while preserving full-image geometry."""
    parts = shape.get("parts") or [shape]
    point_arrays = [
        np.asarray(part.get("points", []), dtype=np.int32)
        for part in parts if len(part.get("points", [])) >= 3]
    if not point_arrays:
        raise ValueError("Empty candidate raster")
    vertices = np.concatenate(point_arrays, axis=0)
    x0 = max(0, int(vertices[:, 0].min()))
    y0 = max(0, int(vertices[:, 1].min()))
    x1 = min(width, int(vertices[:, 0].max()) + 1)
    y1 = min(height, int(vertices[:, 1].max()) + 1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Empty candidate raster")
    shifted_parts = []
    for part in parts:
        shifted_parts.append({
            "points": [[int(x) - x0, int(y) - y0]
                       for x, y in part.get("points", [])],
            "holes": [[[int(x) - x0, int(y) - y0] for x, y in hole]
                      for hole in part.get("holes", [])],
        })
    mask = _rasterize({"parts": shifted_parts}, x1 - x0, y1 - y0)
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError("Empty candidate raster")
    local_x0, local_x1 = int(columns.min()), int(columns.max()) + 1
    local_y0, local_y1 = int(rows.min()), int(rows.max()) + 1
    return {
        "bbox": (x0 + local_x0, y0 + local_y0,
                 x0 + local_x1, y0 + local_y1),
        "mask": mask[local_y0:local_y1, local_x0:local_x1].copy(),
        "area": len(rows),
    }


def known_coverage_record(record, known_masks):
    x0, y0, x1, y1 = record["bbox"]
    candidate = record["mask"].astype(bool)
    return {
        label: max([
            float(np.count_nonzero(candidate & known[y0:y1, x0:x1]))
            / record["area"] for known in masks] or [0.0])
        for label, masks in known_masks.items()
    }


def filtered_candidates(batch, width, height, known_masks):
    raw = []
    for item in batch["items"]:
        result = item.get("result")
        if not result or not result.get("parts"):
            continue
        shape = {
            "id": len(raw) + 1, "label": "unknown_object",
            "points": result["points"], "holes": result.get("holes", []),
            "parts": result["parts"], "score": float(result["score"]),
            "point": item["point"],
        }
        record = bounded_mask_record(shape, width, height)
        shape.update(_record=record, bbox=list(record["bbox"]), area=record["area"])
        raw.append(shape)
    minimum_area = max(32, round(width * height * 0.000033))
    geometric = [shape for shape in raw
                 if shape["score"] >= 0.80
                 and minimum_area <= shape["area"] <= width * height * 0.08
                 and shape["bbox"][2] - shape["bbox"][0] <= width * 0.5
                 and shape["bbox"][3] - shape["bbox"][1] <= height * 0.5]
    kept = []
    for shape in sorted(geometric, key=lambda item: item["score"], reverse=True):
        if any(_iou(shape["_record"], prior["_record"]) >= 0.85 for prior in kept):
            continue
        coverage = known_coverage_record(shape["_record"], known_masks)
        shape["known_coverage"] = coverage
        shape["passes_known_exclusion"] = max(coverage.values(), default=0.0) < 0.5
        kept.append(shape)
    return raw, geometric, kept


def multiscale_panel(image, shape, identifier):
    width, height = image.size
    x0, y0, x1, y1 = shape["bbox"]
    panels = []
    center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
    for factor in (12, 8, 4):
        box_width = max(160, (x1 - x0) * factor)
        box_height = max(100, (y1 - y0) * factor)
        box = (max(0, int(center_x - box_width / 2)),
               max(0, int(center_y - box_height / 2)),
               min(width, int(center_x + box_width / 2)),
               min(height, int(center_y + box_height / 2)))
        crop = image.crop(box).resize((512, 384), Image.Resampling.LANCZOS)
        scale_x = 512 / max(1, box[2] - box[0])
        scale_y = 384 / max(1, box[3] - box[1])
        polygon = [((x - box[0]) * scale_x, (y - box[1]) * scale_y)
                   for x, y in shape["points"]]
        if polygon:
            ImageDraw.Draw(crop).line(
                polygon + [polygon[0]], fill=(255, 40, 40), width=5)
        panels.append(crop)
    box = (max(0, x0 - 40), max(0, y0 - 40),
           min(width, x1 + 40), min(height, y1 + 40))
    crop = image.crop(box)
    mask = _rasterize(shape, width, height)[box[1]:box[3], box[0]:box[2]].astype(bool)
    pixels = np.asarray(crop).copy()
    pixels[mask] = (
        pixels[mask] * 0.75 + np.asarray([0, 230, 255]) * 0.25).astype(np.uint8)
    panels.append(Image.fromarray(pixels).resize((512, 384), Image.Resampling.LANCZOS))
    board = Image.new("RGB", (1024, 768))
    labels = ("WIDE CONTEXT", "CONTEXT", "CLOSE CROP", "MASK / cyan tint")
    for index, (panel, label) in enumerate(zip(panels, labels)):
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, 320, 28), fill=(0, 0, 0))
        draw.text((6, 6), f"{identifier} {label}", fill=(255, 255, 255))
        board.paste(panel, ((index % 2) * 512, (index // 2) * 384))
    output = io.BytesIO()
    board.save(output, "JPEG", quality=92)
    return output.getvalue()


def isolated_mask_image(image, shape):
    width, height = image.size
    x0, y0, x1, y1 = shape["bbox"]
    box = (max(0, x0 - 40), max(0, y0 - 40),
           min(width, x1 + 40), min(height, y1 + 40))
    crop = image.crop(box)
    mask = _rasterize(shape, width, height)[box[1]:box[3], box[0]:box[2]].astype(bool)
    source_pixels = np.asarray(crop).copy()
    pixels = np.full_like(source_pixels, 118)
    pixels[mask] = source_pixels[mask]
    isolated = Image.fromarray(pixels).resize((512, 384), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    isolated.save(output, "JPEG", quality=95)
    return output.getvalue()


def artifact_frame_token(frame):
    """Keep generated artifact names bounded on Windows and traceable by hash."""
    text = str(frame)
    safe = "".join(character if (character.isalnum() or character in "-_")
                   else "-" for character in text).strip("-")
    if safe and len(safe) <= 24:
        return safe
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"id-{digest}"


def classify_identified_candidates(config, identified, image, output, frame,
                                   exemplar_sheet=None, artifact_suffix=""):
    requests = []
    for identifier, shape in identified:
        data = multiscale_panel(image, shape, identifier)
        path = output / (
            f"frame-{artifact_frame_token(frame)}-{identifier.lower()}-"
            f"multiscale{artifact_suffix}.jpg")
        path.write_bytes(data)
        requests.append({"identifier": identifier, "shape": shape, "image": data,
                         "artifact": str(path.resolve())})
    results, durations = [], []
    for start in range(0, len(requests), 3):
        group = requests[start:start + 3]
        started = time.perf_counter()
        prompt = EXEMPLAR_PROMPT if exemplar_sheet else CLASS_PROMPT
        extra_images = (exemplar_sheet,) if exemplar_sheet else ()
        if len(group) == 1:
            values = [chat(config, prompt, group[0]["image"],
                           extra_images=extra_images,
                           response_schema=CLASS_SCHEMA)]
        else:
            values = chat_batch(
                config,
                [{"instruction": prompt, "image": item["image"],
                  "extra_images": extra_images}
                 for item in group],
                response_schema=CLASS_SCHEMA, max_tokens=192)
        durations.append(time.perf_counter() - started)
        for item, value in zip(group, values):
            results.append({
                "id": item["identifier"], "result": value,
                "point": item["shape"]["point"],
                "score": item["shape"]["score"],
                "bbox": item["shape"]["bbox"],
                "area": item["shape"]["area"],
                "known_coverage": item["shape"]["known_coverage"],
                "artifact": item["artifact"],
                "_shape": item["shape"],
            })
    return results, durations


def classify_candidates(config, candidates, image, output, frame, top_k,
                        exemplar_sheet=None, artifact_suffix=""):
    selected = [item for item in candidates
                if item["passes_known_exclusion"]
                and not item.get("_foreground_ego_pair_veto")][:top_k]
    identified = [(f"C{index:02d}", shape)
                  for index, shape in enumerate(selected, 1)]
    return classify_identified_candidates(
        config, identified, image, output, frame,
        exemplar_sheet=exemplar_sheet, artifact_suffix=artifact_suffix)


def merge_exemplar_results(baseline, exemplar, allow_subset=False):
    baseline_ids = [item["id"] for item in baseline]
    exemplar_ids = [item["id"] for item in exemplar]
    if len(exemplar_ids) != len(set(exemplar_ids)):
        raise ValueError("exemplar candidate IDs must be unique")
    if (set(exemplar_ids) - set(baseline_ids)
            or (not allow_subset and exemplar_ids != baseline_ids)):
        raise ValueError("baseline and exemplar candidate order diverged")
    exemplar_by_id = {item["id"]: item for item in exemplar}
    classified = []
    for baseline_item in baseline:
        exemplar_item = exemplar_by_id.get(baseline_item["id"])
        if exemplar_item is None:
            item = baseline_item
            item["baseline_result"] = baseline_item["result"]
            item["baseline_artifact"] = baseline_item["artifact"]
            item["exemplar_evaluated"] = False
            item["dual_pass_consensus"] = False
        else:
            item = exemplar_item
            item["baseline_result"] = baseline_item["result"]
            item["baseline_artifact"] = baseline_item["artifact"]
            item["exemplar_evaluated"] = True
            item["dual_pass_consensus"] = (
                supported_other_robot(baseline_item["result"])
                and supported_other_robot(exemplar_item["result"]))
        classified.append(item)
    return classified


def supported_other_robot(result):
    return result.get("class") == "other_robot"


def foreground_ego_pair_ids(rows, width, height):
    """Find a symmetric pair of lower-image fork-like ego components."""
    eligible = []
    for row in rows:
        x0, y0, x1, y1 = row["bbox"]
        box_width, box_height = x1 - x0, y1 - y0
        center_x = (x0 + x1) / 2
        if (y0 >= height * 0.55
                and 0 < box_width <= width * 0.08
                and box_height <= height * 0.22
                and box_height >= box_width * 1.5
                and abs(center_x - width / 2) <= width * 0.20):
            eligible.append((row["id"], row["bbox"], center_x))
    paired = set()
    for index, (left_id, left_box, left_center) in enumerate(eligible):
        for right_id, right_box, right_center in eligible[index + 1:]:
            if (left_center - width / 2) * (right_center - width / 2) >= 0:
                continue
            first_width, first_height = (
                left_box[2] - left_box[0], left_box[3] - left_box[1])
            second_width, second_height = (
                right_box[2] - right_box[0], right_box[3] - right_box[1])
            overlap = max(0, min(left_box[3], right_box[3])
                          - max(left_box[1], right_box[1]))
            if (abs((left_center + right_center) / 2 - width / 2) <= width * 0.05
                    and width * 0.08 <= abs(left_center - right_center) <= width * 0.25
                    and overlap / min(first_height, second_height) >= 0.80
                    and min(first_width, second_width) / max(first_width, second_width) >= 0.75
                    and min(first_height, second_height) / max(first_height, second_height) >= 0.80):
                paired.update((left_id, right_id))
    return paired


def mark_foreground_ego_pairs(candidates, width, height):
    """Mark ego-pair geometry before the expensive VLM ranking stage."""
    eligible = [candidate for candidate in candidates
                if candidate.get("passes_known_exclusion")]
    probes = [{"id": f"P{index:02d}", "bbox": candidate["bbox"]}
              for index, candidate in enumerate(eligible, 1)]
    paired = foreground_ego_pair_ids(probes, width, height)
    records = []
    for probe, candidate in zip(probes, eligible):
        candidate["_foreground_ego_pair_veto"] = probe["id"] in paired
        if candidate["_foreground_ego_pair_veto"]:
            records.append({
                "id": probe["id"], "bbox": candidate["bbox"],
                "point": candidate.get("point"), "score": candidate.get("score"),
            })
    return records


def classification_disagreement_review_ids(rows, ego_pair_ids):
    """Route one-pass robot evidence to review without promoting it to a label."""
    review = set()
    for row in rows:
        baseline = row.get("baseline_result", {}).get("class")
        exemplar = row.get("result", {}).get("class")
        classes = {baseline, exemplar}
        if (row["id"] not in ego_pair_ids
                and "other_robot" in classes
                and classes & {"vehicle_or_forklift", "unknown_object"}):
            review.add(row["id"])
    return review


def classify_surface_veto(config, rows, image, output, frame):
    pending = [row for row in rows if row.get("dual_pass_consensus")]
    requests = []
    for row in pending:
        data = isolated_mask_image(image, row["_shape"])
        path = output / (
            f"frame-{artifact_frame_token(frame)}-{row['id'].lower()}-"
            "isolated-veto.jpg")
        path.write_bytes(data)
        requests.append({"row": row, "image": data, "artifact": str(path.resolve())})
    durations = []
    for start in range(0, len(requests), 3):
        group = requests[start:start + 3]
        started = time.perf_counter()
        if len(group) == 1:
            values = [chat(config, SURFACE_PROMPT, group[0]["image"],
                           response_schema=SURFACE_SCHEMA)]
        else:
            values = chat_batch(
                config,
                [{"instruction": SURFACE_PROMPT, "image": item["image"]}
                 for item in group],
                response_schema=SURFACE_SCHEMA, max_tokens=128)
        durations.append(time.perf_counter() - started)
        for request, value in zip(group, values):
            request["row"]["surface_veto"] = value
            request["row"]["surface_veto_artifact"] = request["artifact"]
    return durations


def evaluate_frame(frame, image_bytes, existing_annotations, config, point_endpoint,
                   output, top_k, exemplar_sheet=None,
                   adaptive_exemplar_review=False, point_grid=None):
    frame_started = time.perf_counter()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    grid = point_grid or {"columns": 24, "rows": 12,
                          "top_fraction": 2 / 3}
    points = fixed_point_grid(
        width, height, columns=grid["columns"], rows=grid["rows"],
        top_fraction=grid["top_fraction"])
    started = time.perf_counter()
    batch = point_batch(image_bytes, points, point_endpoint)
    scan_seconds = time.perf_counter() - started
    known_started = time.perf_counter()
    known_masks, known_diagnostics = known_instance_masks(
        image_bytes, width, height, config)
    known_instance_seconds = time.perf_counter() - known_started
    filter_started = time.perf_counter()
    raw, geometric, kept = filtered_candidates(
        batch, width, height, known_masks)
    preclassification_ego_pairs = mark_foreground_ego_pairs(
        kept, width, height)
    candidate_filter_seconds = time.perf_counter() - filter_started
    baseline, baseline_seconds = classify_candidates(
        config, kept, image, output, frame, top_k,
        artifact_suffix="-baseline" if exemplar_sheet else "")
    if exemplar_sheet:
        if adaptive_exemplar_review:
            review_classes = {
                "other_robot", "vehicle_or_forklift", "unknown_object",
                "tool_or_equipment"}
            identified = [
                (item["id"], item["_shape"]) for item in baseline
                if item["result"].get("class") in review_classes]
            exemplar, exemplar_seconds = classify_identified_candidates(
                config, identified, image, output, frame,
                exemplar_sheet=exemplar_sheet, artifact_suffix="-exemplar")
            classified = merge_exemplar_results(
                baseline, exemplar, allow_subset=True)
            classification_strategy = "adaptive_exemplar_review"
        else:
            exemplar, exemplar_seconds = classify_candidates(
                config, kept, image, output, frame, top_k,
                exemplar_sheet=exemplar_sheet, artifact_suffix="-exemplar")
            classified = merge_exemplar_results(baseline, exemplar)
            classification_strategy = "separate_dual_pass_batches"
        classify_seconds = {
            "baseline": baseline_seconds, "exemplar": exemplar_seconds}
    else:
        classified = baseline
        for item in classified:
            item["dual_pass_consensus"] = supported_other_robot(item["result"])
        classify_seconds = {"baseline": baseline_seconds}
        classification_strategy = "single_pass_batches"
    surface_veto_seconds = classify_surface_veto(
        config, classified, image, output, frame)
    existing_shapes = [
        shape for shape in existing_annotations
        if shape.get("label") == "other_robot"]
    existing_records = [
        _mask_record(shape, width, height) for shape in existing_shapes]
    ego_pair_ids = foreground_ego_pair_ids(classified, width, height)
    disagreement_ids = classification_disagreement_review_ids(
        classified, ego_pair_ids)
    for item in classified:
        item["foreground_ego_pair_veto"] = item["id"] in ego_pair_ids
        item["classification_disagreement_review"] = (
            item["id"] in disagreement_ids)
    selected = [item for item in classified
                if item["dual_pass_consensus"]
                and not item["foreground_ego_pair_veto"]
                and item.get("surface_veto", {}).get("decision") == "physical_object"]
    uncertain = [item for item in classified
                 if ((item["dual_pass_consensus"]
                      and not item["foreground_ego_pair_veto"]
                      and item.get("surface_veto", {}).get("decision") == "uncertain")
                     or item["classification_disagreement_review"])]
    for item in classified:
        item["existing_annotation_ious"] = [
            _iou(item["_shape"]["_record"], record)
            for record in existing_records]
        item["existing_annotation_iou"] = max(
            item["existing_annotation_ious"] or [0.0])
        item.pop("_shape", None)
    matched_existing = sum(any(
        item["existing_annotation_ious"][index] >= 0.5 for item in selected)
        for index in range(len(existing_records)))
    return {
        "frame": frame,
        "existing_other_robot_count": len(existing_records),
        "existing_other_robot_provenance": [{
            "human_truth": bool(shape.get("human_truth", False)),
            "candidate_only": bool(shape.get("candidate_only", False)),
            "approved_for_training": bool(shape.get("approved_for_training", False)),
            "review_status": shape.get("review_status"),
        } for shape in existing_shapes],
        "point_count": len(points),
        "point_grid": grid,
        "scan_seconds": scan_seconds,
        "known_instance_seconds": known_instance_seconds,
        "candidate_filter_seconds": candidate_filter_seconds,
        "frame_elapsed_seconds": time.perf_counter() - frame_started,
        "batch_diagnostics": batch["diagnostics"],
        "known_prompt_diagnostics": known_diagnostics,
        "raw_candidates": len(raw),
        "geometric_candidates": len(geometric),
        "nms_candidates": len(kept),
        "known_exclusion_passed": sum(
            shape["passes_known_exclusion"] for shape in kept),
        "classified_count": len(classified),
        "classification_batch_seconds": classify_seconds,
        "classification_strategy": classification_strategy,
        "surface_veto_batch_seconds": surface_veto_seconds,
        "selection_rule": (
            "baseline_and_exemplar_supported_other_robot"
            if exemplar_sheet else "baseline_supported_other_robot"),
        "selected_ids": [item["id"] for item in selected],
        "uncertain_review_ids": [item["id"] for item in uncertain],
        "foreground_ego_pair_veto_ids": sorted(ego_pair_ids),
        "preclassification_foreground_ego_pair_veto": preclassification_ego_pairs,
        "classification_disagreement_review_ids": sorted(disagreement_ids),
        "matched_existing_count": matched_existing,
        "unmatched_existing_annotation_ids": [
            item["id"] for item in selected
            if item["existing_annotation_iou"] < 0.5],
        "rows": classified,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--point-sam-url", default="http://127.0.0.1:28766")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--exemplar-sheet")
    parser.add_argument("--adaptive-exemplar-review", action="store_true")
    parser.add_argument("--point-columns", type=int, default=24)
    parser.add_argument("--point-rows", type=int, default=12)
    parser.add_argument("--point-top-fraction", type=float, default=2 / 3)
    args = parser.parse_args()
    if not 1 <= args.top_k <= 24:
        parser.error("--top-k must be within 1..24")
    try:
        fixed_point_grid(64, 64, columns=args.point_columns,
                         rows=args.point_rows,
                         top_fraction=args.point_top_fraction)
    except ValueError as error:
        parser.error(f"invalid point grid: {error}")
    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    topic = state.get("topic")
    annotations_by_frame = state.get("truth") or {}
    frames = [int(value.strip()) for value in args.frames.split(",") if value.strip()]
    if not frames or any(str(frame) not in annotations_by_frame for frame in frames):
        raise ValueError("every evaluation frame must have local existing annotations")
    video = Video(state["video_path"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(state.get("config") or {})
    exemplar_sheet = (Path(args.exemplar_sheet).read_bytes()
                      if args.exemplar_sheet else None)
    frame_reports = []
    for frame in frames:
        report = evaluate_frame(
            frame, video.frame(topic, frame), annotations_by_frame[str(frame)], config,
            args.point_sam_url, output, args.top_k,
            exemplar_sheet=exemplar_sheet,
            adaptive_exemplar_review=args.adaptive_exemplar_review,
            point_grid={"columns": args.point_columns, "rows": args.point_rows,
                        "top_fraction": args.point_top_fraction})
        frame_reports.append(report)
        print(json.dumps({"stage": "frame_complete", **{
            key: report[key] for key in
            ("frame", "selected_ids", "matched_existing_count",
             "unmatched_existing_annotation_ids")}},
            ensure_ascii=False), flush=True)
    summary = {
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "state_written": False,
        "project_exemplar_sheet_used": bool(exemplar_sheet),
        "frames": len(frame_reports),
        "existing_annotation_instances": sum(
            item["existing_other_robot_count"] for item in frame_reports),
        "matched_existing_annotation_instances": sum(
            item["matched_existing_count"] or 0 for item in frame_reports),
        "unmatched_candidate_count": sum(
            len(item["unmatched_existing_annotation_ids"])
            for item in frame_reports),
        "evaluation_boundary": (
            "Existing annotations are used only for overlap scoring and may be "
            "assistant-reviewed candidates rather than human truth. Unmatched candidates "
            "are neither automatic false positives nor training-approved truth."),
        "frame_reports": frame_reports,
    }
    (output / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
