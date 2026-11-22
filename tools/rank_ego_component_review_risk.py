#!/usr/bin/env python3
"""Rank ego micro-fill components for review without making decisions."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

try:
    from tools.build_ego_component_union_review import _verified_bytes, file_sha256
    from tools.build_ego_shadow_component_review import _validated_source_rows
except ModuleNotFoundError:  # Direct script execution.
    from build_ego_component_union_review import (  # type: ignore
        _verified_bytes, file_sha256)
    from build_ego_shadow_component_review import (  # type: ignore
        _validated_source_rows)


def _decode_mask(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as decoded:
        return np.asarray(decoded.convert("L")) > 0


def _decode_gray(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as decoded:
        return np.asarray(decoded.convert("L"), dtype=np.float32)


def _crop_bounds(bbox: list[int], width: int, height: int,
                 minimum_side: int = 32) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
    side = max(minimum_side, x1 - x0, y1 - y0)
    left = max(0, center_x - side // 2)
    top = max(0, center_y - side // 2)
    right = min(width, left + side)
    bottom = min(height, top + side)
    left = max(0, right - side)
    top = max(0, bottom - side)
    return left, top, right, bottom


def grayscale_features(gray: np.ndarray, component: np.ndarray,
                       bbox: list[int], original: np.ndarray | None = None
                       ) -> dict:
    if (gray.ndim != 2 or component.shape != gray.shape or not component.any()
            or original is not None and original.shape != gray.shape):
        raise ValueError("invalid grayscale feature inputs")
    height, width = gray.shape
    x0, y0, x1, y1 = _crop_bounds(bbox, width, height)
    context = gray[y0:y1, x0:x1]
    target_values = gray[component]
    kernel = np.ones((11, 11), dtype=np.uint8)
    ring = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(
        bool) & ~component
    ring_values = gray[ring]
    if not len(ring_values):
        raise ValueError("component has no surrounding ring")
    laplacian = cv2.Laplacian(context, cv2.CV_32F)
    gx = cv2.Sobel(context, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(context, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    magnitude_squared = gx * gx + gy * gy
    valid = magnitude_squared > 1e-6
    if valid.any():
        cos2 = np.zeros_like(gx)
        sin2 = np.zeros_like(gx)
        cos2[valid] = (gx[valid] * gx[valid] - gy[valid] * gy[valid]) / (
            magnitude_squared[valid])
        sin2[valid] = (2.0 * gx[valid] * gy[valid]) / magnitude_squared[valid]
        weights = gradient[valid]
        edge_coherence = float(np.hypot(
            np.average(cos2[valid], weights=weights),
            np.average(sin2[valid], weights=weights)))
    else:
        edge_coherence = 0.0
    gap_distance = None
    background_density = None
    if original is not None:
        if np.logical_and(component, original).any():
            raise ValueError("component overlaps the original mask")
        filled = original | component
        distance = cv2.distanceTransform(
            filled.astype(np.uint8), cv2.DIST_L2, 3)
        gap_distance = float(distance[component].min())
        background_density = float((~filled[y0:y1, x0:x1]).mean())
    width_pixels, height_pixels = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return {
        "target_luma_mean": round(float(target_values.mean()), 6),
        "ring_luma_mean": round(float(ring_values.mean()), 6),
        "target_ring_luma_delta_abs": round(
            abs(float(target_values.mean()) - float(ring_values.mean())), 6),
        "context_luma_std": round(float(context.std()), 6),
        "context_laplacian_variance": round(float(laplacian.var()), 6),
        "context_gradient_mean": round(float(gradient.mean()), 6),
        "context_edge_coherence": round(edge_coherence, 6),
        "gap_distance_pixels": (
            round(gap_distance, 6) if gap_distance is not None else None),
        "context_background_density": (
            round(background_density, 6)
            if background_density is not None else None),
        "bbox_elongation": round(
            max(width_pixels, height_pixels) /
            max(1, min(width_pixels, height_pixels)), 6),
        "context_bbox_xyxy": [x0, y0, x1, y1],
    }


def _percentile_ranks(values: list[float]) -> list[float]:
    """Return deterministic average tie ranks scaled to [0, 1]."""
    if not values:
        return []
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("risk features must be finite")
    if len(values) == 1:
        return [0.5]
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and array[order[end]] == array[order[start]]:
            end += 1
        average = (start + end - 1) / 2.0
        ranks[order[start:end]] = average / (len(values) - 1)
        start = end
    return [round(float(value), 9) for value in ranks]


def rank_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        raise ValueError("no components to rank")
    fields = (
        "target_luma_mean", "context_luma_std",
        "context_laplacian_variance", "context_gradient_mean",
        "context_edge_coherence", "gap_distance_pixels",
        "context_background_density", "area_pixels", "bbox_elongation")
    cameras = sorted({row.get("camera") for row in rows})
    if any(camera not in {"camera1", "camera3"} for camera in cameras):
        raise ValueError("risk rows require a known camera")
    percentiles = {field: [0.0] * len(rows) for field in fields}
    for camera in cameras:
        indexes = [index for index, row in enumerate(rows)
                   if row["camera"] == camera]
        for field in fields:
            camera_ranks = _percentile_ranks(
                [float(rows[index][field]) for index in indexes])
            for index, rank in zip(indexes, camera_ranks):
                percentiles[field][index] = rank
    ranked = []
    for index, row in enumerate(rows):
        darkness = 1.0 - percentiles["target_luma_mean"][index]
        blur = 1.0 - percentiles["context_laplacian_variance"][index]
        low_contrast = 1.0 - percentiles["context_luma_std"][index]
        visibility_risk = (
            0.80 * np.sqrt(max(0.0, darkness * blur))
            + 0.20 * low_contrast)
        gap_proximity = (
            0.60 * (1.0 - percentiles["gap_distance_pixels"][index])
            + 0.40 * percentiles["context_background_density"][index])
        edge_continuity = (
            0.60 * percentiles["context_edge_coherence"][index]
            + 0.40 * percentiles["context_gradient_mean"][index])
        weak_support = min(1.0, float(row["weak_external_support_packages"]))
        geometry = (
            0.60 * percentiles["area_pixels"][index]
            + 0.40 * percentiles["bbox_elongation"][index])
        score = (0.35 * visibility_risk + 0.25 * gap_proximity
                 + 0.20 * edge_continuity + 0.15 * weak_support
                 + 0.05 * geometry)
        unknown_high = darkness >= 0.90 and blur >= 0.90
        enriched = dict(row)
        enriched["risk_factors"] = {
            "darkness_percentile": round(darkness, 6),
            "blur_percentile": round(blur, 6),
            "low_contrast_percentile": round(low_contrast, 6),
            "visibility_risk": round(float(visibility_risk), 6),
            "gap_proximity_risk": round(gap_proximity, 6),
            "edge_continuity_risk": round(edge_continuity, 6),
            "weak_external_support_risk": round(weak_support, 6),
            "geometry_risk": round(geometry, 6),
        }
        enriched["review_priority_score"] = round(score, 6)
        enriched["priority_unknown_high"] = bool(unknown_high)
        enriched["priority_reasons"] = [
            reason for active, reason in (
                (unknown_high, "low_light_and_low_sharpness"),
                (gap_proximity >= 0.75, "near_background_or_opening"),
                (edge_continuity >= 0.75, "continuous_edge_or_seam"),
                (weak_support > 0, "weak_external_package_support"),
                (geometry >= 0.75, "large_or_elongated_micro_component"),
            ) if active]
        enriched["automatic_decision"] = None
        ranked.append(enriched)
    ranked.sort(key=lambda row: (-int(row["priority_unknown_high"]),
                                 -row["review_priority_score"],
                                 row["component_id"]))
    for rank, row in enumerate(ranked, 1):
        row["review_priority_rank"] = rank
        row["review_priority_band"] = (
            "priority_unknown_high" if row["priority_unknown_high"]
            else "priority_1" if rank <= max(1, int(np.ceil(len(ranked) * 0.10)))
            else "priority_2" if rank <= max(1, int(np.ceil(len(ranked) * 0.30)))
            else "standard")
    for camera in cameras:
        camera_rows = sorted(
            (row for row in ranked if row["camera"] == camera),
            key=lambda row: (-int(row["priority_unknown_high"]),
                             -row["review_priority_score"],
                             row["component_id"]))
        for camera_rank, row in enumerate(camera_rows, 1):
            row["camera_review_priority_rank"] = camera_rank
            row["camera_review_priority_band"] = (
                "priority_unknown_high" if row["priority_unknown_high"]
                else "priority_1" if camera_rank <= max(
                    1, int(np.ceil(len(camera_rows) * 0.10)))
                else "priority_2" if camera_rank <= max(
                    1, int(np.ceil(len(camera_rows) * 0.30)))
                else "standard")
    return ranked


def build_ranking(review_queues: list[Path], identity_manifest: Path,
                  component_review_path: Path, output: Path,
                  known_review_receipt: Path | None = None,
                  grayscale_transform: Callable[[np.ndarray], np.ndarray] | None = None,
                  transform_name: str = "none") -> dict:
    if output.exists():
        raise FileExistsError(output)
    originals, failed, identity_sha, queue_bindings = _validated_source_rows(
        review_queues, identity_manifest)
    review_sha = file_sha256(component_review_path)
    review = json.loads(component_review_path.read_text(encoding="utf-8"))
    if (review.get("candidate_only") is not True
            or review.get("human_truth") is not False
            or review.get("approved_for_training") is not False
            or review.get("production_deployed") is not False
            or review.get("default_enabled") is not False
            or review.get("source_masks_modified") is not False
            or review.get("identity_manifest_sha256") != identity_sha):
        raise ValueError("component review safety flags are inconsistent")
    review_rows = review.get("rows", [])
    if (len(review_rows) != int(review.get("component_count", -1))
            or len({row.get("component_id") for row in review_rows}) !=
            len(review_rows)):
        raise ValueError("component review rows are incomplete or duplicated")
    image_cache: dict[str, np.ndarray] = {}
    mask_cache: dict[str, np.ndarray] = {}
    metadata: dict[str, tuple[str, str]] = {}
    for sample_id, source in originals.items():
        parts = sample_id.split("-")
        if len(parts) < 3 or parts[1] not in {"camera1", "camera3"}:
            raise ValueError(f"cannot derive package/camera identity: {sample_id}")
        metadata[sample_id] = (parts[0], parts[1])
        mask_cache[sample_id] = _decode_mask(_verified_bytes(
            source["mask_path"], source["mask_sha256"], "source mask"))
    measured = []
    for row in review_rows:
        sample_id = row.get("sample_id")
        source = originals.get(sample_id)
        if source is None or row.get("source_image_sha256") != source[
                "image_sha256"]:
            raise ValueError(f"component source binding mismatch: {row.get('component_id')}")
        if sample_id not in image_cache:
            decoded = _decode_gray(_verified_bytes(
                source["image_path"], source["image_sha256"], "source image"))
            if grayscale_transform is not None:
                transformed = np.asarray(grayscale_transform(decoded),
                                         dtype=np.float32)
                if (transformed.shape != decoded.shape
                        or not np.isfinite(transformed).all()
                        or transformed.min() < 0 or transformed.max() > 255):
                    raise ValueError("grayscale transform produced invalid data")
                decoded = transformed
            image_cache[sample_id] = decoded
        gray = image_cache[sample_id]
        original = mask_cache[sample_id]
        bbox = row.get("bbox_xyxy")
        if (not isinstance(bbox, list) or len(bbox) != 4
                or not all(isinstance(value, int) for value in bbox)):
            raise ValueError(f"invalid component bbox: {row.get('component_id')}")
        x0, y0, x1, y1 = bbox
        if not (0 <= x0 < x1 <= gray.shape[1]
                and 0 <= y0 < y1 <= gray.shape[0]):
            raise ValueError(f"component bbox is out of bounds: {row.get('component_id')}")
        cropped = _decode_mask(_verified_bytes(
            component_review_path.parent / row.get("component_mask", ""),
            row.get("component_mask_sha256"), "component mask"))
        if (cropped.shape != (y1 - y0, x1 - x0)
                or int(cropped.sum()) != int(row.get("area_pixels", -1))):
            raise ValueError(f"component mask accounting failed: {row.get('component_id')}")
        full = np.zeros_like(gray, dtype=bool)
        full[y0:y1, x0:x1] = cropped
        package, camera = metadata[sample_id]
        region = cv2.dilate(full.astype(np.uint8), np.ones((5, 5), np.uint8),
                            iterations=1).astype(bool)
        external_support = 0
        external_packages = sorted({other_package for other_package, other_camera
                                    in metadata.values()
                                    if other_camera == camera
                                    and other_package != package})
        for external_package in external_packages:
            fractions = [
                float((~mask_cache[other_id])[region].mean())
                for other_id, (other_package, other_camera) in metadata.items()
                if other_package == external_package and other_camera == camera
            ]
            if fractions and float(np.median(fractions)) >= 0.5:
                external_support += 1
        measured.append({
            "component_id": row["component_id"],
            "sample_id": sample_id,
            "area_pixels": int(row["area_pixels"]),
            "bbox_xyxy": bbox,
            "camera": camera,
            "package": package,
            "weak_external_support_packages": external_support,
            "review_image": row.get("review_image"),
            "review_image_sha256": row.get("review_image_sha256"),
            **grayscale_features(gray, full, bbox, original),
        })
    ranked = rank_rows(measured)
    known = None
    if known_review_receipt is not None:
        known_sha = file_sha256(known_review_receipt)
        receipt = json.loads(known_review_receipt.read_text(encoding="utf-8"))
        if (receipt.get("candidate_only") is not True
                or receipt.get("human_truth") is not False
                or receipt.get("approved_for_training") is not False
                or receipt.get("production_deployed") is not False
                or receipt.get("source_component_review_packet_sha256") !=
                review_sha):
            raise ValueError("known review receipt safety binding mismatch")
        by_id = {row["component_id"]: row for row in ranked}
        known_rows = []
        for decision in receipt.get("decisions", []):
            component_id = decision.get("component_id")
            ranked_row = by_id.get(component_id)
            if ranked_row is None:
                raise ValueError(f"known review references unknown component: {component_id}")
            known_rows.append({
                "component_id": component_id,
                "decision": decision.get("decision"),
                "review_priority_rank": ranked_row["review_priority_rank"],
                "review_priority_score": ranked_row["review_priority_score"],
                "review_priority_band": ranked_row["review_priority_band"],
            })
        known = {"receipt_sha256": known_sha, "rows": known_rows}
    packet = {
        "schema_version": "labelagent.ego_component_review_priority.v2",
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "production_deployed": False,
        "default_enabled": False, "automatic_decisions_enabled": False,
        "grayscale_only": True,
        "camera_stratified_percentiles": True,
        "diagnostic_input_transform": transform_name,
        "identity_manifest_sha256": identity_sha,
        "review_queue_bindings": queue_bindings,
        "component_review_sha256": review_sha,
        "component_count": len(ranked),
        "camera_component_counts": {
            camera: sum(row["camera"] == camera for row in ranked)
            for camera in sorted({row["camera"] for row in ranked})
        },
        "failed_closed_input_count": len(failed),
        "priority_1_count": sum(
            row["review_priority_band"] == "priority_1" for row in ranked),
        "priority_2_count": sum(
            row["review_priority_band"] == "priority_2" for row in ranked),
        "known_review_diagnostic": known,
        "rows": ranked,
        "next_step": "Use ranking only to order review. It cannot accept, reject, create truth, or approve training/deployment.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        temporary.write_text(
            json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        readback = json.loads(temporary.read_text(encoding="utf-8"))
        if (readback["component_count"] != len(ranked)
                or any(row.get("automatic_decision") is not None
                       for row in readback["rows"])):
            raise ValueError("risk ranking readback verification failed")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", action="append", type=Path,
                        required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--component-review", type=Path, required=True)
    parser.add_argument("--known-review-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    packet = build_ranking(
        arguments.review_queue, arguments.identity_manifest,
        arguments.component_review, arguments.output,
        arguments.known_review_receipt)
    print(json.dumps({
        "component_count": packet["component_count"],
        "priority_1_count": packet["priority_1_count"],
        "priority_2_count": packet["priority_2_count"],
        "known_review_diagnostic": packet["known_review_diagnostic"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
