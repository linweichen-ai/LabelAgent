#!/usr/bin/env python3
"""Build reversible ego-union denoise shadows without touching source masks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from tools.build_ego_component_union_review import (
        _verified_bytes, file_sha256, load_component_guard,
        load_identity_manifest)
except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
    from build_ego_component_union_review import (  # type: ignore
        _verified_bytes, file_sha256, load_component_guard,
        load_identity_manifest)


def denoise_small_islands(mask: np.ndarray,
                          positive_points: list[tuple[int, int]],
                          minimum_pixels: int = 100,
                          persistent_protection: np.ndarray | None = None
                          ) -> tuple[np.ndarray, dict]:
    if mask.ndim != 2 or minimum_pixels <= 0:
        raise ValueError("mask must be 2D and minimum_pixels must be positive")
    mask = mask.astype(bool)
    if not mask.any():
        raise ValueError("source mask is empty")
    height, width = mask.shape
    for x, y in positive_points:
        if not (0 <= x < width and 0 <= y < height) or not mask[y, x]:
            raise ValueError("positive point is outside the source mask")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    protected_labels = {int(labels[y, x]) for x, y in positive_points}
    persistent_labels = set()
    if persistent_protection is not None:
        if persistent_protection.shape != mask.shape:
            raise ValueError("persistent protection dimensions differ")
        persistent_labels = {
            int(index) for index in np.unique(labels[persistent_protection & mask])
            if int(index) != 0}
        protected_labels.update(persistent_labels)
    keep_labels = set()
    removed_components = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= minimum_pixels or index in protected_labels:
            keep_labels.add(index)
        else:
            removed_components.append(area)
    persistent_small_count = sum(
        int(stats[index, cv2.CC_STAT_AREA]) < minimum_pixels
        for index in persistent_labels)
    cleaned = np.isin(labels, list(keep_labels))
    removed_pixels = int(np.logical_and(mask, ~cleaned).sum())
    if any(not cleaned[y, x] for x, y in positive_points):
        raise ValueError("denoise removed a positive prompt point")
    return cleaned, {
        "minimum_pixels": minimum_pixels,
        "source_area_pixels": int(mask.sum()),
        "cleaned_area_pixels": int(cleaned.sum()),
        "removed_pixels": removed_pixels,
        "removed_area_ratio_of_mask": round(removed_pixels / int(mask.sum()), 6),
        "removed_component_count": len(removed_components),
        "removed_component_sizes": removed_components,
        "protected_component_count": len(protected_labels),
        "persistent_protected_component_count": persistent_small_count,
        "positive_point_count": len(positive_points),
        "positive_points_preserved": True,
    }


def _decode_mask(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as decoded:
        return np.asarray(decoded.convert("L")) > 0


def _write_overlay(image_bytes: bytes, cleaned: np.ndarray,
                   removed: np.ndarray, target: Path) -> None:
    with Image.open(io.BytesIO(image_bytes)) as decoded:
        base = decoded.convert("RGBA")
    if base.size != (cleaned.shape[1], cleaned.shape[0]):
        raise ValueError("source image and candidate mask dimensions differ")
    rgba = np.zeros((*cleaned.shape, 4), dtype=np.uint8)
    rgba[cleaned] = (24, 206, 138, 100)
    rgba[removed] = (225, 70, 235, 220)
    rendered = Image.alpha_composite(base, Image.fromarray(rgba, mode="RGBA"))
    rendered.convert("RGB").save(target, "JPEG", quality=92)


def _write_contact_sheet(output: Path, rows: list[dict]) -> Path:
    cell_width, image_height, label_height, columns = 320, 256, 42, 4
    row_count = (len(rows) + columns - 1) // columns
    canvas = Image.new("RGB", (cell_width * columns,
                               (image_height + label_height) * row_count),
                       (15, 23, 31))
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    for index, row in enumerate(rows):
        left = index % columns * cell_width
        top = index // columns * (image_height + label_height)
        with Image.open(output / row["overlay"]) as image:
            preview = image.convert("RGB")
            preview.thumbnail((cell_width, image_height))
        canvas.paste(preview, (left + (cell_width - preview.width) // 2,
                               top + (image_height - preview.height) // 2))
        draw.rectangle((left, top + image_height,
                        left + cell_width, top + image_height + label_height),
                       fill=(25, 35, 45))
        color = (83, 210, 161)
        draw.text((left + 5, top + image_height + 5), row["sample_id"],
                  fill=color, font=font)
        draw.text((left + 5, top + image_height + 22),
                  (f"removed {row['metrics']['removed_pixels']} px / "
                   f"{row['metrics']['removed_component_count']} islands / "
                   f"{row['metrics']['removed_area_ratio_of_mask']:.4f}"),
                  fill=color, font=font)
    path = output / "denoise_shadow_contact_sheet.jpg"
    canvas.save(path, "JPEG", quality=90)
    return path


def build_shadow(review_queues: list[Path], diagnostic_roots: list[Path],
                 identity_manifest_path: Path, component_guard_path: Path,
                 output: Path, minimum_pixels: int = 100,
                 maximum_removed_ratio: float = 0.01,
                 expected_size: tuple[int, int] = (640, 512),
                 temporal_min_packages: int = 2,
                 temporal_radius_pixels: int = 2,
                 require_temporal_context: bool = True) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if not review_queues or not diagnostic_roots:
        raise ValueError("review queues and diagnostic roots are required")
    identities = load_identity_manifest(identity_manifest_path)
    identity_sha = file_sha256(identity_manifest_path)
    component_guard = load_component_guard(component_guard_path)
    component_guard_sha = file_sha256(component_guard_path)
    diagnostic_dirs = {}
    for root in diagnostic_roots:
        for path in root.iterdir():
            if path.is_dir() and (path / "results.json").is_file():
                if path.name in diagnostic_dirs:
                    raise ValueError(f"duplicate diagnostic sample: {path.name}")
                diagnostic_dirs[path.name] = path
    samples = []
    seen = set()
    for queue_path in review_queues:
        packet = json.loads(queue_path.read_text(encoding="utf-8"))
        gates = packet.get("safety_gates") or {}
        if (packet.get("candidate_only") is not True
                or packet.get("human_truth") is not False
                or packet.get("approved_for_training") is not False
                or packet.get("production_deployed") is not False
                or not gates.get("allowed_source_root")
                or gates.get("component_area_guard_enabled") is not True
                or gates.get("identity_manifest_enabled") is not True
                or gates.get("identity_manifest_sha256") != identity_sha):
            raise ValueError("review queue was not produced by all strict safety gates")
        for row in packet.get("samples", []):
            sample_id = row.get("sample_id")
            if sample_id in seen:
                raise ValueError(f"duplicate sample identity: {sample_id}")
            seen.add(sample_id)
            identity = identities.get(sample_id)
            if identity is None or identity["image_sha256"] != row.get("source_image_sha256"):
                raise ValueError(f"identity binding mismatch: {sample_id}")
            diagnostic_dir = diagnostic_dirs.get(sample_id)
            if diagnostic_dir is None:
                raise ValueError(f"missing diagnostic result: {sample_id}")
            report = json.loads((diagnostic_dir / "results.json").read_text(encoding="utf-8"))
            if (report.get("candidate_only") is not True
                    or report.get("human_truth") is not False
                    or report.get("approved_for_training") is not False
                    or report.get("production_deployed") is not False
                    or report.get("source_state_modified") is not False
                    or report.get("image_sha256") != row.get("source_image_sha256")):
                raise ValueError(f"diagnostic safety or image binding mismatch: {sample_id}")
            components = []
            for component in report.get("rows", []):
                if (component.get("candidate_available") is not True
                        or component.get("prompt_constraints_passed") is not True):
                    raise ValueError(f"diagnostic component failed closed: {sample_id}")
                name = component.get("name")
                if name not in component_guard:
                    raise ValueError(f"component guard missing: {sample_id}/{name}")
                component_mask_bytes = _verified_bytes(
                    diagnostic_dir / component.get("candidate_mask", ""),
                    component.get("candidate_mask_sha256"), "component mask")
                component_mask = _decode_mask(component_mask_bytes)
                if (component_mask.shape[1], component_mask.shape[0]) != expected_size:
                    raise ValueError(f"unexpected component mask dimensions: {sample_id}")
                positive_points = [
                    (int(point["x"]), int(point["y"]))
                    for point in component.get("points", [])
                    if point.get("label") == 1]
                negative_points = [
                    (int(point["x"]), int(point["y"]))
                    for point in component.get("points", [])
                    if point.get("label") == 0]
                if not positive_points:
                    raise ValueError(f"positive prompts are missing: {sample_id}/{name}")
                if any(not component_mask[y, x] for x, y in positive_points):
                    raise ValueError(f"source component misses a positive point: {sample_id}/{name}")
                if any(component_mask[y, x] for x, y in negative_points):
                    raise ValueError(f"source component violates a negative point: {sample_id}/{name}")
                components.append({
                    "name": name,
                    "mask": component_mask,
                    "positive_points": positive_points,
                    "negative_points": negative_points,
                    "source_mask_sha256": component["candidate_mask_sha256"],
                })
            queue_dir = queue_path.parent
            mask_bytes = _verified_bytes(
                queue_dir / row.get("union_mask", ""),
                row.get("union_mask_sha256"), "union mask")
            image_bytes = _verified_bytes(
                Path(row.get("source_image", "")),
                row.get("source_image_sha256"), "source image")
            source_union = _decode_mask(mask_bytes)
            if (source_union.shape[1], source_union.shape[0]) != expected_size:
                raise ValueError(f"unexpected union mask dimensions: {sample_id}")
            reconstructed = np.logical_or.reduce(
                [component["mask"] for component in components])
            if not np.array_equal(reconstructed, source_union):
                raise ValueError(f"component masks do not reconstruct union: {sample_id}")
            samples.append({
                "sample_id": sample_id,
                "package": identity["package"],
                "camera": identity["camera"],
                "mask": source_union,
                "image_bytes": image_bytes,
                "components": components,
                "source_mask_sha256": row["union_mask_sha256"],
                "source_image_sha256": row["source_image_sha256"],
                "source_queue_sha256": file_sha256(queue_path),
                "diagnostic_sha256": file_sha256(diagnostic_dir / "results.json"),
            })
    packages = {sample["package"] for sample in samples}
    if require_temporal_context and len(packages) < temporal_min_packages:
        raise ValueError("insufficient packages for temporal persistence protection")
    if temporal_min_packages < 1 or temporal_radius_pixels < 0:
        raise ValueError("temporal persistence settings are invalid")
    kernel_size = temporal_radius_pixels * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    package_component_masks = {}
    for sample in samples:
        for component in sample["components"]:
            key = (sample["camera"], component["name"], sample["package"])
            existing = package_component_masks.get(key)
            package_component_masks[key] = (component["mask"].copy()
                                             if existing is None
                                             else np.logical_or(existing,
                                                                component["mask"]))
    temporal_support = {}
    group_keys = {(camera, name) for camera, name, _ in package_component_masks}
    for camera, name in group_keys:
        group_masks = []
        for package in sorted(packages):
            package_mask = package_component_masks.get((camera, name, package))
            if package_mask is None:
                continue
            expanded = cv2.dilate(
                package_mask.astype(np.uint8), kernel, iterations=1) > 0
            group_masks.append(expanded)
        if len(group_masks) >= temporal_min_packages:
            votes = np.sum(np.stack(group_masks, axis=0), axis=0)
            temporal_support[(camera, name)] = votes >= temporal_min_packages
        else:
            temporal_support[(camera, name)] = np.zeros(
                (expected_size[1], expected_size[0]), dtype=bool)
    output.mkdir(parents=True)
    rows = []
    for sample in sorted(samples, key=lambda item: item["sample_id"]):
        cleaned_components = []
        component_rows = []
        for component in sample["components"]:
            cleaned, metrics = denoise_small_islands(
                component["mask"], component["positive_points"], minimum_pixels,
                temporal_support[(sample["camera"], component["name"])])
            if metrics["removed_area_ratio_of_mask"] > maximum_removed_ratio:
                raise ValueError(
                    f"component removal budget exceeded: {sample['sample_id']}/"
                    f"{component['name']}")
            if any(cleaned[y, x] for x, y in component["negative_points"]):
                raise ValueError(
                    f"denoise violates a negative point: {sample['sample_id']}/"
                    f"{component['name']}")
            area_ratio = round(float(cleaned.mean()), 6)
            guard = component_guard[component["name"]]
            if not guard["min_area_ratio"] <= area_ratio <= guard["max_area_ratio"]:
                raise ValueError(
                    f"denoised component area outside guard: {sample['sample_id']}/"
                    f"{component['name']}")
            rerun, _ = denoise_small_islands(
                cleaned, component["positive_points"], minimum_pixels,
                temporal_support[(sample["camera"], component["name"])])
            if not np.array_equal(rerun, cleaned):
                raise ValueError(
                    f"denoise is not idempotent: {sample['sample_id']}/"
                    f"{component['name']}")
            cleaned_components.append(cleaned)
            component_rows.append({
                "name": component["name"],
                "source_mask_sha256": component["source_mask_sha256"],
                "area_ratio_after": area_ratio,
                "area_guard": guard,
                "negative_point_count": len(component["negative_points"]),
                "negative_points_preserved": True,
                "idempotent": True,
                **metrics,
            })
        cleaned = np.logical_or.reduce(cleaned_components)
        if np.logical_and(cleaned, ~sample["mask"]).any():
            raise ValueError(f"denoise added pixels: {sample['sample_id']}")
        removed = sample["mask"] & ~cleaned
        metrics = {
            "minimum_pixels": minimum_pixels,
            "source_area_pixels": int(sample["mask"].sum()),
            "cleaned_area_pixels": int(cleaned.sum()),
            "removed_pixels": int(removed.sum()),
            "removed_area_ratio_of_mask": round(
                float(removed.sum()) / int(sample["mask"].sum()), 6),
            "removed_component_count": sum(
                component["removed_component_count"] for component in component_rows),
            "added_pixels": 0,
            "positive_points_preserved": True,
            "negative_points_preserved": True,
            "component_budget_passed": True,
            "component_area_guards_passed": True,
            "idempotent": True,
        }
        mask_name = f"{sample['sample_id']}-denoise-shadow-mask.png"
        Image.fromarray(cleaned.astype(np.uint8) * 255, mode="L").save(
            output / mask_name, "PNG", optimize=True)
        overlay_name = f"{sample['sample_id']}-denoise-shadow-overlay.jpg"
        _write_overlay(sample["image_bytes"], cleaned, removed,
                       output / overlay_name)
        rows.append({
            "sample_id": sample["sample_id"],
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "production_deployed": False,
            "review_status": "assistant_review_required",
            "source_image_sha256": sample["source_image_sha256"],
            "source_union_mask_sha256": sample["source_mask_sha256"],
            "source_queue_sha256": sample["source_queue_sha256"],
            "diagnostic_sha256": sample["diagnostic_sha256"],
            "components": component_rows,
            "metrics": metrics,
            "mask": mask_name,
            "mask_sha256": file_sha256(output / mask_name),
            "overlay": overlay_name,
            "overlay_sha256": file_sha256(output / overlay_name),
        })
    contact = _write_contact_sheet(output, rows)
    packet = {
        "schema_version": "labelagent.ego_union_denoise_shadow.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "production_deployed": False,
        "formal_state_modified": False,
        "source_masks_modified": False,
        "sample_count": len(rows),
        "minimum_pixels": minimum_pixels,
        "connectivity": 8,
        "expected_size": list(expected_size),
        "maximum_removed_ratio_per_component": maximum_removed_ratio,
        "temporal_persistence": {
            "enabled": require_temporal_context,
            "minimum_distinct_packages": temporal_min_packages,
            "dilation_radius_pixels": temporal_radius_pixels,
        },
        "removed_pixel_total": sum(row["metrics"]["removed_pixels"] for row in rows),
        "removed_component_total": sum(
            row["metrics"]["removed_component_count"] for row in rows),
        "positive_points_preserved_all": all(
            row["metrics"]["positive_points_preserved"] for row in rows),
        "negative_points_preserved_all": all(
            row["metrics"]["negative_points_preserved"] for row in rows),
        "added_pixel_total": sum(row["metrics"]["added_pixels"] for row in rows),
        "identity_manifest_sha256": identity_sha,
        "component_guard_sha256": component_guard_sha,
        "contact_sheet": contact.name,
        "contact_sheet_sha256": file_sha256(contact),
        "samples": rows,
        "next_step": (
            "Review magenta removed islands pixel by pixel. Do not replace source masks "
            "or approve training from this shadow packet."),
    }
    target = output / "denoise_shadow.json"
    target.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", action="append", type=Path, required=True)
    parser.add_argument("--diagnostic-root", action="append", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--component-guard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-pixels", type=int, default=100)
    parser.add_argument("--temporal-min-packages", type=int, default=2)
    parser.add_argument("--temporal-radius-pixels", type=int, default=2)
    arguments = parser.parse_args()
    packet = build_shadow(
        arguments.review_queue, arguments.diagnostic_root,
        arguments.identity_manifest, arguments.component_guard,
        arguments.output, arguments.minimum_pixels,
        temporal_min_packages=arguments.temporal_min_packages,
        temporal_radius_pixels=arguments.temporal_radius_pixels)
    print(json.dumps({
        "output": str((arguments.output / "denoise_shadow.json").resolve()),
        "sample_count": packet["sample_count"],
        "removed_pixel_total": packet["removed_pixel_total"],
        "removed_component_total": packet["removed_component_total"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
