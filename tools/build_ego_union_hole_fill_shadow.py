#!/usr/bin/env python3
"""Build reversible component-level ego hole-fill shadows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from tools.build_ego_component_union_review import (
        EXPECTED_COMPONENTS, _verified_bytes, file_sha256, load_component_guard,
        load_identity_manifest)
except ModuleNotFoundError:  # Direct script execution.
    from build_ego_component_union_review import (  # type: ignore
        EXPECTED_COMPONENTS, _verified_bytes, file_sha256, load_component_guard,
        load_identity_manifest)


def _decode_mask(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as decoded:
        return np.asarray(decoded.convert("L")) > 0


def enclosed_hole_labels(mask: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    background = (~mask.astype(bool)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        background, connectivity=8)
    border = set(np.concatenate((labels[0, :], labels[-1, :],
                                 labels[:, 0], labels[:, -1])).tolist())
    holes = {index: int(stats[index, cv2.CC_STAT_AREA])
             for index in range(1, count) if index not in border}
    return labels, holes


def small_hole_mask(mask: np.ndarray, maximum_pixels: int) -> np.ndarray:
    labels, holes = enclosed_hole_labels(mask)
    selected = [label for label, area in holes.items() if area < maximum_pixels]
    return np.isin(labels, selected)


def fill_small_holes(mask: np.ndarray,
                     negative_points: list[tuple[int, int]],
                     maximum_pixels: int = 100,
                     persistent_protection: np.ndarray | None = None
                     ) -> tuple[np.ndarray, dict]:
    if mask.ndim != 2 or maximum_pixels <= 0:
        raise ValueError("mask must be 2D and maximum_pixels must be positive")
    source = mask.astype(bool)
    if not source.any():
        raise ValueError("source mask is empty")
    height, width = source.shape
    for x, y in negative_points:
        if not (0 <= x < width and 0 <= y < height) or source[y, x]:
            raise ValueError("negative point is invalid for the source mask")
    if persistent_protection is not None and persistent_protection.shape != source.shape:
        raise ValueError("persistent protection dimensions differ")
    labels, holes = enclosed_hole_labels(source)
    negative_labels = {int(labels[y, x]) for x, y in negative_points}
    persistent_labels = set()
    if persistent_protection is not None:
        persistent_labels = {
            int(label) for label in np.unique(labels[persistent_protection])
            if int(label) in holes}
    filled_labels = []
    filled_sizes = []
    protected_large = 0
    protected_negative = 0
    protected_persistent = 0
    for label, area in holes.items():
        if area >= maximum_pixels:
            protected_large += 1
        elif label in negative_labels:
            protected_negative += 1
        elif label in persistent_labels:
            protected_persistent += 1
        else:
            filled_labels.append(label)
            filled_sizes.append(area)
    added = np.isin(labels, filled_labels)
    filled = source | added
    if np.logical_and(source, ~filled).any():
        raise ValueError("hole fill removed source foreground")
    if any(filled[y, x] for x, y in negative_points):
        raise ValueError("hole fill violated a negative prompt point")
    return filled, {
        "maximum_hole_pixels_exclusive": maximum_pixels,
        "source_area_pixels": int(source.sum()),
        "filled_area_pixels": int(filled.sum()),
        "added_pixels": int(added.sum()),
        "added_area_ratio_of_mask": round(
            int(added.sum()) / int(source.sum()), 6),
        "enclosed_hole_count": len(holes),
        "filled_hole_count": len(filled_labels),
        "filled_hole_sizes": filled_sizes,
        "large_hole_protected_count": protected_large,
        "negative_point_protected_hole_count": protected_negative,
        "persistent_protected_hole_count": protected_persistent,
        "negative_point_count": len(negative_points),
        "negative_points_preserved": True,
    }


def added_pixels_within_budget(added_pixels: int, source_pixels: int,
                               maximum_ratio: float) -> bool:
    if (type(added_pixels) is not int or type(source_pixels) is not int
            or added_pixels < 0 or source_pixels <= 0
            or not math.isfinite(maximum_ratio)
            or not 0.0 <= maximum_ratio <= 1.0):
        raise ValueError("addition budget settings are invalid")
    return added_pixels <= math.floor(source_pixels * maximum_ratio)


def _write_overlay(image_bytes: bytes, filled: np.ndarray,
                   added: np.ndarray, target: Path) -> None:
    with Image.open(io.BytesIO(image_bytes)) as decoded:
        base = decoded.convert("RGBA")
    if base.size != (filled.shape[1], filled.shape[0]):
        raise ValueError("source image and candidate mask dimensions differ")
    rgba = np.zeros((*filled.shape, 4), dtype=np.uint8)
    rgba[filled] = (24, 206, 138, 95)
    rgba[added] = (225, 70, 235, 230)
    Image.alpha_composite(base, Image.fromarray(rgba, mode="RGBA")).convert(
        "RGB").save(target, "JPEG", quality=92)


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
        draw.text((left + 5, top + image_height + 5), row["sample_id"],
                  fill=(83, 210, 161), font=font)
        draw.text((left + 5, top + image_height + 22),
                  (f"added {row['metrics']['added_pixels']} px / "
                   f"{row['metrics']['filled_hole_count']} holes / "
                   f"{row['metrics']['added_area_ratio_of_mask']:.4f}"),
                  fill=(83, 210, 161), font=font)
    target = output / "hole_fill_shadow_contact_sheet.jpg"
    canvas.save(target, "JPEG", quality=90)
    return target


def build_shadow(review_queues: list[Path], diagnostic_roots: list[Path],
                 identity_manifest_path: Path, component_guard_path: Path,
                 output: Path, maximum_hole_pixels: int = 100,
                 maximum_added_ratio: float = 0.01,
                 expected_size: tuple[int, int] = (640, 512),
                 persistence_min_external_packages: int = 2,
                 persistence_radius_pixels: int = 2,
                 require_persistence_context: bool = True) -> dict:
    if output.exists():
        raise FileExistsError(output)
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
    samples, failed_closed_inputs, seen = [], [], set()
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
            raise ValueError("review queue was not produced by strict safety gates")
        allowed_root = Path(gates["allowed_source_root"]).resolve(strict=True)
        for row in packet.get("samples", []):
            sample_id = row.get("sample_id")
            if sample_id in seen:
                raise ValueError(f"duplicate sample identity: {sample_id}")
            seen.add(sample_id)
            if row.get("union_available") is not True:
                reasons = row.get("reasons")
                if not isinstance(reasons, list) or not reasons:
                    raise ValueError(f"failed-closed input lacks reasons: {sample_id}")
                failed_closed_inputs.append({"sample_id": sample_id,
                                             "reasons": reasons})
                continue
            identity = identities.get(sample_id)
            if identity is None or identity["image_sha256"] != row.get(
                    "source_image_sha256"):
                raise ValueError(f"identity binding mismatch: {sample_id}")
            image_path = Path(row.get("source_image", ""))
            image_resolved = image_path.resolve(strict=True)
            try:
                image_resolved.relative_to(allowed_root)
            except ValueError as error:
                raise ValueError(f"source image is outside allowed root: {sample_id}") from error
            if identity.get("relative_path") is not None:
                if image_resolved != (allowed_root / identity["relative_path"]).resolve(
                        strict=True):
                    raise ValueError(f"source relative path mismatch: {sample_id}")
            elif (image_resolved.name != f"{identity['frame_id']}.png"
                  or image_resolved.parent.name != f"camera_{identity['camera'][-1]}"
                  or image_resolved.parent.parent.name != identity["package"]):
                raise ValueError(f"source nested path mismatch: {sample_id}")
            diagnostic_dir = diagnostic_dirs.get(sample_id)
            if diagnostic_dir is None:
                raise ValueError(f"missing diagnostic result: {sample_id}")
            report_path = diagnostic_dir / "results.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (report.get("candidate_only") is not True
                    or report.get("human_truth") is not False
                    or report.get("approved_for_training") is not False
                    or report.get("production_deployed") is not False
                    or report.get("source_state_modified") is not False
                    or report.get("image_sha256") != row.get("source_image_sha256")):
                raise ValueError(f"diagnostic binding mismatch: {sample_id}")
            components = []
            for component in report.get("rows", []):
                if (component.get("candidate_available") is not True
                        or component.get("prompt_constraints_passed") is not True):
                    raise ValueError(f"diagnostic component failed closed: {sample_id}")
                name = component.get("name")
                if name not in component_guard:
                    raise ValueError(f"component guard missing: {sample_id}/{name}")
                mask = _decode_mask(_verified_bytes(
                    diagnostic_dir / component.get("candidate_mask", ""),
                    component.get("candidate_mask_sha256"), "component mask"))
                if (mask.shape[1], mask.shape[0]) != expected_size:
                    raise ValueError(f"unexpected component mask dimensions: {sample_id}")
                positive = [(int(point["x"]), int(point["y"]))
                            for point in component.get("points", [])
                            if point.get("label") == 1]
                negative = [(int(point["x"]), int(point["y"]))
                            for point in component.get("points", [])
                            if point.get("label") == 0]
                height, width = mask.shape
                if any(not (0 <= x < width and 0 <= y < height)
                       for x, y in positive + negative):
                    raise ValueError(f"prompt point is out of bounds: {sample_id}/{name}")
                if not positive or any(not mask[y, x] for x, y in positive):
                    raise ValueError(f"source component misses a positive point: {sample_id}/{name}")
                if any(mask[y, x] for x, y in negative):
                    raise ValueError(f"source component violates a negative point: {sample_id}/{name}")
                components.append({"name": name, "mask": mask,
                                   "positive_points": positive,
                                   "negative_points": negative,
                                   "source_mask_sha256": component["candidate_mask_sha256"]})
            expected_names = set(EXPECTED_COMPONENTS[identity["camera"]])
            observed_names = {component["name"] for component in components}
            if observed_names != expected_names or len(components) != len(expected_names):
                raise ValueError(f"component set is incomplete: {sample_id}")
            queue_dir = queue_path.parent
            source_union = _decode_mask(_verified_bytes(
                queue_dir / row.get("union_mask", ""),
                row.get("union_mask_sha256"), "union mask"))
            image_bytes = _verified_bytes(image_resolved,
                                          row.get("source_image_sha256"),
                                          "source image")
            if (source_union.shape[1], source_union.shape[0]) != expected_size:
                raise ValueError(f"unexpected union mask dimensions: {sample_id}")
            if not np.array_equal(np.logical_or.reduce(
                    [component["mask"] for component in components]), source_union):
                raise ValueError(f"component masks do not reconstruct union: {sample_id}")
            samples.append({"sample_id": sample_id,
                            "package": identity["package"],
                            "camera": identity["camera"],
                            "mask": source_union, "image_bytes": image_bytes,
                            "components": components,
                            "source_mask_sha256": row["union_mask_sha256"],
                            "source_image_sha256": row["source_image_sha256"],
                            "source_queue_sha256": file_sha256(queue_path),
                            "diagnostic_sha256": file_sha256(report_path)})
    if seen != set(identities):
        raise ValueError("review queues and identity manifest differ")
    if ((require_persistence_context and persistence_min_external_packages < 2)
            or persistence_radius_pixels < 0):
        raise ValueError("persistence settings are invalid")
    package_holes = {}
    for sample in samples:
        for component in sample["components"]:
            key = (sample["camera"], component["name"], sample["package"])
            holes = small_hole_mask(component["mask"], maximum_hole_pixels)
            package_holes[key] = holes if key not in package_holes else (
                package_holes[key] | holes)
    kernel = np.ones((persistence_radius_pixels * 2 + 1,) * 2, dtype=np.uint8)
    rows, pending_outputs = [], []
    for sample in sorted(samples, key=lambda item: item["sample_id"]):
        filled_components, component_rows = [], []
        for component in sample["components"]:
            external = []
            for (camera, name, package), hole_mask in package_holes.items():
                if (camera == sample["camera"] and name == component["name"]
                        and package != sample["package"]):
                    external.append(cv2.dilate(hole_mask.astype(np.uint8), kernel,
                                               iterations=1) > 0)
            if require_persistence_context and len(external) < persistence_min_external_packages:
                raise ValueError("insufficient external packages for hole persistence")
            protection = (np.sum(np.stack(external), axis=0) >=
                          persistence_min_external_packages
                          if require_persistence_context else
                          np.zeros_like(component["mask"]))
            filled, metrics = fill_small_holes(
                component["mask"], component["negative_points"],
                maximum_hole_pixels, protection)
            if not added_pixels_within_budget(
                    metrics["added_pixels"], metrics["source_area_pixels"],
                    maximum_added_ratio):
                raise ValueError(
                    f"component addition budget exceeded: {sample['sample_id']}/"
                    f"{component['name']} observed="
                    f"{metrics['added_area_ratio_of_mask']:.6f} limit="
                    f"{maximum_added_ratio:.6f} added_pixels="
                    f"{metrics['added_pixels']}")
            if any(filled[y, x] for x, y in component["negative_points"]):
                raise ValueError(f"hole fill violates a negative point: {sample['sample_id']}/{component['name']}")
            area_ratio = round(float(filled.mean()), 6)
            guard = component_guard[component["name"]]
            if not guard["min_area_ratio"] <= area_ratio <= guard["max_area_ratio"]:
                raise ValueError(f"filled component area outside guard: {sample['sample_id']}/{component['name']}")
            rerun, _ = fill_small_holes(
                filled, component["negative_points"], maximum_hole_pixels,
                protection)
            if not np.array_equal(rerun, filled):
                raise ValueError(f"hole fill is not idempotent: {sample['sample_id']}/{component['name']}")
            filled_components.append(filled)
            component_rows.append({"name": component["name"],
                                   "source_mask_sha256": component["source_mask_sha256"],
                                   "area_ratio_after": area_ratio,
                                   "area_guard": guard,
                                   "positive_points_preserved": True,
                                   "idempotent": True, **metrics})
        filled_union = np.logical_or.reduce(filled_components)
        if np.logical_and(sample["mask"], ~filled_union).any():
            raise ValueError(f"hole fill removed union pixels: {sample['sample_id']}")
        added = filled_union & ~sample["mask"]
        metrics = {
            "source_area_pixels": int(sample["mask"].sum()),
            "filled_area_pixels": int(filled_union.sum()),
            "added_pixels": int(added.sum()),
            "added_area_ratio_of_mask": round(
                int(added.sum()) / int(sample["mask"].sum()), 6),
            "filled_hole_count": sum(row["filled_hole_count"] for row in component_rows),
            "removed_pixels": 0,
            "positive_points_preserved": True,
            "negative_points_preserved": True,
            "component_budget_passed": True,
            "component_area_guards_passed": True,
            "idempotent": True,
        }
        mask_name = f"{sample['sample_id']}-hole-fill-shadow-mask.png"
        overlay_name = f"{sample['sample_id']}-hole-fill-shadow-overlay.jpg"
        row = {
            "sample_id": sample["sample_id"], "candidate_only": True,
            "human_truth": False, "approved_for_training": False,
            "production_deployed": False,
            "review_status": "assistant_review_required",
            "source_image_sha256": sample["source_image_sha256"],
            "source_union_mask_sha256": sample["source_mask_sha256"],
            "source_queue_sha256": sample["source_queue_sha256"],
            "diagnostic_sha256": sample["diagnostic_sha256"],
            "components": component_rows, "metrics": metrics,
            "mask": mask_name, "overlay": overlay_name,
        }
        rows.append(row)
        pending_outputs.append((row, filled_union, added, sample["image_bytes"]))
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        for row, filled_union, added, image_bytes in pending_outputs:
            Image.fromarray(filled_union.astype(np.uint8) * 255, mode="L").save(
                staging / row["mask"], "PNG", optimize=True)
            _write_overlay(image_bytes, filled_union, added,
                           staging / row["overlay"])
            row["mask_sha256"] = file_sha256(staging / row["mask"])
            row["overlay_sha256"] = file_sha256(staging / row["overlay"])
        contact = _write_contact_sheet(staging, rows)
        packet = {
        "schema_version": "labelagent.ego_union_hole_fill_shadow.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "production_deployed": False,
        "formal_state_modified": False, "source_masks_modified": False,
        "sample_count": len(rows),
        "failed_closed_input_count": len(failed_closed_inputs),
        "failed_closed_inputs": failed_closed_inputs,
        "maximum_hole_pixels_exclusive": maximum_hole_pixels,
        "maximum_added_ratio_per_component": maximum_added_ratio,
        "connectivity": 8, "expected_size": list(expected_size),
        "persistent_hole_protection": {
            "enabled": require_persistence_context,
            "minimum_external_packages": persistence_min_external_packages,
            "dilation_radius_pixels": persistence_radius_pixels,
        },
        "added_pixel_total": sum(row["metrics"]["added_pixels"] for row in rows),
        "filled_hole_total": sum(row["metrics"]["filled_hole_count"] for row in rows),
        "removed_pixel_total": 0,
        "positive_points_preserved_all": True,
        "negative_points_preserved_all": True,
        "identity_manifest_sha256": identity_sha,
        "component_guard_sha256": component_guard_sha,
        "contact_sheet": contact.name,
        "contact_sheet_sha256": file_sha256(contact),
        "samples": rows,
        "next_step": "Review every magenta added region; never replace source masks or approve training from this shadow packet.",
        }
        target = staging / "hole_fill_shadow.json"
        target.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        json.loads(target.read_text(encoding="utf-8"))
        staging.replace(output)
    except Exception:
        if staging.exists() and staging.parent == output.parent:
            shutil.rmtree(staging)
        raise
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", action="append", type=Path, required=True)
    parser.add_argument("--diagnostic-root", action="append", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--component-guard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-hole-pixels", type=int, default=100)
    arguments = parser.parse_args()
    packet = build_shadow(
        arguments.review_queue, arguments.diagnostic_root,
        arguments.identity_manifest, arguments.component_guard,
        arguments.output, arguments.maximum_hole_pixels)
    print(json.dumps({
        "output": str((arguments.output / "hole_fill_shadow.json").resolve()),
        "sample_count": packet["sample_count"],
        "added_pixel_total": packet["added_pixel_total"],
        "filled_hole_total": packet["filled_hole_total"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
