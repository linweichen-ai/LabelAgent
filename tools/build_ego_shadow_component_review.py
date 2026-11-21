#!/usr/bin/env python3
"""Build per-connected-component, three-scale review pages for ego shadows."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from tools.build_ego_component_union_review import (
        _verified_bytes, file_sha256, load_identity_manifest)
    from tools.build_ego_shadow_zoom_review import added_region_bbox
except ModuleNotFoundError:  # Direct script execution.
    from build_ego_component_union_review import (  # type: ignore
        _verified_bytes, file_sha256, load_identity_manifest)
    from build_ego_shadow_zoom_review import added_region_bbox  # type: ignore


def _decode_mask(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as decoded:
        return np.asarray(decoded.convert("L")) > 0


def _component_digest(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    x0, y0, x1, y1 = bbox
    cropped = np.ascontiguousarray(mask[y0:y1, x0:x1], dtype=np.uint8)
    payload = f"{cropped.shape[0]}x{cropped.shape[1]}:".encode("ascii")
    return hashlib.sha256(payload + cropped.tobytes()).hexdigest()


def extract_added_components(added: np.ndarray,
                             maximum_pixels_exclusive: int) -> list[dict]:
    """Return a deterministic, complete 8-connected partition of added."""
    if (added.ndim != 2 or maximum_pixels_exclusive <= 1
            or added.dtype != np.bool_):
        raise ValueError("invalid component extraction settings")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        added.astype(np.uint8), connectivity=8)
    components = []
    reconstructed = np.zeros_like(added)
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        component = labels == label
        bbox = (x, y, x + width, y + height)
        if area != int(component.sum()) or area <= 0:
            raise ValueError("component area accounting failed")
        if area >= maximum_pixels_exclusive:
            raise ValueError(
                f"added component exceeds frozen size guard: {area}")
        if np.logical_and(reconstructed, component).any():
            raise ValueError("added components overlap")
        reconstructed |= component
        components.append({
            "mask": component,
            "bbox": bbox,
            "area": area,
            "digest": _component_digest(component, bbox),
        })
    if not np.array_equal(reconstructed, added):
        raise ValueError("added components do not reconstruct the added mask")
    components.sort(key=lambda row: (
        row["bbox"][1], row["bbox"][0], row["bbox"][3], row["bbox"][2],
        row["area"], row["digest"]))
    return components


def _validated_source_rows(review_queues: list[Path], identity_manifest: Path
                           ) -> tuple[dict, set[str], str, list[dict]]:
    identities = load_identity_manifest(identity_manifest)
    identity_sha = file_sha256(identity_manifest)
    originals: dict[str, dict] = {}
    failed: set[str] = set()
    queue_bindings = []
    for queue_path in review_queues:
        queue_sha = file_sha256(queue_path)
        packet = json.loads(queue_path.read_text(encoding="utf-8"))
        gates = packet.get("safety_gates") or {}
        if (packet.get("candidate_only") is not True
                or packet.get("human_truth") is not False
                or packet.get("approved_for_training") is not False
                or packet.get("production_deployed") is not False
                or gates.get("identity_manifest_sha256") != identity_sha):
            raise ValueError("review queue safety flags are inconsistent")
        allowed_root = Path(gates["allowed_source_root"]).resolve(strict=True)
        queue_bindings.append({"sha256": queue_sha,
                               "sample_count": len(packet.get("samples", []))})
        for row in packet.get("samples", []):
            sample_id = row.get("sample_id")
            if not sample_id or sample_id in originals or sample_id in failed:
                raise ValueError(f"duplicate or missing sample identity: {sample_id}")
            if row.get("union_available") is not True:
                failed.add(sample_id)
                continue
            identity = identities.get(sample_id)
            if identity is None or identity["image_sha256"] != row.get(
                    "source_image_sha256"):
                raise ValueError(f"identity binding mismatch: {sample_id}")
            image_path = Path(row.get("source_image", "")).resolve(strict=True)
            try:
                image_path.relative_to(allowed_root)
            except ValueError as error:
                raise ValueError(
                    f"source image is outside allowed root: {sample_id}") from error
            if identity.get("relative_path") is not None:
                expected = (allowed_root / identity["relative_path"]).resolve(
                    strict=True)
                if image_path != expected:
                    raise ValueError(f"source relative path mismatch: {sample_id}")
            elif (image_path.name != f"{identity['frame_id']}.png"
                  or image_path.parent.name != f"camera_{identity['camera'][-1]}"
                  or image_path.parent.parent.name != identity["package"]):
                raise ValueError(f"source nested path mismatch: {sample_id}")
            originals[sample_id] = {
                "image_path": image_path,
                "image_sha256": row.get("source_image_sha256"),
                "mask_path": queue_path.parent / row.get("union_mask", ""),
                "mask_sha256": row.get("union_mask_sha256"),
                "queue_sha256": queue_sha,
            }
    return originals, failed, identity_sha, queue_bindings


def _overlay(base: Image.Image, original: np.ndarray, all_added: np.ndarray,
             target: np.ndarray) -> Image.Image:
    rgba = np.zeros((*original.shape, 4), dtype=np.uint8)
    rgba[original] = (24, 206, 138, 58)
    rgba[all_added] = (250, 190, 40, 105)
    rgba[target] = (245, 40, 220, 255)
    return Image.alpha_composite(base.convert("RGBA"),
                                 Image.fromarray(rgba, mode="RGBA"))


def _fit(image: Image.Image, size: tuple[int, int], nearest: bool = False
         ) -> Image.Image:
    copy = image.convert("RGB")
    if nearest and copy.width <= size[0] and copy.height <= size[1]:
        integer_scale = max(1, min(size[0] // max(copy.width, 1),
                                   size[1] // max(copy.height, 1)))
        copy = copy.resize((copy.width * integer_scale,
                            copy.height * integer_scale),
                           Image.Resampling.NEAREST)
    else:
        copy.thumbnail(size, Image.Resampling.NEAREST if nearest
                       else Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (13, 20, 28))
    canvas.paste(copy, ((size[0] - copy.width) // 2,
                        (size[1] - copy.height) // 2))
    return canvas


def expanded_local_box(bbox: tuple[int, int, int, int],
                       crop_bbox: tuple[int, int, int, int], gap: int = 3
                       ) -> tuple[int, int, int, int]:
    """Return an outline box outside the target pixels where space permits."""
    if gap < 1:
        raise ValueError("outline gap must be positive")
    x0, y0, x1, y1 = bbox
    cx0, cy0, cx1, cy1 = crop_bbox
    if not (cx0 <= x0 < x1 <= cx1 and cy0 <= y0 < y1 <= cy1):
        raise ValueError("target bbox is outside crop bbox")
    return (max(0, x0 - cx0 - gap), max(0, y0 - cy0 - gap),
            min(cx1 - cx0 - 1, x1 - cx0 - 1 + gap),
            min(cy1 - cy0 - 1, y1 - cy0 - 1 + gap))


def _render_triptych(image_bytes: bytes, original: np.ndarray,
                     all_added: np.ndarray, target: np.ndarray,
                     bbox: tuple[int, int, int, int], component_id: str,
                     area: int, destination: Path) -> None:
    with Image.open(io.BytesIO(image_bytes)) as decoded:
        base = decoded.convert("RGBA")
    if base.size != (original.shape[1], original.shape[0]):
        raise ValueError("source image and mask dimensions differ")
    rendered = _overlay(base, original, all_added, target)
    full = rendered.convert("RGB")
    full_draw = ImageDraw.Draw(full)
    full_draw.rectangle(bbox, outline=(255, 220, 40), width=3)
    context_bbox = added_region_bbox(target, padding=24, minimum_side=96)
    micro_bbox = added_region_bbox(target, padding=5, minimum_side=32)
    if context_bbox is None or micro_bbox is None:
        raise ValueError("empty target component")
    context = rendered.crop(context_bbox)
    context_draw = ImageDraw.Draw(context)
    context_draw.rectangle(expanded_local_box(bbox, context_bbox, gap=3),
                           outline=(255, 220, 40), width=1)
    raw_micro = base.crop(micro_bbox).convert("RGB")
    overlay_micro = rendered.crop(micro_bbox)
    overlay_draw = ImageDraw.Draw(overlay_micro)
    overlay_draw.rectangle(expanded_local_box(bbox, micro_bbox, gap=3),
                           outline=(255, 220, 40), width=1)
    canvas = Image.new("RGB", (1240, 360), (9, 15, 22))
    canvas.paste(_fit(full, (345, 290)), (10, 58))
    canvas.paste(_fit(context, (255, 255), nearest=True), (375, 76))
    canvas.paste(_fit(raw_micro, (260, 260), nearest=True), (655, 72))
    canvas.paste(_fit(overlay_micro, (260, 260), nearest=True), (955, 72))
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    draw.text((12, 12), f"{component_id} | {area} px | bbox {list(bbox)}",
              fill=(245, 245, 248), font=font)
    draw.text((12, 30),
              "LEGEND green=source mask | amber=other additions | magenta=target | yellow=outside locator",
              fill=(210, 220, 230), font=font)
    draw.text((12, 47), "FULL LOCATION", fill=(255, 220, 40), font=font)
    draw.text((377, 60), "96PX CONTEXT OVERLAY", fill=(255, 220, 40),
              font=font)
    draw.text((657, 56), "32PX RAW MICRO", fill=(255, 220, 40), font=font)
    draw.text((957, 56), "32PX OVERLAY MICRO", fill=(255, 220, 40),
              font=font)
    canvas.save(destination, "PNG", optimize=True)


def _write_pages(output: Path, rows: list[dict], page_size: int = 24
                 ) -> list[dict]:
    pages = []
    columns, cell_width, cell_height = 4, 400, 168
    font = ImageFont.load_default()
    for page_index, start in enumerate(range(0, len(rows), page_size), 1):
        page_rows = rows[start:start + page_size]
        row_count = (len(page_rows) + columns - 1) // columns
        canvas = Image.new("RGB", (columns * cell_width, row_count * cell_height),
                           (13, 20, 28))
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(page_rows):
            left = index % columns * cell_width
            top = index // columns * cell_height
            with Image.open(output / row["review_image"]) as image:
                preview = image.convert("RGB")
                preview.thumbnail((cell_width - 6, cell_height - 30))
            canvas.paste(preview, (left + (cell_width - preview.width) // 2,
                                   top + 2))
            draw.text((left + 4, top + cell_height - 24),
                      f"{row['component_id']} | {row['area_pixels']} px",
                      fill=(245, 90, 220), font=font)
        name = f"component_review_page_{page_index:02d}.jpg"
        target = output / name
        canvas.save(target, "JPEG", quality=93)
        pages.append({"page": page_index, "first_component_index": start + 1,
                      "last_component_index": start + len(page_rows),
                      "image": name, "sha256": file_sha256(target)})
    return pages


def build_component_review(review_queues: list[Path], identity_manifest: Path,
                           shadow_packet_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    originals, failed, identity_sha, queue_bindings = _validated_source_rows(
        review_queues, identity_manifest)
    shadow_sha = file_sha256(shadow_packet_path)
    shadow = json.loads(shadow_packet_path.read_text(encoding="utf-8"))
    if (shadow.get("candidate_only") is not True
            or shadow.get("human_truth") is not False
            or shadow.get("approved_for_training") is not False
            or shadow.get("production_deployed") is not False
            or shadow.get("source_masks_modified") is not False
            or shadow.get("identity_manifest_sha256") != identity_sha):
        raise ValueError("shadow packet safety flags are inconsistent")
    maximum_pixels = int(shadow.get("maximum_hole_pixels_exclusive", 0))
    shadow_rows = {row.get("sample_id"): row
                   for row in shadow.get("samples", [])}
    if len(shadow_rows) != len(shadow.get("samples", [])):
        raise ValueError("duplicate shadow sample identity")
    if set(shadow_rows) != set(originals):
        raise ValueError("shadow and original sample sets differ")

    pending, rows = [], []
    component_total = 0
    for sample_id in sorted(originals):
        source, shadow_row = originals[sample_id], shadow_rows[sample_id]
        if (shadow_row.get("source_image_sha256") != source["image_sha256"]
                or shadow_row.get("source_union_mask_sha256") !=
                source["mask_sha256"]
                or shadow_row.get("source_queue_sha256") !=
                source["queue_sha256"]):
            raise ValueError(f"shadow binding mismatch: {sample_id}")
        image_bytes = _verified_bytes(source["image_path"],
                                      source["image_sha256"], "source image")
        original = _decode_mask(_verified_bytes(
            source["mask_path"], source["mask_sha256"], "source mask"))
        shadow_mask = _decode_mask(_verified_bytes(
            shadow_packet_path.parent / shadow_row.get("mask", ""),
            shadow_row.get("mask_sha256"), "shadow mask"))
        if (original.shape != shadow_mask.shape
                or np.logical_and(original, ~shadow_mask).any()):
            raise ValueError(f"shadow is not an only-add mask: {sample_id}")
        added = shadow_mask & ~original
        components = extract_added_components(added, maximum_pixels)
        expected_sample_components = sum(
            int(row.get("filled_hole_count", -1))
            for row in shadow_row.get("components", []))
        if (not shadow_row.get("components")
                or any(int(row.get("filled_hole_count", -1)) < 0
                       for row in shadow_row.get("components", []))
                or len(components) != expected_sample_components):
            raise ValueError(f"component count does not match shadow: {sample_id}")
        if sum(row["area"] for row in components) != int(added.sum()):
            raise ValueError(f"component pixel accounting failed: {sample_id}")
        for sample_index, component in enumerate(components, 1):
            component_total += 1
            component_id = f"{sample_id}-c{sample_index:03d}"
            review_name = f"{component_id}-review.png"
            mask_name = f"{component_id}-mask.png"
            row = {
                "component_index": component_total,
                "sample_component_index": sample_index,
                "component_id": component_id,
                "sample_id": sample_id,
                "candidate_only": True, "human_truth": False,
                "approved_for_training": False, "production_deployed": False,
                "assistant_review": "unreviewed",
                "area_pixels": component["area"],
                "bbox_xyxy": list(component["bbox"]),
                "component_digest": component["digest"],
                "source_image_sha256": source["image_sha256"],
                "source_union_mask_sha256": source["mask_sha256"],
                "shadow_mask_sha256": shadow_row.get("mask_sha256"),
                "review_image": review_name,
                "component_mask": mask_name,
            }
            rows.append(row)
            pending.append((row, image_bytes, original, added,
                            component["mask"], component["bbox"]))
    if component_total != int(shadow.get("filled_hole_total", -1)):
        raise ValueError("global component count does not match shadow packet")

    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        for row, image_bytes, original, added, component, bbox in pending:
            x0, y0, x1, y1 = bbox
            mask_crop = (component[y0:y1, x0:x1].astype(np.uint8) * 255)
            Image.fromarray(mask_crop, mode="L").save(
                staging / row["component_mask"], "PNG", optimize=True)
            _render_triptych(image_bytes, original, added, component, bbox,
                             row["component_id"], row["area_pixels"],
                             staging / row["review_image"])
            row["component_mask_sha256"] = file_sha256(
                staging / row["component_mask"])
            row["review_image_sha256"] = file_sha256(
                staging / row["review_image"])
        pages = _write_pages(staging, rows)
        packet = {
            "schema_version": "labelagent.ego_shadow_component_review.v1",
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False, "production_deployed": False,
            "default_enabled": False, "source_masks_modified": False,
            "review_status": "unreviewed",
            "connectivity": 8,
            "identity_manifest_sha256": identity_sha,
            "review_queue_bindings": queue_bindings,
            "shadow_packet_sha256": shadow_sha,
            "changed_sample_count": len({row["sample_id"] for row in rows}),
            "unchanged_sample_count": len(originals) - len({
                row["sample_id"] for row in rows}),
            "failed_closed_input_count": len(failed),
            "component_count": len(rows),
            "added_pixel_total": sum(row["area_pixels"] for row in rows),
            "page_count": len(pages),
            "pages": pages,
            "rows": rows,
            "next_step": "Review every component as pass/fail/uncertain. This packet cannot approve truth, training or deployment.",
        }
        if packet["added_pixel_total"] != int(shadow.get("added_pixel_total", -1)):
            raise ValueError("global added-pixel count does not match shadow packet")
        target = staging / "component_review.json"
        target.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        readback = json.loads(target.read_text(encoding="utf-8"))
        if (readback["component_count"] != len(rows)
                or any(file_sha256(staging / row["review_image"]) !=
                       row["review_image_sha256"] for row in rows)
                or any(file_sha256(staging / row["component_mask"]) !=
                       row["component_mask_sha256"] for row in rows)
                or any(file_sha256(staging / page["image"]) != page["sha256"]
                       for page in pages)):
            raise ValueError("component review readback verification failed")
        staging.replace(output)
    except Exception:
        if staging.exists() and staging.parent == output.parent:
            shutil.rmtree(staging)
        raise
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", action="append", type=Path,
                        required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--shadow-packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    packet = build_component_review(
        arguments.review_queue, arguments.identity_manifest,
        arguments.shadow_packet, arguments.output)
    print(json.dumps({
        "changed_sample_count": packet["changed_sample_count"],
        "component_count": packet["component_count"],
        "added_pixel_total": packet["added_pixel_total"],
        "page_count": packet["page_count"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
