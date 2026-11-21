#!/usr/bin/env python3
"""Build hash-bound zoom review crops for an ego shadow packet."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from tools.build_ego_component_union_review import _verified_bytes, file_sha256
except ModuleNotFoundError:  # Direct script execution.
    from build_ego_component_union_review import _verified_bytes, file_sha256  # type: ignore


def _decode_mask(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as decoded:
        return np.asarray(decoded.convert("L")) > 0


def added_region_bbox(added: np.ndarray, padding: int = 12,
                      minimum_side: int = 96) -> tuple[int, int, int, int] | None:
    if added.ndim != 2 or padding < 0 or minimum_side <= 0:
        raise ValueError("invalid added-region crop settings")
    ys, xs = np.where(added)
    if not len(xs):
        return None
    height, width = added.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
    target_width = max(x1 - x0 + 2 * padding, minimum_side)
    target_height = max(y1 - y0 + 2 * padding, minimum_side)
    x0 = max(0, center_x - target_width // 2)
    y0 = max(0, center_y - target_height // 2)
    x1 = min(width, x0 + target_width)
    y1 = min(height, y0 + target_height)
    x0 = max(0, x1 - target_width)
    y0 = max(0, y1 - target_height)
    return x0, y0, x1, y1


def _render_crop(image_bytes: bytes, original: np.ndarray, shadow: np.ndarray,
                 bbox: tuple[int, int, int, int], target: Path) -> None:
    with Image.open(io.BytesIO(image_bytes)) as decoded:
        base = decoded.convert("RGBA")
    if base.size != (original.shape[1], original.shape[0]):
        raise ValueError("source image and mask dimensions differ")
    added = shadow & ~original
    rgba = np.zeros((*original.shape, 4), dtype=np.uint8)
    rgba[original] = (24, 206, 138, 70)
    rgba[added] = (245, 40, 220, 245)
    rendered = Image.alpha_composite(base, Image.fromarray(rgba, mode="RGBA"))
    crop = rendered.crop(bbox).convert("RGB")
    scale = max(1, min(8, 640 // max(crop.width, 1), 420 // max(crop.height, 1)))
    if scale > 1:
        crop = crop.resize((crop.width * scale, crop.height * scale),
                           Image.Resampling.NEAREST)
    crop.save(target, "PNG", optimize=True)


def _write_contact_sheet(output: Path, rows: list[dict]) -> Path:
    cell_width, image_height, label_height, columns = 420, 280, 44, 3
    row_count = (len(rows) + columns - 1) // columns
    canvas = Image.new("RGB", (cell_width * columns,
                               (image_height + label_height) * row_count),
                       (15, 23, 31))
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    for index, row in enumerate(rows):
        left = index % columns * cell_width
        top = index // columns * (image_height + label_height)
        with Image.open(output / row["crop"]) as image:
            preview = image.convert("RGB")
            preview.thumbnail((cell_width, image_height))
        canvas.paste(preview, (left + (cell_width - preview.width) // 2,
                               top + (image_height - preview.height) // 2))
        draw.rectangle((left, top + image_height,
                        left + cell_width, top + image_height + label_height),
                       fill=(25, 35, 45))
        draw.text((left + 5, top + image_height + 5), row["sample_id"],
                  fill=(83, 210, 161), font=font)
        draw.text((left + 5, top + image_height + 23),
                  f"added {row['added_pixels']} px | bbox {row['bbox_xyxy']}",
                  fill=(230, 100, 220), font=font)
    target = output / "hole_fill_zoom_contact_sheet.jpg"
    canvas.save(target, "JPEG", quality=92)
    return target


def build_review(review_queues: list[Path], shadow_packet_path: Path,
                 output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    originals = {}
    failed = set()
    for queue_path in review_queues:
        packet = json.loads(queue_path.read_text(encoding="utf-8"))
        if (packet.get("candidate_only") is not True
                or packet.get("human_truth") is not False
                or packet.get("approved_for_training") is not False
                or packet.get("production_deployed") is not False):
            raise ValueError("review queue safety flags are inconsistent")
        for row in packet.get("samples", []):
            sample_id = row.get("sample_id")
            if sample_id in originals or sample_id in failed:
                raise ValueError(f"duplicate sample identity: {sample_id}")
            if row.get("union_available") is not True:
                failed.add(sample_id)
                continue
            originals[sample_id] = {
                "image_path": Path(row.get("source_image", "")),
                "image_sha256": row.get("source_image_sha256"),
                "mask_path": queue_path.parent / row.get("union_mask", ""),
                "mask_sha256": row.get("union_mask_sha256"),
            }
    shadow = json.loads(shadow_packet_path.read_text(encoding="utf-8"))
    if (shadow.get("candidate_only") is not True
            or shadow.get("human_truth") is not False
            or shadow.get("approved_for_training") is not False
            or shadow.get("production_deployed") is not False
            or shadow.get("source_masks_modified") is not False):
        raise ValueError("shadow packet safety flags are inconsistent")
    shadow_rows = {row.get("sample_id"): row for row in shadow.get("samples", [])}
    if set(shadow_rows) != set(originals):
        raise ValueError("shadow and original sample sets differ")
    pending, rows = [], []
    for sample_id in sorted(originals):
        source = originals[sample_id]
        shadow_row = shadow_rows[sample_id]
        if shadow_row.get("source_image_sha256") != source["image_sha256"]:
            raise ValueError(f"shadow image binding mismatch: {sample_id}")
        if shadow_row.get("source_union_mask_sha256") != source["mask_sha256"]:
            raise ValueError(f"shadow mask binding mismatch: {sample_id}")
        image_bytes = _verified_bytes(source["image_path"], source["image_sha256"],
                                      "source image")
        original = _decode_mask(_verified_bytes(source["mask_path"],
                                                source["mask_sha256"],
                                                "source mask"))
        shadow_mask = _decode_mask(_verified_bytes(
            shadow_packet_path.parent / shadow_row.get("mask", ""),
            shadow_row.get("mask_sha256"), "shadow mask"))
        if original.shape != shadow_mask.shape or np.logical_and(
                original, ~shadow_mask).any():
            raise ValueError(f"shadow is not an only-add mask: {sample_id}")
        added = shadow_mask & ~original
        bbox = added_region_bbox(added)
        if bbox is None:
            continue
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            added.astype(np.uint8), connectivity=8)
        crop_name = f"{sample_id}-hole-fill-zoom.png"
        row = {
            "sample_id": sample_id, "candidate_only": True,
            "human_truth": False, "approved_for_training": False,
            "production_deployed": False, "assistant_review": "unreviewed",
            "added_pixels": int(added.sum()),
            "added_component_count": count - 1,
            "maximum_added_component_pixels": (
                int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0),
            "bbox_xyxy": list(bbox), "crop": crop_name,
        }
        rows.append(row)
        pending.append((row, image_bytes, original, shadow_mask, bbox))
    output.mkdir(parents=True)
    for row, image_bytes, original, shadow_mask, bbox in pending:
        _render_crop(image_bytes, original, shadow_mask, bbox,
                     output / row["crop"])
        row["crop_sha256"] = file_sha256(output / row["crop"])
    contact = _write_contact_sheet(output, rows)
    packet = {
        "schema_version": "labelagent.ego_shadow_zoom_review.v1",
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "production_deployed": False,
        "source_masks_modified": False,
        "shadow_packet_sha256": file_sha256(shadow_packet_path),
        "changed_sample_count": len(rows),
        "unchanged_sample_count": len(originals) - len(rows),
        "failed_closed_input_count": len(failed),
        "added_pixel_total": sum(row["added_pixels"] for row in rows),
        "contact_sheet": contact.name,
        "contact_sheet_sha256": file_sha256(contact),
        "rows": rows,
        "next_step": "Review every magenta zoom region. This packet cannot approve truth, training or deployment.",
    }
    target = output / "zoom_review.json"
    target.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", action="append", type=Path, required=True)
    parser.add_argument("--shadow-packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    packet = build_review(arguments.review_queue, arguments.shadow_packet,
                          arguments.output)
    print(json.dumps({
        "changed_sample_count": packet["changed_sample_count"],
        "unchanged_sample_count": packet["unchanged_sample_count"],
        "added_pixel_total": packet["added_pixel_total"],
        "contact_sheet_sha256": packet["contact_sheet_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
