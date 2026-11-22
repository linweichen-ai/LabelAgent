#!/usr/bin/env python3
"""Build a same-camera temporal review sheet for one ego mask component."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from tools.evaluate_ego_tail_vlm_guard import file_sha256
except ModuleNotFoundError:  # Direct execution adds tools/, not the repo root.
    from evaluate_ego_tail_vlm_guard import file_sha256


IDENTITY = re.compile(
    r"^(?P<package>\d{3})-(?P<camera>camera\d)-idx(?P<index>\d+)-"
    r"(?P<frame>\d+)(?:-c\d+)?$")


def crop_box_around_bbox(bbox: list[int], image_size: tuple[int, int],
                         side: int = 96) -> tuple[int, int, int, int]:
    if len(bbox) != 4 or side <= 0:
        raise ValueError("invalid bbox or crop side")
    x0, y0, x1, y1 = bbox
    width, height = image_size
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("component bbox is outside the image")
    side = min(side, width, height)
    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2
    left = min(max(center_x - side // 2, 0), width - side)
    top = min(max(center_y - side // 2, 0), height - side)
    return left, top, left + side, top + side


def parse_identity(value: str) -> dict[str, int | str]:
    match = IDENTITY.fullmatch(value)
    if not match:
        raise ValueError(f"unsupported component identity: {value}")
    result: dict[str, int | str] = match.groupdict()
    result["index"] = int(result["index"])
    return result


def photometric_evidence(image: Image.Image, target: np.ndarray,
                         radius: int = 2) -> dict[str, float | int]:
    """Measure target-versus-nearby luminance without making a decision."""
    if radius < 1 or target.ndim != 2 or target.shape != (
            image.height, image.width) or not target.any():
        raise ValueError("invalid target mask or photometric radius")
    padded = np.pad(target, radius, mode="constant")
    dilated = np.zeros_like(target, dtype=bool)
    height, width = target.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            dilated |= padded[dy:dy + height, dx:dx + width]
    ring = dilated & ~target
    if not ring.any():
        raise ValueError("photometric ring is empty")
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    luminance = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1]
                 + 0.0722 * rgb[..., 2])
    target_mean = float(luminance[target].mean())
    ring_mean = float(luminance[ring].mean())
    return {
        "target_pixels": int(target.sum()),
        "ring_pixels": int(ring.sum()),
        "target_luma_mean": round(target_mean, 3),
        "ring_luma_mean": round(ring_mean, 3),
        "target_minus_ring_luma": round(target_mean - ring_mean, 3),
    }


def build_temporal_sheet(component_review_path: Path, component_id: str,
                         source_dir: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    packet = json.loads(component_review_path.read_text(encoding="utf-8"))
    if (packet.get("candidate_only") is not True
            or packet.get("human_truth") is not False
            or packet.get("approved_for_training") is not False
            or packet.get("production_deployed") is not False
            or packet.get("default_enabled") is not False):
        raise ValueError("component review safety flags are inconsistent")
    rows = [row for row in packet.get("rows", [])
            if row.get("component_id") == component_id]
    if len(rows) != 1:
        raise ValueError("component identity is missing or duplicated")
    row = rows[0]
    identity = parse_identity(component_id)
    source_pattern = (
        f"{identity['package']}-{identity['camera']}-idx*-*.png")
    candidates = []
    for path in source_dir.glob(source_pattern):
        parsed = parse_identity(path.stem)
        candidates.append((int(parsed["index"]), str(parsed["frame"]), path))
    candidates.sort()
    if len(candidates) < 2:
        raise ValueError("at least two temporal source frames are required")
    expected_name = f"{row['sample_id']}.png"
    target_paths = [path for _, _, path in candidates
                    if path.name == expected_name]
    if len(target_paths) != 1 or file_sha256(target_paths[0]) != row.get(
            "source_image_sha256"):
        raise ValueError("target source image binding differs")
    mask_path = component_review_path.parent / row.get("component_mask", "")
    if file_sha256(mask_path) != row.get("component_mask_sha256"):
        raise ValueError("component mask binding differs")
    with Image.open(mask_path) as mask_image:
        local_target_mask = np.asarray(mask_image.convert("L")) > 0
    if int(local_target_mask.sum()) != int(row.get("area_pixels", -1)):
        raise ValueError("component mask area differs")

    decoded = []
    image_size = None
    for index, frame_id, path in candidates:
        with Image.open(path) as image:
            raw = image.convert("RGB")
        if image_size is None:
            image_size = raw.size
        elif raw.size != image_size:
            raise ValueError("temporal image dimensions differ")
        decoded.append((index, frame_id, path, raw))
    assert image_size is not None
    x0, y0, x1, y1 = row["bbox_xyxy"]
    if local_target_mask.shape != (y1 - y0, x1 - x0):
        raise ValueError("component mask dimensions differ from its bbox")
    target_mask = np.zeros((image_size[1], image_size[0]), dtype=bool)
    target_mask[y0:y1, x0:x1] = local_target_mask
    crop_box = crop_box_around_bbox(row["bbox_xyxy"], image_size)
    columns, cell_width, cell_height = 3, 320, 340
    row_count = (len(decoded) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_width, row_count * cell_height),
                       (9, 15, 22))
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    cx0, cy0, _, _ = crop_box
    local_box = (max(0, x0 - cx0 - 4), max(0, y0 - cy0 - 4),
                 min(crop_box[2] - cx0 - 1, x1 - cx0 - 1 + 4),
                 min(crop_box[3] - cy0 - 1, y1 - cy0 - 1 + 4))
    sheet_rows = []
    for position, (index, frame_id, path, image) in enumerate(decoded):
        crop = image.crop(crop_box)
        crop_draw = ImageDraw.Draw(crop)
        crop_draw.rectangle(local_box, outline=(255, 220, 40), width=1)
        crop = crop.resize((300, 300), Image.Resampling.NEAREST)
        left = position % columns * cell_width
        top = position // columns * cell_height
        canvas.paste(crop, (left + 10, top + 4))
        elapsed = (int(frame_id) - int(identity["frame"])) / 1_000_000_000
        draw.text((left + 10, top + 310),
                  f"idx {index} | delta {elapsed:+.3f}s",
                  fill=(255, 220, 40), font=font)
        draw.text((left + 10, top + 326), path.name,
                  fill=(225, 232, 240), font=font)
        sheet_rows.append({
            "index": index,
            "frame_id": frame_id,
            "delta_seconds": round(elapsed, 6),
            "source_image": str(path),
            "source_image_sha256": file_sha256(path),
            "photometric": photometric_evidence(image, target_mask),
        })

    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    try:
        sheet_name = f"{component_id}-temporal-review.png"
        sheet_path = temporary / sheet_name
        canvas.save(sheet_path, "PNG", optimize=True)
        result = {
            "schema_version": "labelagent.ego_component_temporal_review.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_only": True,
            "assistant_review_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "production_deployed": False,
            "default_enabled": False,
            "automatic_decisions_enabled": False,
            "component_review_sha256": file_sha256(component_review_path),
            "component_id": component_id,
            "bbox_xyxy": row["bbox_xyxy"],
            "crop_box_xyxy": list(crop_box),
            "same_pixel_location_only": True,
            "motion_compensation_applied": False,
            "temporal_frame_count": len(sheet_rows),
            "photometric_dark_delta_all_frames": all(
                item["photometric"]["target_minus_ring_luma"] < 0
                for item in sheet_rows),
            "rows": sheet_rows,
            "sheet": sheet_name,
            "sheet_sha256": file_sha256(sheet_path),
            "boundary": (
                "This sheet provides same-camera, same-pixel temporal context "
                "for a fixed ego structure. Any visible drift or ambiguity "
                "must remain uncertain; it is not truth or registration."),
        }
        manifest = temporary / "temporal_review_manifest.json"
        manifest.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        readback = json.loads(manifest.read_text(encoding="utf-8"))
        if readback["temporal_frame_count"] != len(sheet_rows):
            raise ValueError("temporal review readback mismatch")
        temporary.replace(output)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-review", required=True, type=Path)
    parser.add_argument("--component-id", required=True)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    result = build_temporal_sheet(
        arguments.component_review, arguments.component_id,
        arguments.source_dir, arguments.output)
    print(json.dumps({key: result[key] for key in (
        "component_id", "temporal_frame_count", "sheet_sha256")},
        ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
