#!/usr/bin/env python3
"""Build a hash-bound visual review sheet for changed ROI shadow candidates."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


SCHEMA_VERSION = "labelagent.pallet-roi-shadow-review.v1"


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def safe_name(value, field):
    if (not isinstance(value, str) or not value or Path(value).name != value
            or "/" in value or "\\" in value):
        raise ValueError(f"unsafe {field}")
    return value


def safe_path(root, value, field):
    name = safe_name(value, field)
    root = Path(root).resolve()
    target = (root / name).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"unsafe {field} path escape") from exc
    return target


def read_bound_image(path, expected_sha):
    payload = path.read_bytes()
    if digest(payload) != expected_sha:
        raise ValueError(f"SHA-256 mismatch: {path.name}")
    if path.read_bytes() != payload:
        raise ValueError(f"input changed during read: {path.name}")
    with Image.open(io.BytesIO(payload)) as decoded:
        return decoded.convert("RGB")


def overlay(image, mask, color):
    array = np.asarray(image).copy()
    target = mask != 0
    tint = np.asarray(color, dtype=np.float32)
    array[target] = np.clip(
        array[target].astype(np.float32) * .55 + tint * .45, 0, 255)
    return Image.fromarray(array.astype(np.uint8))


def panel(image, title, size=(360, 300)):
    canvas = Image.new("RGB", size, (18, 24, 30))
    available = (size[0] - 16, size[1] - 42)
    copy = image.copy()
    copy.thumbnail(available, Image.Resampling.LANCZOS)
    canvas.paste(copy, ((size[0] - copy.width) // 2,
                        34 + (available[1] - copy.height) // 2))
    ImageDraw.Draw(canvas).text((10, 9), title, fill=(238, 244, 248))
    return canvas


def detail_panel(image, title, size=(640, 480), *, nearest=False):
    canvas = Image.new("RGB", size, (18, 24, 30))
    available = (size[0] - 16, size[1] - 42)
    scale = min(available[0] / image.width, available[1] / image.height)
    target = (max(1, int(round(image.width * scale))),
              max(1, int(round(image.height * scale))))
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    copy = image.resize(target, resample)
    canvas.paste(copy, ((size[0] - copy.width) // 2,
                        34 + (available[1] - copy.height) // 2))
    ImageDraw.Draw(canvas).text((10, 9), title, fill=(238, 244, 248))
    return canvas


def detail_box(mask, width, height, minimum_span=128, margin=48):
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("changed candidate contains no added pixels")
    left = int(xs.min()) - margin
    right = int(xs.max()) + margin + 1
    top = int(ys.min()) - margin
    bottom = int(ys.max()) + margin + 1
    if right - left < minimum_span:
        center = (left + right) // 2
        left = center - minimum_span // 2
        right = left + minimum_span
    if bottom - top < minimum_span:
        center = (top + bottom) // 2
        top = center - minimum_span // 2
        bottom = top + minimum_span
    if left < 0:
        right -= left; left = 0
    if top < 0:
        bottom -= top; top = 0
    if right > width:
        left -= right - width; right = width
    if bottom > height:
        top -= bottom - height; bottom = height
    return (max(left, 0), max(top, 0), right, bottom)


def build_review(snapshot_dir, scan_dir, output_dir, *, include_details=False,
                 include_components=False, include_component_context=False):
    snapshot_dir = Path(snapshot_dir).resolve()
    scan_dir = Path(scan_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if include_component_context and not include_components:
        raise ValueError("component context requires component sheets")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    snapshot_payload = (snapshot_dir / "manifest.json").read_bytes()
    scan_payload = (scan_dir / "manifest.json").read_bytes()
    snapshot = json.loads(snapshot_payload.decode("utf-8"))
    scan = json.loads(scan_payload.decode("utf-8"))
    if (scan.get("schema_version") !=
            "labelagent.pallet-evidence-roi-offline-scan.v2"
            or scan.get("snapshot_manifest_sha256") != digest(snapshot_payload)
            or scan.get("candidate_only") is not True
            or scan.get("human_truth") is not False
            or scan.get("training_authorized") is not False
            or scan.get("deployment_authorized") is not False):
        raise ValueError("ROI scan binding or safety flags invalid")
    images = {}
    for item in snapshot.get("images") or []:
        frame = item.get("frame")
        if (isinstance(frame, bool) or not isinstance(frame, int)
                or frame < 0 or frame in images):
            raise ValueError("snapshot frame identity invalid")
        images[frame] = item
    changed_rows = [row for row in scan.get("rows") or []
                    if int(row.get("changed_pixels") or 0) > 0]
    if not changed_rows:
        raise ValueError("ROI scan contains no changed candidates")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    wrapper = Path(tempfile.mkdtemp(prefix=".pallet-roi-review-",
                                    dir=output_dir.parent))
    try:
        staging = wrapper / output_dir.name
        staging.mkdir()
        review_rows = []
        rendered = []
        identities = set()
        component_sheet_count = 0
        component_context_sheet_count = 0
        for row in changed_rows:
            frame = row.get("frame")
            candidate_id = row.get("candidate_id")
            if (isinstance(candidate_id, bool)
                    or not isinstance(candidate_id, int) or candidate_id < 0):
                raise ValueError("candidate identity invalid")
            identity = (frame, candidate_id)
            if identity in identities:
                raise ValueError("duplicate changed candidate identity")
            identities.add(identity)
            item = images.get(frame)
            if not isinstance(item, dict):
                raise ValueError(f"missing source frame {frame}")
            source_path = safe_path(
                snapshot_dir, item.get("image"), "source image")
            source = read_bound_image(
                source_path, item.get("sha256"))
            before_path = safe_path(
                scan_dir, row.get("before_mask"), "before mask")
            after_path = safe_path(
                scan_dir, row.get("after_mask"), "after mask")
            before = np.asarray(read_bound_image(
                before_path, row.get("before_mask_sha256")).convert("L"))
            after = np.asarray(read_bound_image(
                after_path, row.get("after_mask_sha256")).convert("L"))
            if before.shape != (source.height, source.width) or after.shape != before.shape:
                raise ValueError("mask and source dimensions differ")
            before_bool = before != 0
            after_bool = after != 0
            if np.any(before_bool & ~after_bool):
                raise ValueError("review candidate is not add-only")
            added = after_bool & ~before_bool
            if int(np.count_nonzero(added)) != row.get("changed_pixels"):
                raise ValueError("review candidate changed pixel count mismatch")
            if (cv2.connectedComponents(
                    before_bool.astype(np.uint8), connectivity=8)[0]
                    != cv2.connectedComponents(
                        after_bool.astype(np.uint8), connectivity=8)[0]):
                raise ValueError("review candidate foreground component count changed")
            union = (before != 0) | (after != 0)
            ys, xs = np.where(union)
            if not len(xs):
                raise ValueError("changed candidate mask empty")
            margin = 40
            box = (max(int(xs.min()) - margin, 0),
                   max(int(ys.min()) - margin, 0),
                   min(int(xs.max()) + margin + 1, source.width),
                   min(int(ys.max()) + margin + 1, source.height))
            crop = source.crop(box)
            before_crop = before[box[1]:box[3], box[0]:box[2]]
            after_crop = after[box[1]:box[3], box[0]:box[2]]
            diff_crop = before_crop != after_crop
            diff_image = np.asarray(crop).copy()
            diff_image[diff_crop] = np.array([255, 45, 125], dtype=np.uint8)
            label = f"frame {frame} / candidate {candidate_id}"
            rendered.append([
                panel(crop, f"{label} | source"),
                panel(overlay(crop, before_crop, (25, 220, 145)), "before"),
                panel(overlay(crop, after_crop, (35, 170, 255)), "after"),
                panel(Image.fromarray(diff_image),
                      f"added pixels: {row['changed_pixels']}"),
            ])
            review_row = {
                "frame": frame,
                "candidate_id": candidate_id,
                "changed_pixels": row.get("changed_pixels"),
                "source_image_sha256": item.get("sha256"),
                "before_mask_sha256": row.get("before_mask_sha256"),
                "after_mask_sha256": row.get("after_mask_sha256"),
                "crop_box": list(box),
            }
            if include_details:
                focus = detail_box(added, source.width, source.height)
                detail_source = source.crop(focus)
                detail_before = before[focus[1]:focus[3], focus[0]:focus[2]]
                detail_after = after[focus[1]:focus[3], focus[0]:focus[2]]
                detail_added = added[focus[1]:focus[3], focus[0]:focus[2]]
                added_image = np.asarray(detail_source).copy()
                added_image[detail_added] = np.array([255, 45, 125], dtype=np.uint8)
                detail = Image.new("RGB", (640 * 4, 480), (12, 18, 24))
                panes = [
                    detail_panel(detail_source, f"{label} | source detail"),
                    detail_panel(overlay(
                        detail_source, detail_before, (25, 220, 145)), "before"),
                    detail_panel(overlay(
                        detail_source, detail_after, (35, 170, 255)), "after"),
                    detail_panel(Image.fromarray(added_image),
                                 f"added pixels: {row['changed_pixels']}",
                                 nearest=True),
                ]
                for column, pane in enumerate(panes):
                    detail.paste(pane, (column * 640, 0))
                detail_name = (
                    f"frame-{frame:06d}-candidate-{candidate_id}-detail.png")
                detail_path = staging / detail_name
                detail.save(detail_path, "PNG", optimize=False, compress_level=9)
                review_row.update({
                    "detail_sheet": detail_name,
                    "detail_sheet_sha256": digest(detail_path.read_bytes()),
                    "detail_crop_box": list(focus),
                })
            if include_components:
                source_array = np.asarray(source)
                gray = cv2.cvtColor(source_array, cv2.COLOR_RGB2GRAY)
                gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                gradient = cv2.magnitude(gradient_x, gradient_y)
                component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
                    added.astype(np.uint8), connectivity=8)
                components = []
                for label_index in range(1, component_count):
                    x, y, component_width, component_height, component_area = [
                        int(value) for value in stats[label_index]]
                    component_mask = labels == label_index
                    components.append((y, x, component_area, component_width,
                                       component_height, component_mask))
                components.sort(key=lambda item: (item[0], item[1], item[2]))
                component_rows = []
                ring_kernel = np.ones((5, 5), dtype=np.uint8)
                contact_kernel = np.ones((3, 3), dtype=np.uint8)
                before_dilated = cv2.dilate(
                    before_bool.astype(np.uint8), contact_kernel,
                    iterations=1).astype(bool)
                for component_index, component in enumerate(components, start=1):
                    (y, x, component_area, component_width,
                     component_height, component_mask) = component
                    component_focus = detail_box(
                        component_mask, source.width, source.height,
                        minimum_span=96, margin=32)
                    component_source = source.crop(component_focus)
                    component_before = before[
                        component_focus[1]:component_focus[3],
                        component_focus[0]:component_focus[2]]
                    component_after_mask = (
                        before_bool | component_mask).astype(np.uint8) * 255
                    component_after = component_after_mask[
                        component_focus[1]:component_focus[3],
                        component_focus[0]:component_focus[2]]
                    component_local = component_mask[
                        component_focus[1]:component_focus[3],
                        component_focus[0]:component_focus[2]]
                    added_image = np.asarray(component_source).copy()
                    added_image[component_local] = np.array(
                        [255, 45, 125], dtype=np.uint8)
                    detail = Image.new(
                        "RGB", (640 * 4, 480), (12, 18, 24))
                    component_label = (
                        f"{label} | component {component_index}")
                    panes = [
                        detail_panel(
                            component_source, f"{component_label} | source"),
                        detail_panel(overlay(
                            component_source, component_before,
                            (25, 220, 145)), "before"),
                        detail_panel(overlay(
                            component_source, component_after,
                            (35, 170, 255)), "after component"),
                        detail_panel(
                            Image.fromarray(added_image),
                            f"component pixels: {component_area}", nearest=True),
                    ]
                    for column, pane in enumerate(panes):
                        detail.paste(pane, (column * 640, 0))
                    component_name = (
                        f"frame-{frame:06d}-candidate-{candidate_id}-"
                        f"component-{component_index:02d}-detail.png")
                    component_path = staging / component_name
                    detail.save(
                        component_path, "PNG", optimize=False,
                        compress_level=9)

                    component_context_row = {}
                    if include_component_context:
                        candidate_context = source.crop(box)
                        context_detail = Image.new(
                            "RGB", (640 * 5, 480), (12, 18, 24))
                        context_panes = [
                            detail_panel(
                                candidate_context,
                                f"{component_label} | candidate context"),
                            detail_panel(
                                component_source, "local source"),
                            detail_panel(overlay(
                                component_source, component_before,
                                (25, 220, 145)), "local before"),
                            detail_panel(overlay(
                                component_source, component_after,
                                (35, 170, 255)), "local after"),
                            detail_panel(
                                Image.fromarray(added_image),
                                f"component pixels: {component_area}",
                                nearest=True),
                        ]
                        for column, pane in enumerate(context_panes):
                            context_detail.paste(pane, (column * 640, 0))
                        context_name = (
                            f"frame-{frame:06d}-candidate-{candidate_id}-"
                            f"component-{component_index:02d}-context-detail.png")
                        context_path = staging / context_name
                        context_detail.save(
                            context_path, "PNG", optimize=False,
                            compress_level=9)
                        component_context_row = {
                            "context_detail_sheet": context_name,
                            "context_detail_sheet_sha256": digest(
                                context_path.read_bytes()),
                            "context_crop_box": list(box),
                        }
                        component_context_sheet_count += 1

                    ring = (cv2.dilate(
                        component_mask.astype(np.uint8), ring_kernel,
                        iterations=1).astype(bool)
                        & ~component_mask & before_bool)
                    notch_mean = source_array[component_mask].astype(
                        np.float32).mean(axis=0)
                    ring_mean = (
                        source_array[ring].astype(np.float32).mean(axis=0)
                        if ring.any() else None)
                    component_after_full = before_bool | component_mask
                    standalone_component_count_preserved = (
                        cv2.connectedComponents(
                            before_bool.astype(np.uint8), connectivity=8)[0]
                        == cv2.connectedComponents(
                            component_after_full.astype(np.uint8),
                            connectivity=8)[0])
                    if not standalone_component_count_preserved:
                        raise ValueError(
                            "review component is not independently connected")
                    component_row = {
                        "component_index": component_index,
                        "area": component_area,
                        "bbox": [x, y, x + component_width - 1,
                                 y + component_height - 1],
                        "contact_pixels": int(np.count_nonzero(
                            component_mask & before_dilated)),
                        "contact_ratio": round(
                            int(np.count_nonzero(
                                component_mask & before_dilated))
                            / component_area, 6),
                        "notch_gradient_mean": round(float(
                            gradient[component_mask].mean()), 3),
                        "ring_gradient_mean": (
                            round(float(gradient[ring].mean()), 3)
                            if ring.any() else None),
                        "gray_delta": (
                            round(float(ring_mean.mean()
                                        - notch_mean.mean()), 3)
                            if ring_mean is not None else None),
                        "color_distance": (
                            round(float(np.linalg.norm(
                                ring_mean - notch_mean)), 3)
                            if ring_mean is not None else None),
                        "standalone_component_count_preserved": (
                            standalone_component_count_preserved),
                        "detail_sheet": component_name,
                        "detail_sheet_sha256": digest(
                            component_path.read_bytes()),
                        "detail_crop_box": list(component_focus),
                    }
                    component_row.update(component_context_row)
                    component_rows.append(component_row)
                    component_sheet_count += 1
                if sum(item["area"] for item in component_rows) != int(
                        np.count_nonzero(added)):
                    raise ValueError("component pixel accounting mismatch")
                review_row["component_sheets"] = component_rows
            review_rows.append(review_row)

        sheet = Image.new("RGB", (360 * 4, 300 * len(rendered)), (12, 18, 24))
        for row_index, panes in enumerate(rendered):
            for column, pane in enumerate(panes):
                sheet.paste(pane, (column * 360, row_index * 300))
        sheet_path = staging / "contact-sheet.jpg"
        sheet.save(sheet_path, "JPEG", quality=95, subsampling=0)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "snapshot_manifest_sha256": digest(snapshot_payload),
            "scan_manifest_sha256": digest(scan_payload),
            "changed_candidate_count": len(review_rows),
            "candidate_only": True,
            "human_truth": False,
            "training_authorized": False,
            "deployment_authorized": False,
            "geometry_policy": "appearance_evidence_roi",
            "workbench_write": False,
            "detail_sheets_included": include_details,
            "contact_sheet_sha256": digest(sheet_path.read_bytes()),
            "rows": review_rows,
        }
        if include_components:
            manifest["component_sheets_included"] = True
            manifest["component_sheet_count"] = component_sheet_count
        if include_component_context:
            manifest["component_context_sheets_included"] = True
            manifest["component_context_sheet_count"] = (
                component_context_sheet_count)
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2,
                       sort_keys=True) + "\n", encoding="utf-8")
        manifest_sha = digest(manifest_path.read_bytes())
        staging.replace(output_dir)
        return {
            "schema_version": SCHEMA_VERSION,
            "changed_candidate_count": len(review_rows),
            "contact_sheet": str(output_dir / sheet_path.name),
            "contact_sheet_sha256": manifest["contact_sheet_sha256"],
            "detail_sheet_count": (
                len(review_rows) if include_details else 0),
            "component_sheet_count": component_sheet_count,
            "component_context_sheet_count": component_context_sheet_count,
            "manifest_sha256": manifest_sha,
        }
    finally:
        shutil.rmtree(wrapper, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--scan-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-details", action="store_true")
    parser.add_argument("--include-components", action="store_true")
    parser.add_argument("--include-component-context", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_review(
        args.snapshot_dir, args.scan_dir, args.output,
        include_details=args.include_details,
        include_components=args.include_components,
        include_component_context=args.include_component_context),
        ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
