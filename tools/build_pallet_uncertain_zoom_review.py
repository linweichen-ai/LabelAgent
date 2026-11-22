#!/usr/bin/env python3
"""Build a deterministic, candidate-only zoom pack for six frozen pallet masks.

Every panel is derived from the same 640x512 frozen source image.  Scaling makes
existing pixels easier to inspect; it does not add visual information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


SCHEMA = "labelagent.pallet_uncertain_zoom_review.v1"
EXPECTED_REVIEW_MANIFEST_SHA = "b1670d71a5b1f6e6365a8900578457dd54cfd6aa89b6ef2120b1e6f327682007"
EXPECTED = {
    ("006", "1789899601982497364", 1, "11"),
    ("006", "1789899601982497364", 1, "12"),
    ("006", "1789899601982497364", 1, "13"),
    ("006", "1789899601982497364", 1, "14"),
    ("008", "1789899778882397075", 1, "67"),
    ("008", "1789899778882397075", 1, "78"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bound_json(path: Path) -> tuple[dict[str, Any], str]:
    require(path.is_file(), f"missing JSON input: {path}")
    before = sha256(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if sha256(path) != before:
        raise RuntimeError(f"input changed while reading: {path.name}")
    require(isinstance(value, dict), f"expected JSON object: {path.name}")
    return value, before


def bind(path: Path, expected: str, role: str,
         receipts: list[dict[str, Any]], frozen: dict[Path, str]) -> str:
    require(path.is_file(), f"missing {role}: {path}")
    observed = sha256(path)
    require(observed == expected, f"{role} hash mismatch: {path.name}")
    receipts.append({"role": role, "name": path.name,
                     "bytes": path.stat().st_size, "sha256": observed})
    frozen[path.resolve()] = observed
    return observed


def verify_frozen(frozen: dict[Path, str]) -> None:
    changed = [str(path) for path, expected in frozen.items()
               if not path.is_file() or sha256(path) != expected]
    if changed:
        raise RuntimeError(f"frozen input changed while rendering: {changed}")


def resolve_under(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes frozen root: {relative}") from error
    return candidate


def validate_safety(review: dict[str, Any], frozen: dict[str, Any]) -> None:
    require(review.get("status") == "candidate_only_human_review_pack_not_deployed",
            "Round721 status drift")
    require(review.get("candidate_only") is True
            and review.get("human_pixel_truth") is False
            and review.get("approved_for_training") is False
            and review.get("training_authorized") is False
            and review.get("automatic_approval") is False
            and review.get("human_truth_writes") == 0
            and review.get("training_data_writes") == 0
            and review.get("workbench_writes") == 0
            and review.get("oss_writes") == 0
            and review.get("production_deployed") is False,
            "Round721 safety boundary drift")
    require(frozen.get("evaluation_only") is True
            and frozen.get("training_return_allowed") is False
            and frozen.get("candidate_write_allowed") is False
            and frozen.get("human_truth_write_allowed") is False
            and frozen.get("workbench_writes") == 0
            and frozen.get("human_truth_writes") == 0
            and frozen.get("oss_writes") == 0,
            "Round674 safety boundary drift")


def item_key(item: dict[str, Any]) -> tuple[str, str, int, str]:
    return (str(item.get("sequence")), str(item.get("frame_id")),
            int(item.get("camera", -1)), str(item.get("obj_id")))


def frozen_image_index(manifest: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in manifest.get("files", []):
        key = (str(row.get("sequence")), str(row.get("frame_id")),
               int(str(row.get("camera", "camera_-1")).split("_")[-1]))
        require(key not in result, f"duplicate Round674 image identity: {key}")
        result[key] = row
    return result


def bbox_int(box: list[float]) -> tuple[int, int, int, int]:
    require(isinstance(box, list) and len(box) == 4, "projected bbox is invalid")
    x1, y1, x2, y2 = map(float, box)
    require(x2 > x1 and y2 > y1, "projected bbox is empty")
    return int(x1), int(y1), int(x2 + .999), int(y2 + .999)


def crop_for_group(items: list[dict[str, Any]], mask: Image.Image) -> tuple[int, int, int, int]:
    boxes = [bbox_int(item["projected_bbox"]) for item in items]
    xs = [value for box in boxes for value in (box[0], box[2])]
    ys = [value for box in boxes for value in (box[1], box[3])]
    mask_box = mask.getbbox()
    if mask_box:
        xs.extend((mask_box[0], mask_box[2]))
        ys.extend((mask_box[1], mask_box[3]))
    margin = 8
    left, top = max(0, min(xs) - margin), max(0, min(ys) - margin)
    right, bottom = min(mask.width, max(xs) + margin), min(mask.height, max(ys) + margin)
    require(right > left and bottom > top, "zoom crop is empty")
    return left, top, right, bottom


def panel_canvas(image: Image.Image, size: int = 352) -> Image.Image:
    canvas = Image.new("RGB", (size, size), (15, 24, 31))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def render_zoom(source: Image.Image, mask: Image.Image, group: list[dict[str, Any]],
                target: dict[str, Any], output: Path) -> tuple[str, dict[str, Any]]:
    crop = crop_for_group(group, mask)
    raw = source.crop(crop)
    mask_crop = mask.crop(crop)
    scale = max(1, min(8, 352 // max(1, raw.width), 352 // max(1, raw.height)))
    nearest = raw.resize((raw.width * scale, raw.height * scale), Image.Resampling.NEAREST)
    smooth = raw.resize((raw.width * scale, raw.height * scale), Image.Resampling.LANCZOS)
    overlay = nearest.convert("RGBA")
    mask_scaled = mask_crop.resize(nearest.size, Image.Resampling.NEAREST)
    tint = Image.new("RGBA", nearest.size, (255, 214, 40, 0))
    tint.putalpha(mask_scaled.point(lambda value: 118 if value else 0))
    overlay = Image.alpha_composite(overlay, tint)
    draw = ImageDraw.Draw(overlay)
    ox, oy = crop[0], crop[1]
    for item in sorted(group, key=lambda value: int(value["obj_id"])):
        x1, y1, x2, y2 = bbox_int(item["projected_bbox"])
        coords = tuple(int(round(value * scale)) for value in
                       (x1 - ox, y1 - oy, x2 - ox, y2 - oy))
        is_target = str(item["obj_id"]) == str(target["obj_id"])
        color = (255, 214, 40, 255) if is_target else (45, 215, 235, 255)
        draw.rectangle(coords, outline=color, width=3 if is_target else 1)
        draw.text((coords[0] + 2, coords[1] + 1), str(item["obj_id"]), fill=color,
                  font=ImageFont.load_default())

    canvas = Image.new("RGB", (1480, 430), "white")
    draw_canvas = ImageDraw.Draw(canvas)
    title = (f"{target['candidate_id']} | source=640x512 | scale={scale}x | "
             "candidate-only | no new visual information")
    draw_canvas.text((12, 10), title, fill=(10, 20, 26), font=ImageFont.load_default())
    labels = ["RAW ROI (1:1)", f"NEAREST {scale}x (pixel-faithful)",
              f"LANCZOS {scale}x (viewing only)", "OVERLAY + conflict bboxes"]
    views = [panel_canvas(raw), panel_canvas(nearest), panel_canvas(smooth),
             panel_canvas(overlay.convert("RGB"))]
    for index, (label, view) in enumerate(zip(labels, views)):
        x = 12 + index * 366
        draw_canvas.text((x, 34), label, fill=(10, 20, 26), font=ImageFont.load_default())
        canvas.paste(view, (x, 58))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=False, compress_level=9)
    receipt = {"relative_path": output.name, "bytes": output.stat().st_size,
               "sha256": sha256(output), "crop_xyxy": list(crop),
               "crop_size": [raw.width, raw.height], "nearest_scale": scale}
    return receipt["sha256"], receipt


def locate_mask(name: str, roots: list[Path]) -> Path:
    matches: list[Path] = []
    for root in roots:
        root = root.resolve()
        require(root.is_dir(), f"mask root is not a directory: {root}")
        candidate = (root / name).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"mask name escapes root: {name}") from error
        if candidate.is_file():
            matches.append(candidate)
    require(len(matches) == 1, f"candidate mask lookup must resolve once: {name}")
    return matches[0]


def build_pack(review_manifest_path: Path, frozen_manifest_path: Path,
               mask_roots: list[Path], output_dir: Path) -> dict[str, Any]:
    review, review_sha = read_bound_json(review_manifest_path)
    frozen_manifest, frozen_sha = read_bound_json(frozen_manifest_path)
    require(review_sha == EXPECTED_REVIEW_MANIFEST_SHA,
            "Round721 manifest identity mismatch")
    validate_safety(review, frozen_manifest)
    require(not output_dir.exists(), "output directory already exists")

    receipts: list[dict[str, Any]] = []
    frozen_files: dict[Path, str] = {}
    bind(review_manifest_path, review_sha, "round721_manifest", receipts, frozen_files)
    bind(frozen_manifest_path, frozen_sha, "round674_manifest", receipts, frozen_files)

    all_items: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for item in review.get("items", []):
        key = item_key(item)
        require(key not in all_items, f"duplicate Round721 item: {key}")
        all_items[key] = item
    require(EXPECTED.issubset(all_items), "uncertain identity set is incomplete")
    selected = {key: all_items[key] for key in EXPECTED}
    require(set(selected) == EXPECTED, "uncertain identity set drift")
    for item in selected.values():
        require(item.get("candidate_only") is True
                and item.get("human_pixel_truth") is False
                and item.get("approved_for_training") is False,
                "selected item safety boundary drift")

    frames = {(str(frame["sequence"]), str(frame["frame_id"]), int(frame["camera"])): frame
              for frame in review.get("frames", [])}
    require(len(frames) == len(review.get("frames", [])), "duplicate Round721 frame")
    frozen_images = frozen_image_index(frozen_manifest)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp-", dir=output_dir.parent))
    try:
        output_items: list[dict[str, Any]] = []
        cached_frames: dict[tuple[str, str, int], tuple[Image.Image, dict[str, Any]]] = {}
        for key in sorted(EXPECTED, key=lambda value: (value[0], int(value[3]))):
            sequence, frame_id, camera, obj_id = key
            frame_key = key[:3]
            frame = frames.get(frame_key)
            image_row = frozen_images.get(frame_key)
            require(frame is not None and image_row is not None, f"frame identity missing: {frame_key}")
            require(frame.get("source_image_sha256") == image_row.get("sha256"),
                    f"image receipt mismatch: {frame_key}")
            if frame_key not in cached_frames:
                image_path = resolve_under(frozen_manifest_path.parent,
                                           str(image_row["relative_path"]))
                bind(image_path, str(image_row["sha256"]),
                     f"source_image_{sequence}", receipts, frozen_files)
                label_path = resolve_under(frozen_manifest_path.parent,
                                           f"labels/{sequence}/{frame_id}.json")
                bind(label_path, str(frame["source_label_sha256"]),
                     f"source_label_{sequence}", receipts, frozen_files)
                label = json.loads(label_path.read_text(encoding="utf-8"))
                require(str(label.get("frame_id")) == frame_id, "label frame_id drift")
                pallet_ids = {str(obj.get("obj_id")) for obj in label.get("objects", [])
                              if obj.get("obj_type") == "Pallet"}
                require(set(frame["review_scope_obj_ids"]).issubset(pallet_ids),
                        "label pallet identity drift")
                with Image.open(image_path) as opened:
                    source = opened.convert("RGB")
                require(source.size == (640, 512), "source resolution is not 640x512")
                cached_frames[frame_key] = (source, frame)
            source, frame = cached_frames[frame_key]
            group = [all_items[(sequence, frame_id, camera, member)]
                     for member in frame["review_scope_obj_ids"]]
            require(len(group) == len(frame["review_scope_obj_ids"]), "conflict group drift")
            target = selected[key]
            mask_path = locate_mask(str(target["source_mask_name"]), mask_roots)
            bind(mask_path, str(target["source_mask_sha256"]),
                 f"candidate_mask_{sequence}_obj{obj_id}", receipts, frozen_files)
            with Image.open(mask_path) as opened:
                mask = opened.convert("L")
            require(mask.size == source.size, "candidate mask resolution mismatch")
            relative = f"{sequence}-frame{frame_id}-camera{camera}-obj{obj_id}-zoom.png"
            _, image_receipt = render_zoom(source, mask, group, target, temporary / relative)
            output_items.append({
                "candidate_id": target["candidate_id"], "sequence": sequence,
                "frame_id": frame_id, "camera": camera, "obj_id": obj_id,
                "source_resolution": [640, 512], "no_new_visual_information": True,
                "candidate_mask_name": mask_path.name,
                "candidate_mask_sha256": target["source_mask_sha256"],
                "conflict_group_obj_ids": list(frame["review_scope_obj_ids"]),
                "target_bbox": target["projected_bbox"], **image_receipt,
                "candidate_only": True, "human_pixel_truth": False,
                "approved_for_training": False, "assistant_decision": None,
                "human_decision": None,
            })

        require(len(output_items) == 6, "zoom item count drift")
        report = {
            "schema_version": SCHEMA,
            "status": "candidate_only_uncertain_zoom_review_not_deployed",
            "summary": {"zoom_item_count": 6, "source_frame_count": 2},
            "source_resolution": [640, 512],
            "no_new_visual_information": True,
            "scaling_note": "nearest/Lanczos panels enlarge the same frozen 640x512 pixels",
            "source_inputs": sorted(receipts, key=lambda row: (row["role"], row["name"])),
            "items": output_items,
            "candidate_only": True, "human_pixel_truth": False,
            "approved_for_training": False, "training_authorized": False,
            "automatic_approval": False, "human_truth_writes": 0,
            "training_data_writes": 0, "workbench_writes": 0, "oss_writes": 0,
            "production_deployed": False, "readback_verified": True,
        }
        manifest = temporary / "manifest.json"
        manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        require(json.loads(manifest.read_text(encoding="utf-8")) == report,
                "output manifest readback mismatch")
        verify_frozen(frozen_files)
        temporary.replace(output_dir)
        return {"output": str(output_dir), "manifest": str(output_dir / "manifest.json"),
                "manifest_sha256": sha256(output_dir / "manifest.json"),
                "zoom_item_count": len(output_items)}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round721-manifest", type=Path, required=True)
    parser.add_argument("--round674-manifest", type=Path, required=True)
    parser.add_argument("--mask-root", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_pack(args.round721_manifest, args.round674_manifest,
                                args.mask_root, args.output_dir), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
