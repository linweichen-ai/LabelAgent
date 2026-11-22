#!/usr/bin/env python3
"""Build a same-ROI comparison pack for three frozen pallet candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCHEMA = "labelagent.pallet_stable_candidate_comparison.v1"
ROUND721_SHA = "b1670d71a5b1f6e6365a8900578457dd54cfd6aa89b6ef2120b1e6f327682007"
ROUND728_SHA = "dde20194fb20fbe18a98d7d8ccbf00d5304f9aec25abba6506d79fd08bbf9c3a"
TARGETS = {
    ("006", "1789899601982497364", 1, "12"),
    ("006", "1789899601982497364", 1, "13"),
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


def read_json_bound(path: Path) -> tuple[dict[str, Any], str]:
    require(path.is_file(), f"missing JSON input: {path}")
    before = sha256(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if sha256(path) != before:
        raise RuntimeError(f"input changed while reading: {path.name}")
    require(isinstance(value, dict), f"expected JSON object: {path.name}")
    return value, before


def bind(path: Path, expected: str, role: str,
         receipts: list[dict[str, Any]], frozen: dict[Path, str]) -> None:
    require(path.is_file(), f"missing {role}: {path}")
    observed = sha256(path)
    require(observed == expected, f"{role} hash mismatch: {path.name}")
    receipts.append({"role": role, "name": path.name,
                     "bytes": path.stat().st_size, "sha256": observed})
    frozen[path.resolve()] = observed


def verify_frozen(frozen: dict[Path, str]) -> None:
    changed = [str(path) for path, expected in frozen.items()
               if not path.is_file() or sha256(path) != expected]
    if changed:
        raise RuntimeError(f"frozen comparison input changed: {changed}")


def resolve_under(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes frozen root: {relative}") from error
    return candidate


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
    require(len(matches) == 1, f"mask lookup must resolve exactly once: {name}")
    return matches[0]


def key_from_item(item: dict[str, Any]) -> tuple[str, str, int, str]:
    return (str(item.get("sequence")), str(item.get("frame_id")),
            int(item.get("camera", -1)), str(item.get("obj_id")))


def key_from_stability(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (str(row.get("sequence")), str(row.get("frame_id")),
            int(row.get("camera", -1)), str(row.get("target_obj_id")))


def image_index(manifest: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in manifest.get("files", []):
        key = (str(row.get("sequence")), str(row.get("frame_id")),
               int(str(row.get("camera", "camera_-1")).split("_")[-1]))
        require(key not in result, f"duplicate Round674 image identity: {key}")
        result[key] = row
    return result


def validate_boundaries(review: dict[str, Any], stability: dict[str, Any],
                        frozen: dict[str, Any]) -> None:
    require(review.get("status") == "candidate_only_human_review_pack_not_deployed",
            "Round721 status drift")
    require(stability.get("status") ==
            "offline_same_source_prompt_mask_stability_audit_not_deployed",
            "Round728 status drift")
    for name, value in (("Round721", review), ("Round728", stability)):
        require(value.get("candidate_only") is True
                and value.get("human_pixel_truth") is False
                and value.get("approved_for_training") is False
                and value.get("training_authorized") is False
                and value.get("automatic_approval") is False
                and value.get("human_truth_writes") == 0
                and value.get("training_data_writes") == 0
                and value.get("workbench_writes") == 0
                and value.get("oss_writes") == 0
                and value.get("production_deployed") is False,
                f"{name} safety boundary drift")
    require(stability.get("independent_validation") is False,
            "Round728 must remain same-source evidence")
    require(frozen.get("evaluation_only") is True
            and frozen.get("training_return_allowed") is False
            and frozen.get("candidate_write_allowed") is False
            and frozen.get("human_truth_write_allowed") is False
            and frozen.get("workbench_writes") == 0
            and frozen.get("human_truth_writes") == 0
            and frozen.get("oss_writes") == 0,
            "Round674 safety boundary drift")


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        array = np.asarray(opened.convert("L"), dtype=np.uint8) > 0
    require(array.shape == (512, 640), f"mask resolution drift: {path.name}")
    require(bool(array.any()), f"empty mask: {path.name}")
    return array


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.count_nonzero(left | right))
    require(union > 0, "empty mask union")
    return float(np.count_nonzero(left & right) / union)


def crop_box(masks: list[np.ndarray], bbox: list[float]) -> tuple[int, int, int, int]:
    require(isinstance(bbox, list) and len(bbox) == 4, "target bbox is invalid")
    x1, y1, x2, y2 = map(float, bbox)
    xs, ys = [int(x1), int(x2 + .999)], [int(y1), int(y2 + .999)]
    for mask in masks:
        ys_nonzero, xs_nonzero = np.nonzero(mask)
        xs.extend((int(xs_nonzero.min()), int(xs_nonzero.max()) + 1))
        ys.extend((int(ys_nonzero.min()), int(ys_nonzero.max()) + 1))
    margin = 10
    return (max(0, min(xs) - margin), max(0, min(ys) - margin),
            min(640, max(xs) + margin), min(512, max(ys) + margin))


def overlay_crop(source: Image.Image, mask: np.ndarray,
                 crop: tuple[int, int, int, int], scale: int) -> Image.Image:
    raw = np.asarray(source.convert("RGB"), dtype=np.float32)[crop[1]:crop[3], crop[0]:crop[2]].copy()
    selected = mask[crop[1]:crop[3], crop[0]:crop[2]]
    color = np.asarray((35, 220, 150), dtype=np.float32)
    raw[selected] = raw[selected] * .50 + color * .50
    image = Image.fromarray(np.clip(raw, 0, 255).astype(np.uint8), mode="RGB")
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def fit_panel(image: Image.Image, width: int = 292, height: int = 330) -> Image.Image:
    canvas = Image.new("RGB", (width, height), (15, 24, 31))
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def render_comparison(source: Image.Image, masks: list[np.ndarray], crop: tuple[int, int, int, int],
                      headings: list[str], output: Path) -> dict[str, Any]:
    crop_width, crop_height = crop[2] - crop[0], crop[3] - crop[1]
    scale = max(1, min(8, 292 // crop_width, 330 // crop_height))
    raw = source.crop(crop).resize((crop_width * scale, crop_height * scale),
                                   Image.Resampling.NEAREST)
    panels = [fit_panel(raw)] + [fit_panel(overlay_crop(source, mask, crop, scale))
                                 for mask in masks]
    canvas = Image.new("RGB", (1500, 430), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8),
              "source=640x512 | no new information | same-source | not independent | candidate-only",
              fill=(15, 25, 30), font=ImageFont.load_default())
    for index, (heading, panel) in enumerate(zip(headings, panels)):
        x = 10 + index * 298
        draw.multiline_text((x, 30), heading, fill=(15, 25, 30),
                            font=ImageFont.load_default(), spacing=2)
        canvas.paste(panel, (x, 92))
    canvas.save(output, format="PNG", compress_level=9)
    return {"relative_path": output.name, "bytes": output.stat().st_size,
            "sha256": sha256(output), "crop_xyxy": list(crop), "nearest_scale": scale}


def build_pack(review_path: Path, stability_path: Path, frozen_manifest_path: Path,
               mask_roots: list[Path], output_dir: Path) -> dict[str, Any]:
    review, review_sha = read_json_bound(review_path)
    stability, stability_sha = read_json_bound(stability_path)
    frozen_manifest, frozen_sha = read_json_bound(frozen_manifest_path)
    require(review_sha == ROUND721_SHA, "Round721 manifest identity mismatch")
    require(stability_sha == ROUND728_SHA, "Round728 manifest identity mismatch")
    validate_boundaries(review, stability, frozen_manifest)
    require(not output_dir.exists(), "output directory already exists")

    review_items: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for item in review.get("items", []):
        key = key_from_item(item)
        require(key not in review_items, f"duplicate Round721 identity: {key}")
        review_items[key] = item
    stability_rows: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for row in stability.get("rows", []):
        key = key_from_stability(row)
        require(key not in stability_rows, f"duplicate Round728 identity: {key}")
        stability_rows[key] = row
    require(set(stability_rows) == TARGETS, "Round728 target identity set drift")
    require(TARGETS.issubset(review_items), "Round721 target identity set drift")

    image_rows = image_index(frozen_manifest)
    receipts: list[dict[str, Any]] = []
    frozen: dict[Path, str] = {}
    bind(review_path, review_sha, "round721_manifest", receipts, frozen)
    bind(stability_path, stability_sha, "round728_manifest", receipts, frozen)
    bind(frozen_manifest_path, frozen_sha, "round674_manifest", receipts, frozen)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp-", dir=output_dir.parent))
    try:
        output_items = []
        for key in sorted(TARGETS, key=lambda value: (value[0], int(value[3]))):
            sequence, frame_id, camera, obj_id = key
            current = review_items[key]
            stable = stability_rows[key]
            image_row = image_rows.get(key[:3])
            require(image_row is not None, f"Round674 source image missing: {key[:3]}")
            image_path = resolve_under(frozen_manifest_path.parent,
                                       str(image_row["relative_path"]))
            bind(image_path, str(image_row["sha256"]),
                 f"source_image_{sequence}_obj{obj_id}", receipts, frozen)
            with Image.open(image_path) as opened:
                source = opened.convert("RGB")
            require(source.size == (640, 512), "source image resolution drift")

            medoid = stable.get("assistant_medoid_candidate") or {}
            require(medoid.get("candidate_only") is True
                    and medoid.get("human_pixel_truth") is False
                    and medoid.get("approved_for_training") is False,
                    "medoid boundary drift")
            current_path = locate_mask(str(current.get("source_mask_name", "")), mask_roots)
            medoid_path = locate_mask(str(medoid.get("mask_name", "")), mask_roots)
            consensus50_info = stable.get("consensus50_mask") or {}
            consensus75_info = stable.get("consensus75_mask") or {}
            consensus50_path = resolve_under(stability_path.parent,
                                             str(consensus50_info.get("relative_path", "")))
            consensus75_path = resolve_under(stability_path.parent,
                                             str(consensus75_info.get("relative_path", "")))
            bind(current_path, str(current["source_mask_sha256"]),
                 f"current_candidate_{sequence}_obj{obj_id}", receipts, frozen)
            bind(medoid_path, str(medoid["mask_sha256"]),
                 f"medoid_candidate_{sequence}_obj{obj_id}", receipts, frozen)
            bind(consensus50_path, str(consensus50_info["sha256"]),
                 f"consensus50_{sequence}_obj{obj_id}", receipts, frozen)
            bind(consensus75_path, str(consensus75_info["sha256"]),
                 f"consensus75_{sequence}_obj{obj_id}", receipts, frozen)
            current_mask, medoid_mask = load_mask(current_path), load_mask(medoid_path)
            consensus50, consensus75 = load_mask(consensus50_path), load_mask(consensus75_path)
            masks = [current_mask, medoid_mask, consensus50, consensus75]
            pixels = [int(mask.sum()) for mask in masks]
            require(pixels[0] == int(current["mask_pixels"]), "Round721 pixel count drift")
            require(pixels[2] == int(consensus50_info["pixels"])
                    and pixels[3] == int(consensus75_info["pixels"]),
                    "Round728 consensus pixel count drift")
            old_medoid_iou = mask_iou(current_mask, medoid_mask)
            headings = [
                f"RAW ROI\n640x512 source pixels",
                f"ROUND721 CURRENT\npolicy={current.get('policy') or 'round676_depth_split'}\n"
                f"px={pixels[0]} IoU-medoid={old_medoid_iou:.4f}\n"
                f"IoU-c50={mask_iou(current_mask, consensus50):.4f}",
                f"ROUND728 MEDOID\npolicy={medoid['policy']}\npx={pixels[1]}\n"
                f"IoU-c50={mask_iou(medoid_mask, consensus50):.4f}",
                f"CONSENSUS 50%\npx={pixels[2]}\n"
                f"IoU-medoid={mask_iou(consensus50, medoid_mask):.4f}",
                f"CONSENSUS 75%\npx={pixels[3]}\n"
                f"IoU-medoid={mask_iou(consensus75, medoid_mask):.4f}",
            ]
            crop = crop_box(masks, current["projected_bbox"])
            output_name = f"{sequence}-frame{frame_id}-camera{camera}-obj{obj_id}-comparison.png"
            image_receipt = render_comparison(source, masks, crop, headings,
                                              temporary / output_name)
            output_items.append({
                "candidate_id": current["candidate_id"], "sequence": sequence,
                "frame_id": frame_id, "camera": camera, "obj_id": obj_id,
                "source_resolution": [640, 512], "no_new_visual_information": True,
                "same_source_same_image": True, "independent_validation": False,
                "current_candidate": {"policy": current.get("policy"),
                                      "mask_name": current_path.name,
                                      "mask_sha256": current["source_mask_sha256"],
                                      "pixels": pixels[0]},
                "assistant_medoid_candidate": {"policy": medoid["policy"],
                                               "mask_name": medoid_path.name,
                                               "mask_sha256": medoid["mask_sha256"],
                                               "pixels": pixels[1]},
                "consensus50": {**consensus50_info, "iou_to_medoid": mask_iou(consensus50, medoid_mask)},
                "consensus75": {**consensus75_info, "iou_to_medoid": mask_iou(consensus75, medoid_mask)},
                "current_equals_medoid_sha256": current["source_mask_sha256"] == medoid["mask_sha256"],
                "current_equals_medoid_pixels": bool(np.array_equal(current_mask, medoid_mask)),
                "current_medoid_pixel_xor_count": int(np.count_nonzero(current_mask ^ medoid_mask)),
                "current_minus_medoid_pixel_count": pixels[0] - pixels[1],
                "current_medoid_iou": old_medoid_iou,
                "current_consensus50_iou": mask_iou(current_mask, consensus50),
                "current_consensus75_iou": mask_iou(current_mask, consensus75),
                "comparison_image": image_receipt,
                "assistant_decision": None, "human_decision": None,
                "candidate_only": True, "human_pixel_truth": False,
                "approved_for_training": False,
            })

        report = {
            "schema_version": SCHEMA,
            "status": "candidate_only_stable_candidate_comparison_not_deployed",
            "summary": {"target_count": 3, "comparison_image_count": 3},
            "source_inputs": sorted(receipts, key=lambda row: (row["role"], row["name"])),
            "items": output_items,
            "source_resolution": [640, 512], "no_new_visual_information": True,
            "same_source_same_image": True, "independent_validation": False,
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
        verify_frozen(frozen)
        temporary.replace(output_dir)
        return {"output": str(output_dir), "manifest": str(output_dir / "manifest.json"),
                "manifest_sha256": sha256(output_dir / "manifest.json"),
                **report["summary"]}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round721-manifest", type=Path, required=True)
    parser.add_argument("--round728-manifest", type=Path, required=True)
    parser.add_argument("--round674-manifest", type=Path, required=True)
    parser.add_argument("--mask-root", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_pack(args.round721_manifest, args.round728_manifest,
                                args.round674_manifest, args.mask_root, args.output_dir),
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
