"""Build a local-only SAM3 crop dataset from assistant-reviewed manual instances.

The output is deliberately labelled as an isolated candidate dataset.  It does
not turn assistant review into human truth and it never includes test24.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


REVIEW_SCHEMA = "labelagent.assistant-delegated-instance-review.v1"
CANDIDATE_SCHEMA = "labelagent.cvat-manual-instance-candidates.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: JSON root must be an object")
    return value


def _parts(shape: dict[str, Any]) -> list[dict[str, Any]]:
    parts = shape.get("parts")
    if isinstance(parts, list) and parts:
        return parts
    return [{"points": shape.get("points"), "holes": shape.get("holes", [])}]


def _points(value: Any, *, field: str) -> list[tuple[float, float]]:
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError(f"{field} must contain at least three points")
    result: list[tuple[float, float]] = []
    for point in value:
        if (
            not isinstance(point, list)
            or len(point) != 2
            or not all(isinstance(item, (int, float)) for item in point)
        ):
            raise ValueError(f"{field} contains an invalid point")
        result.append((float(point[0]), float(point[1])))
    return result


def _render_mask(shape: dict[str, Any], width: int, height: int) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for part_index, part in enumerate(_parts(shape)):
        if not isinstance(part, dict):
            raise ValueError(f"part {part_index} is invalid")
        draw.polygon(_points(part.get("points"), field=f"part {part_index} points"), fill=1)
        holes = part.get("holes", [])
        if not isinstance(holes, list):
            raise ValueError(f"part {part_index} holes are invalid")
        for hole_index, hole in enumerate(holes):
            draw.polygon(
                _points(hole, field=f"part {part_index} hole {hole_index}"), fill=0
            )
    return mask


def _uncompressed_rle(mask: Image.Image) -> dict[str, Any]:
    width, height = mask.size
    pixels = mask.load()
    counts: list[int] = []
    previous = 0
    run = 0
    for x in range(width):
        for y in range(height):
            value = 1 if pixels[x, y] else 0
            if value == previous:
                run += 1
            else:
                counts.append(run)
                run = 1
                previous = value
    counts.append(run)
    return {"size": [height, width], "counts": counts}


def _review_decisions(review: dict[str, Any], candidate_ids: set[str]) -> dict[str, str]:
    if review.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError("assistant review schema differs")
    if review.get("reviewer_kind") != "independent_assistant_visual_review":
        raise ValueError("assistant reviewer kind differs")
    for field in ("human_truth", "training_authorized", "automatic_approval"):
        if review.get(field) is not False:
            raise ValueError(f"assistant review must keep {field}=false")
    rows = review.get("decisions")
    if not isinstance(rows, list):
        raise ValueError("assistant review decisions are missing")
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("assistant review row is invalid")
        candidate_id = row.get("candidate_id")
        decision = row.get("decision")
        if candidate_id not in candidate_ids:
            raise ValueError("assistant review contains an unknown candidate")
        if candidate_id in result:
            raise ValueError("assistant review contains a duplicate candidate")
        if decision not in {"pass", "reject", "uncertain"}:
            raise ValueError("assistant review decision is invalid")
        result[candidate_id] = decision
    if set(result) != candidate_ids:
        raise ValueError("assistant review must cover every candidate exactly once")
    if sum(value == "pass" for value in result.values()) != review.get(
        "assistant_pass_count"
    ):
        raise ValueError("assistant pass count differs")
    return result


def build(
    candidate_path: Path,
    review_path: Path,
    image_root: Path,
    output_dir: Path,
    *,
    padding_fraction: float = 0.12,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not 0.0 <= padding_fraction <= 0.5:
        raise ValueError("padding fraction must be between 0 and 0.5")
    candidate_path = candidate_path.resolve()
    review_path = review_path.resolve()
    image_root = image_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("output directory must be new")

    candidate_manifest = read_json(candidate_path)
    if candidate_manifest.get("schema_version") != CANDIDATE_SCHEMA:
        raise ValueError("candidate schema differs")
    if candidate_manifest.get("status") != "prepared_not_training_authorized":
        raise ValueError("candidate manifest is not in the prepared state")
    policy = candidate_manifest.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("candidate policy is missing")
    required_false = (
        "human_truth_claimed",
        "training_authorized",
        "test24_included",
        "auto_shapes_included",
        "full_frame_semantic_loss_allowed",
        "negative_sampling_allowed",
    )
    if any(policy.get(field) is not False for field in required_false):
        raise ValueError("candidate policy crossed a safety boundary")
    candidates = candidate_manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate list is empty")
    candidate_ids = {
        item.get("candidate_id") for item in candidates if isinstance(item, dict)
    }
    if None in candidate_ids or len(candidate_ids) != len(candidates):
        raise ValueError("candidate IDs are invalid or duplicated")

    review = read_json(review_path)
    if review.get("candidate_manifest_sha256") != sha256(candidate_path):
        raise ValueError("assistant review does not bind the candidate manifest")
    decisions = _review_decisions(review, candidate_ids)
    selected = [item for item in candidates if decisions[item["candidate_id"]] == "pass"]
    if not selected:
        raise ValueError("assistant review selected no candidates")

    output_dir.mkdir(parents=True)
    images_dir = output_dir / "images"
    images_dir.mkdir()
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for index, candidate in enumerate(selected, start=1):
        image = candidate.get("image")
        shape = candidate.get("shape")
        supervision = candidate.get("supervision")
        if not isinstance(image, dict) or not isinstance(shape, dict):
            raise ValueError("candidate image or shape is invalid")
        if (
            not isinstance(supervision, dict)
            or supervision.get("positive_instance_only") is not True
            or supervision.get("outside_instance") != "ignore"
            or supervision.get("negative_sampling_allowed") is not False
        ):
            raise ValueError("candidate supervision policy differs")
        if shape.get("label") != "pallet":
            raise ValueError("only pallet candidates are supported")
        source_name = image.get("file")
        if not isinstance(source_name, str) or Path(source_name).name != source_name:
            raise ValueError("candidate image filename is unsafe")
        source_path = image_root / source_name
        if not source_path.is_file() or sha256(source_path) != image.get("sha256"):
            raise ValueError("candidate source image hash differs")

        with Image.open(source_path) as opened:
            rgb = opened.convert("RGB")
        mask = _render_mask(shape, *rgb.size)
        bbox = mask.getbbox()
        if bbox is None:
            raise ValueError("candidate mask is empty")
        left, top, right, bottom = bbox
        span = max(right - left, bottom - top)
        padding = max(2, round(span * padding_fraction))
        crop_box = (
            max(0, left - padding),
            max(0, top - padding),
            min(rgb.width, right + padding),
            min(rgb.height, bottom + padding),
        )
        crop_image = rgb.crop(crop_box)
        crop_mask = mask.crop(crop_box)
        crop_mask_bbox = crop_mask.getbbox()
        if crop_mask_bbox is None:
            raise ValueError("cropped candidate mask is empty")
        x0, y0, x1, y1 = crop_mask_bbox
        area = crop_mask.histogram()[1]
        filename = f"assistant_pallet_{index:03d}.png"
        target = images_dir / filename
        crop_image.save(target, format="PNG", optimize=True)
        coco_images.append(
            {
                "id": index,
                "file_name": f"images/{filename}",
                "width": crop_image.width,
                "height": crop_image.height,
                "candidate_id": candidate["candidate_id"],
                "source_image_sha256": image["sha256"],
                "crop_image_sha256": sha256(target),
            }
        )
        coco_annotations.append(
            {
                "id": index,
                "image_id": index,
                "category_id": 1,
                "segmentation": _uncompressed_rle(crop_mask),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": area,
                "iscrowd": 0,
            }
        )
        selected_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "source_image_sha256": image["sha256"],
                "crop_image": f"images/{filename}",
                "crop_image_sha256": sha256(target),
                "crop_box_xyxy": list(crop_box),
                "mask_area": area,
            }
        )

    coco = {
        "info": {
            "description": "Assistant-delegated pallet instance crop candidate set",
            "assistant_reviewed": True,
            "human_truth": False,
            "training_authorized": False,
            "test24_included": False,
            "production_deployed": False,
            "supervision": "positive_instance_crop_only",
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": 1, "name": "pallet", "supercategory": "pallet"}],
    }
    coco_path = output_dir / "_assistant_delegated_pallet_crops.coco.json"
    coco_path.write_text(
        json.dumps(coco, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "labelagent.sam3-assistant-instance-crop-dataset.v1",
        "status": "ready_for_isolated_candidate_training",
        "candidate_manifest_sha256": sha256(candidate_path),
        "assistant_review_sha256": sha256(review_path),
        "output_coco": coco_path.name,
        "output_coco_sha256": sha256(coco_path),
        "candidate_count": len(candidates),
        "assistant_pass_count": len(selected),
        "assistant_uncertain_count": sum(
            value == "uncertain" for value in decisions.values()
        ),
        "assistant_reject_count": sum(value == "reject" for value in decisions.values()),
        "crop_count": len(coco_images),
        "padding_fraction": padding_fraction,
        "human_truth": False,
        "training_authorized": False,
        "test24_included": False,
        "production_deployed": False,
        "auto_deploy": False,
        "selected": selected_rows,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    receipt = {
        key: manifest[key]
        for key in (
            "schema_version",
            "status",
            "candidate_manifest_sha256",
            "assistant_review_sha256",
            "output_coco_sha256",
            "candidate_count",
            "assistant_pass_count",
            "assistant_uncertain_count",
            "assistant_reject_count",
            "crop_count",
            "padding_fraction",
            "human_truth",
            "training_authorized",
            "test24_included",
            "production_deployed",
            "auto_deploy",
        )
    }
    receipt["private_candidate_ids_published"] = False
    receipt["raw_images_published"] = False
    return manifest, receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--assistant-review", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    parser.add_argument("--padding-fraction", type=float, default=0.12)
    args = parser.parse_args()
    if args.receipt_output.exists():
        raise ValueError("receipt output must be new")
    _, receipt = build(
        args.candidates,
        args.assistant_review,
        args.image_root,
        args.output_dir,
        padding_fraction=args.padding_fraction,
    )
    args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


