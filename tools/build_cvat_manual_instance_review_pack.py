"""Render a local-only review pack for manual CVAT instance candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_geometry_sha(shape: dict[str, Any]) -> str:
    payload = {key: shape.get(key, [] if key != "parts" else [])
               for key in ("points", "holes", "parts")}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evaluation_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"evaluation row {line_number} is not an object")
        if (value.get("split") != "test24" or value.get("test24") is not True
                or value.get("evaluation_only") is not True
                or value.get("training_return_allowed") is not False):
            raise ValueError(f"evaluation row {line_number} violates frozen test24 policy")
        digest = value.get("image_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"evaluation row {line_number} has invalid image hash")
        if digest in hashes:
            raise ValueError(f"evaluation row {line_number} repeats an image hash")
        hashes.add(digest)
    if not hashes:
        raise ValueError("frozen test24 list is empty")
    return hashes


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _points(shape: dict[str, Any]) -> list[tuple[float, float]]:
    parts = shape.get("parts")
    rows = parts if isinstance(parts, list) and parts else [shape]
    points: list[tuple[float, float]] = []
    for part in rows:
        if not isinstance(part, dict):
            continue
        for point in part.get("points") or []:
            if isinstance(point, list) and len(point) == 2:
                points.append((float(point[0]), float(point[1])))
    return points


def _crop_box(shape: dict[str, Any], image: Image.Image) -> tuple[int, int, int, int]:
    points = _points(shape)
    if len(points) < 3:
        raise ValueError("candidate shape has fewer than three points")
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    pad = max(24, round(span * 0.18))
    return (
        max(0, math.floor(min(xs)) - pad),
        max(0, math.floor(min(ys)) - pad),
        min(image.width, math.ceil(max(xs)) + pad + 1),
        min(image.height, math.ceil(max(ys)) + pad + 1),
    )


def _overlay(image: Image.Image, shape: dict[str, Any]) -> Image.Image:
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    parts = shape.get("parts")
    rows = parts if isinstance(parts, list) and parts else [shape]
    for part in rows:
        if not isinstance(part, dict):
            continue
        outer = [tuple(map(float, point)) for point in (part.get("points") or [])]
        if len(outer) >= 3:
            draw.polygon(outer, fill=(30, 210, 145, 70), outline=(255, 146, 43, 255), width=2)
        for hole in part.get("holes") or []:
            ring = [tuple(map(float, point)) for point in hole]
            if len(ring) >= 3:
                draw.line(ring + [ring[0]], fill=(255, 72, 72, 255), width=1)
    return Image.alpha_composite(base, layer).convert("RGB")


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (18, 27, 34))
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def build_pack(
    candidate_manifest: Path,
    candidate_receipt: Path,
    frozen_evaluation_jsonl: Path,
    image_root: Path,
    output_dir: Path,
    *,
    per_page: int = 6,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_manifest = candidate_manifest.resolve()
    candidate_receipt = candidate_receipt.resolve()
    frozen_evaluation_jsonl = frozen_evaluation_jsonl.resolve()
    image_root = image_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("choose a fresh review output directory")
    if per_page < 1 or per_page > 12:
        raise ValueError("per_page must be between 1 and 12")
    value = _read(candidate_manifest)
    receipt_value = _read(candidate_receipt)
    candidate_digest = sha256(candidate_manifest)
    if receipt_value.get("schema_version") != "labelagent.cvat-manual-instance-candidate-receipt.v1":
        raise ValueError("candidate receipt schema differs")
    if receipt_value.get("status") != "prepared_not_training_authorized":
        raise ValueError("candidate receipt status differs")
    if receipt_value.get("errors") != []:
        raise ValueError("candidate receipt contains errors")
    if receipt_value.get("private_manifest_sha256") != candidate_digest:
        raise ValueError("candidate manifest hash differs from receipt")
    if value.get("schema_version") != "labelagent.cvat-manual-instance-candidates.v1":
        raise ValueError("candidate manifest schema differs")
    if value.get("status") != "prepared_not_training_authorized" or value.get("errors") != []:
        raise ValueError("candidate manifest is not a clean prepared candidate set")
    policy = value.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("candidate policy is absent")
    required = {
        "training_authorized": False,
        "test24_included": False,
        "auto_shapes_included": False,
        "outside_manual_instance": "ignore",
        "full_frame_semantic_loss_allowed": False,
        "negative_sampling_allowed": False,
    }
    for key, expected in required.items():
        if policy.get(key) != expected:
            raise ValueError(f"candidate policy differs: {key}")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate list is absent")

    expected_summary = receipt_value.get("summary")
    if not isinstance(expected_summary, dict):
        raise ValueError("candidate receipt summary is absent")
    if (expected_summary.get("candidate_count") != len(candidates)
            or expected_summary.get("frame_count") != 81
            or expected_summary.get("verified_image_count") != 81
            or expected_summary.get("label_counts") != {"pallet": 81}
            or len(candidates) != 81):
        raise ValueError("candidate inventory differs from the locked 81-instance set")

    cards: list[tuple[str, Image.Image]] = []
    item_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_frames: set[int] = set()
    candidate_image_hashes: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("candidate is not an object")
        candidate_id = candidate.get("candidate_id")
        image_meta = candidate.get("image")
        shape = candidate.get("shape")
        supervision = candidate.get("supervision")
        if not isinstance(candidate_id, str) or candidate_id in seen:
            raise ValueError("candidate id is invalid or duplicated")
        seen.add(candidate_id)
        if not isinstance(image_meta, dict) or not isinstance(shape, dict):
            raise ValueError(f"candidate {candidate_id}: image or shape is absent")
        metadata = shape.get("cvat_metadata")
        if not isinstance(metadata, dict) or metadata.get("source") != "manual":
            raise ValueError(f"candidate {candidate_id}: source is not manual")
        if shape.get("label") != "pallet":
            raise ValueError(f"candidate {candidate_id}: label is not pallet")
        frame_id = candidate.get("frame_id")
        shape_id = shape.get("id")
        if (not isinstance(frame_id, int) or not isinstance(shape_id, int)
                or candidate_id != f"{frame_id}:{shape_id}"):
            raise ValueError(f"candidate {candidate_id}: identity binding differs")
        if frame_id in seen_frames:
            raise ValueError(f"candidate {candidate_id}: frame is duplicated")
        seen_frames.add(frame_id)
        if not isinstance(supervision, dict) or supervision.get("outside_instance") != "ignore":
            raise ValueError(f"candidate {candidate_id}: supervision policy differs")
        filename = image_meta.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"candidate {candidate_id}: unsafe image filename")
        image_path = image_root / filename
        if not image_path.is_file() or sha256(image_path) != image_meta.get("sha256"):
            raise ValueError(f"candidate {candidate_id}: image is absent or hash differs")
        candidate_image_hashes.add(image_meta["sha256"])
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        crop_box = _crop_box(shape, image)
        original = _fit(image.crop(crop_box), (540, 280))
        overlay = _fit(_overlay(image, shape).crop(crop_box), (540, 280))
        card = Image.new("RGB", (1080, 312), "white")
        card.paste(original, (0, 32))
        card.paste(overlay, (540, 32))
        draw = ImageDraw.Draw(card)
        draw.text((8, 8), f"{candidate_id}  {shape.get('label')}  source=manual", fill="black", font=ImageFont.load_default())
        draw.text((548, 8), "thin outline: review only; no automatic approval", fill="black", font=ImageFont.load_default())
        cards.append((candidate_id, card))

        item_rows.append({
            "candidate_id": candidate_id,
            "frame_id": frame_id,
            "source_image_sha256": image_meta["sha256"],
            "canonical_geometry_sha256": _canonical_geometry_sha(shape),
        })

    if len(seen_frames) != 81 or len(candidate_image_hashes) != 81:
        raise ValueError("candidate frames or image hashes are not uniquely 81")
    test24_hashes = _evaluation_hashes(frozen_evaluation_jsonl)
    overlap = sorted(candidate_image_hashes & test24_hashes)
    if overlap:
        raise ValueError(f"candidate images overlap frozen test24: {len(overlap)}")

    output_dir.mkdir(parents=True)
    item_dir = output_dir / "items"
    item_dir.mkdir()
    for index, ((candidate_id, card), item) in enumerate(zip(cards, item_rows), 1):
        item_path = item_dir / f"review_item_{index:03d}.jpg"
        card.save(item_path, quality=93, optimize=True)
        item["focused_image_file"] = f"items/{item_path.name}"
        item["focused_image_sha256"] = sha256(item_path)

    pages: list[dict[str, Any]] = []
    for offset in range(0, len(cards), per_page):
        batch = cards[offset:offset + per_page]
        page = Image.new("RGB", (1080, 312 * len(batch)), (238, 243, 245))
        ids: list[str] = []
        for row, (candidate_id, card) in enumerate(batch):
            page.paste(card, (0, row * 312))
            ids.append(candidate_id)
        path = output_dir / f"review_page_{len(pages) + 1:02d}.jpg"
        page.save(path, quality=91, optimize=True)
        pages.append({"file": path.name, "sha256": sha256(path), "candidate_ids": ids})

    review_manifest = {
        "schema_version": "labelagent.cvat-manual-instance-review-pack.v1",
        "status": "pending_independent_review",
        "candidate_manifest_sha256": sha256(candidate_manifest),
        "candidate_receipt_sha256": sha256(candidate_receipt),
        "frozen_test24_manifest_sha256": sha256(frozen_evaluation_jsonl),
        "frozen_test24_image_count": len(test24_hashes),
        "candidate_image_count": len(candidate_image_hashes),
        "test24_intersection_count": 0,
        "candidate_count": len(cards),
        "page_count": len(pages),
        "line_width_pixels": 2,
        "review_fields": ["is_pallet", "contour_complete", "no_merge", "no_major_omission"],
        "reviewer_decisions": [],
        "training_authorized": False,
        "automatic_approval": False,
        "candidate_only": True,
        "human_truth": False,
        "production_deployed": False,
        "review_items": item_rows,
        "pages": pages,
    }
    review_path = output_dir / "review_manifest.json"
    review_path.write_text(json.dumps(review_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    aggregate = hashlib.sha256("".join(page["sha256"] for page in pages).encode("ascii")).hexdigest()
    receipt = {
        "schema_version": "labelagent.cvat-manual-instance-review-receipt.v1",
        "status": "pending_independent_review",
        "candidate_manifest_sha256": sha256(candidate_manifest),
        "candidate_receipt_sha256": sha256(candidate_receipt),
        "frozen_test24_manifest_sha256": sha256(frozen_evaluation_jsonl),
        "frozen_test24_image_count": len(test24_hashes),
        "candidate_image_count": len(candidate_image_hashes),
        "test24_intersection_count": 0,
        "review_manifest_sha256": sha256(review_path),
        "page_aggregate_sha256": aggregate,
        "candidate_count": len(cards),
        "page_count": len(pages),
        "line_width_pixels": 2,
        "training_authorized": False,
        "automatic_approval": False,
        "candidate_only": True,
        "human_truth": False,
        "production_deployed": False,
        "private_review_images_published": False,
        "private_geometry_published": False,
        "raw_images_published": False,
    }
    return review_manifest, receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--candidate-receipt", required=True, type=Path)
    parser.add_argument("--frozen-evaluation-jsonl", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    parser.add_argument("--per-page", type=int, default=6)
    args = parser.parse_args()
    if args.receipt_output.exists():
        raise ValueError("choose a fresh receipt output")
    _, receipt = build_pack(
        args.candidates, args.candidate_receipt, args.frozen_evaluation_jsonl,
        args.image_root, args.output_dir, per_page=args.per_page)
    args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
