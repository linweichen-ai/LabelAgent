#!/usr/bin/env python3
"""Audit strict ego union candidates without modifying masks or truth.

For every sample, a leave-one-package-out reference is built from the other
packages of the same camera.  Metrics and red/yellow/green difference images
are review aids only; they never approve training or production deployment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from tools.build_ego_component_union_review import (
        _verified_bytes, file_sha256, load_identity_manifest)
except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
    from build_ego_component_union_review import (  # type: ignore
        _verified_bytes, file_sha256, load_identity_manifest)


DEFAULT_THRESHOLDS = {
    "camera1": {
        "min_reference_iou": 0.93,
        "min_reference_recall": 0.95,
        "max_candidate_excess": 0.05,
        "max_small_island_ratio": 0.01,
        "max_hole_ratio": 0.06,
    },
    "camera3": {
        "min_reference_iou": 0.80,
        "min_reference_recall": 0.86,
        "max_candidate_excess": 0.16,
        "max_small_island_ratio": 0.01,
        "max_hole_ratio": 0.06,
    },
}


def _sample_parts(sample_id: str) -> tuple[str, str, str]:
    match = re.fullmatch(r"(\d{3})-(camera[13])-(\d+)", sample_id)
    if match is None:
        raise ValueError(f"invalid sample identity: {sample_id}")
    return match.group(1), match.group(2), match.group(3)


def _component_sizes(mask: np.ndarray) -> list[tuple[int, bool]]:
    """Return 8-connected component sizes and border-touch flags."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    border_labels = set(np.concatenate((labels[0, :], labels[-1, :],
                                        labels[:, 0], labels[:, -1])).tolist())
    return [(int(stats[index, cv2.CC_STAT_AREA]), index in border_labels)
            for index in range(1, count)]


def mask_metrics(candidate: np.ndarray, reference: np.ndarray,
                 small_island_pixels: int = 100) -> dict:
    if candidate.shape != reference.shape or candidate.ndim != 2:
        raise ValueError("candidate and reference masks must share a 2D shape")
    candidate = candidate.astype(bool)
    reference = reference.astype(bool)
    candidate_area = int(candidate.sum())
    reference_area = int(reference.sum())
    if not candidate_area or not reference_area:
        raise ValueError("candidate and reference masks must be non-empty")
    intersection = int(np.logical_and(candidate, reference).sum())
    union = int(np.logical_or(candidate, reference).sum())
    excess = int(np.logical_and(candidate, ~reference).sum())
    missing = int(np.logical_and(reference, ~candidate).sum())
    components = _component_sizes(candidate)
    small_area = sum(size for size, _ in components if size < small_island_pixels)
    background = _component_sizes(~candidate)
    hole_sizes = [size for size, touches_border in background if not touches_border]
    vertical = int(np.not_equal(candidate[1:, :], candidate[:-1, :]).sum())
    horizontal = int(np.not_equal(candidate[:, 1:], candidate[:, :-1]).sum())
    perimeter = (vertical + horizontal + int(candidate[0, :].sum())
                 + int(candidate[-1, :].sum()) + int(candidate[:, 0].sum())
                 + int(candidate[:, -1].sum()))
    return {
        "candidate_area_pixels": candidate_area,
        "reference_area_pixels": reference_area,
        "intersection_pixels": intersection,
        "missing_pixels": missing,
        "excess_pixels": excess,
        "reference_iou": round(intersection / union, 6),
        "reference_recall": round(intersection / reference_area, 6),
        "candidate_excess": round(excess / candidate_area, 6),
        "component_count": len(components),
        "main_component_count": sum(size >= small_island_pixels for size, _ in components),
        "small_island_count": sum(size < small_island_pixels for size, _ in components),
        "small_island_area_ratio": round(small_area / candidate_area, 6),
        "hole_count": len(hole_sizes),
        "hole_area_ratio": round(sum(hole_sizes) / candidate_area, 6),
        "grid_perimeter_pixels": perimeter,
        "perimeter_area_ratio": round(perimeter / candidate_area, 6),
    }


def threshold_alerts(metrics: dict, camera: str,
                     thresholds: dict[str, dict[str, float]]) -> list[str]:
    rule = thresholds[camera]
    checks = (
        (metrics["reference_iou"] < rule["min_reference_iou"], "reference_iou_low"),
        (metrics["reference_recall"] < rule["min_reference_recall"],
         "reference_recall_low"),
        (metrics["candidate_excess"] > rule["max_candidate_excess"],
         "candidate_excess_high"),
        (metrics["small_island_area_ratio"] > rule["max_small_island_ratio"],
         "small_island_ratio_high"),
        (metrics["hole_area_ratio"] > rule["max_hole_ratio"], "hole_ratio_high"),
    )
    return [name for failed, name in checks if failed]


def _decode_mask(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as decoded:
        return np.asarray(decoded.convert("L")) > 0


def _write_difference(image_bytes: bytes, candidate: np.ndarray,
                      reference: np.ndarray, target: Path) -> None:
    with Image.open(io.BytesIO(image_bytes)) as decoded:
        base = decoded.convert("RGBA")
    if base.size != (candidate.shape[1], candidate.shape[0]):
        raise ValueError("source image and union mask dimensions differ")
    overlap = candidate & reference
    missing = reference & ~candidate
    excess = candidate & ~reference
    rgba = np.zeros((*candidate.shape, 4), dtype=np.uint8)
    rgba[overlap] = (24, 206, 138, 105)
    rgba[missing] = (250, 190, 40, 150)
    rgba[excess] = (238, 70, 70, 170)
    rendered = Image.alpha_composite(base, Image.fromarray(rgba, mode="RGBA"))
    rendered.convert("RGB").save(target, "JPEG", quality=90)


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
        with Image.open(output / row["difference_image"]) as image:
            preview = image.convert("RGB")
            preview.thumbnail((cell_width, image_height))
        canvas.paste(preview, (left + (cell_width - preview.width) // 2,
                               top + (image_height - preview.height) // 2))
        draw.rectangle((left, top + image_height,
                        left + cell_width, top + image_height + label_height),
                       fill=(25, 35, 45))
        color = (244, 130, 90) if row["threshold_alerts"] else (83, 210, 161)
        draw.text((left + 5, top + image_height + 5), row["sample_id"],
                  fill=color, font=font)
        draw.text((left + 5, top + image_height + 22),
                  (f"IoU {row['metrics']['reference_iou']:.3f} | "
                   f"miss {1-row['metrics']['reference_recall']:.3f} | "
                   f"extra {row['metrics']['candidate_excess']:.3f}"),
                  fill=color, font=font)
    path = output / "gap_excess_contact_sheet.jpg"
    canvas.save(path, "JPEG", quality=88)
    return path


def build_audit(review_queues: list[Path], identity_manifest_path: Path,
                output: Path, thresholds: dict | None = None,
                small_island_pixels: int = 100) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if len(review_queues) < 2:
        raise ValueError("at least two strict review queues are required")
    thresholds = thresholds or DEFAULT_THRESHOLDS
    identities = load_identity_manifest(identity_manifest_path)
    samples = []
    seen = set()
    for queue_path in review_queues:
        packet = json.loads(queue_path.read_text(encoding="utf-8"))
        if (packet.get("candidate_only") is not True
                or packet.get("human_truth") is not False
                or packet.get("approved_for_training") is not False
                or packet.get("production_deployed") is not False):
            raise ValueError("review queue safety flags are inconsistent")
        gates = packet.get("safety_gates") or {}
        if (not gates.get("allowed_source_root")
                or gates.get("component_area_guard_enabled") is not True
                or gates.get("identity_manifest_enabled") is not True
                or gates.get("identity_manifest_sha256")
                != file_sha256(identity_manifest_path)):
            raise ValueError("review queue was not produced by all strict safety gates")
        for row in packet.get("samples", []):
            sample_id = row.get("sample_id")
            if sample_id in seen:
                raise ValueError(f"duplicate sample identity: {sample_id}")
            seen.add(sample_id)
            package, camera, frame_id = _sample_parts(sample_id)
            identity = identities.get(sample_id)
            if identity is None or identity != {
                    "package": package, "camera": camera, "frame_id": frame_id,
                    "image_sha256": row.get("source_image_sha256")}:
                raise ValueError(f"identity binding mismatch: {sample_id}")
            queue_dir = queue_path.parent
            mask_path = queue_dir / row.get("union_mask", "")
            mask_bytes = _verified_bytes(
                mask_path, row.get("union_mask_sha256"), "union mask")
            image_path = Path(row.get("source_image", ""))
            image_bytes = _verified_bytes(
                image_path, row.get("source_image_sha256"), "source image")
            samples.append({
                "sample_id": sample_id, "package": package, "camera": camera,
                "frame_id": frame_id, "mask": _decode_mask(mask_bytes),
                "image_bytes": image_bytes,
                "source_image_sha256": row["source_image_sha256"],
                "union_mask_sha256": row["union_mask_sha256"],
                "source_queue": str(queue_path.resolve()),
                "source_queue_sha256": file_sha256(queue_path),
            })
    packages = {sample["package"] for sample in samples}
    if len(packages) < 3:
        raise ValueError("at least three packages are required for leave-one-package-out audit")
    output.mkdir(parents=True)
    rows = []
    for sample in sorted(samples, key=lambda item: item["sample_id"]):
        external = [other["mask"] for other in samples
                    if other["camera"] == sample["camera"]
                    and other["package"] != sample["package"]]
        external_packages = {other["package"] for other in samples
                             if other["camera"] == sample["camera"]
                             and other["package"] != sample["package"]}
        if len(external_packages) < 2 or len(external) < 4:
            raise ValueError(
                f"insufficient external-package reference masks: {sample['sample_id']}")
        shape = sample["mask"].shape
        if any(mask.shape != shape for mask in external):
            raise ValueError("reference mask dimensions differ")
        votes = np.sum(np.stack(external, axis=0), axis=0)
        required_votes = (2 * len(external) + 2) // 3
        reference = votes >= required_votes
        metrics = mask_metrics(sample["mask"], reference, small_island_pixels)
        alerts = threshold_alerts(metrics, sample["camera"], thresholds)
        difference_name = f"{sample['sample_id']}-gap-excess.jpg"
        _write_difference(sample["image_bytes"], sample["mask"], reference,
                          output / difference_name)
        rows.append({
            "sample_id": sample["sample_id"],
            "package": sample["package"],
            "camera": sample["camera"],
            "frame_id": sample["frame_id"],
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "production_deployed": False,
            "review_status": "high_priority_review" if alerts else "review_required",
            "threshold_alerts": alerts,
            "external_package_count": len(external_packages),
            "external_mask_count": len(external),
            "reference_required_votes": required_votes,
            "source_image_sha256": sample["source_image_sha256"],
            "union_mask_sha256": sample["union_mask_sha256"],
            "source_queue": sample["source_queue"],
            "source_queue_sha256": sample["source_queue_sha256"],
            "metrics": metrics,
            "difference_image": difference_name,
            "difference_image_sha256": file_sha256(output / difference_name),
            "assistant_review": {
                "external_swallow": "unreviewed",
                "component_gap": "unreviewed",
                "occlusion_uncertain": True,
                "review_note": "",
            },
        })
    contact = _write_contact_sheet(output, rows)
    audit = {
        "schema_version": "labelagent.ego_union_pixel_audit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "production_deployed": False,
        "formal_state_modified": False,
        "sample_count": len(rows),
        "threshold_alert_count": sum(bool(row["threshold_alerts"]) for row in rows),
        "thresholds_are_review_triage_only": True,
        "thresholds": thresholds,
        "small_island_pixels": small_island_pixels,
        "identity_manifest_sha256": file_sha256(identity_manifest_path),
        "contact_sheet": contact.name,
        "contact_sheet_sha256": file_sha256(contact),
        "samples": rows,
        "next_step": (
            "Review every difference image. Metrics can prioritize review but cannot "
            "approve truth, training or deployment."),
    }
    target = output / "audit.json"
    target.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", action="append", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--small-island-pixels", type=int, default=100)
    arguments = parser.parse_args()
    if arguments.small_island_pixels <= 0:
        parser.error("--small-island-pixels must be positive")
    audit = build_audit(arguments.review_queue, arguments.identity_manifest,
                        arguments.output,
                        small_island_pixels=arguments.small_island_pixels)
    print(json.dumps({
        "output": str((arguments.output / "audit.json").resolve()),
        "sample_count": audit["sample_count"],
        "threshold_alert_count": audit["threshold_alert_count"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
