#!/usr/bin/env python3
"""Review whether bound historical crop proposals recover partial pallets.

The tool is deliberately read-only.  It selects the highest-change small
pallet rows from a full geometry audit, verifies that every augmented frame
retains the complete baseline prefix, and ranks only the appended historical
SAM crop proposals.  It never calls a model or writes workbench candidates,
human truth, or training approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from source.agent.edit_agent import _rasterize


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(payload: dict) -> dict:
    nested = payload.get("candidates_by_topic")
    return nested if isinstance(nested, dict) else payload


def _camera_name(topic: str) -> str:
    match = re.search(r"/(camera_\d+)(?:/|$)", str(topic))
    if not match:
        raise ValueError(f"cannot map topic to camera directory: {topic}")
    return match.group(1)


def _mask_record(shape: dict, width: int, height: int) -> dict:
    mask = _rasterize(shape, width, height).astype(bool)
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError(f"empty shape: {shape.get('id')}")
    return {
        "mask": mask,
        "area": int(len(rows)),
        "bbox": (int(columns.min()), int(rows.min()),
                 int(columns.max()) + 1, int(rows.max()) + 1),
    }


def expansion_metrics(baseline: dict, proposal: dict, width: int,
                      height: int) -> dict:
    first = _mask_record(baseline, width, height)
    second = _mask_record(proposal, width, height)
    intersection = int(np.count_nonzero(first["mask"] & second["mask"]))
    union = first["area"] + second["area"] - intersection
    rows, columns = np.nonzero(first["mask"])
    points = np.column_stack([columns, rows]).astype(float)
    center = points.mean(axis=0)
    covariance = np.cov(points - center, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    def axis_extents(mask):
        ys, xs = np.nonzero(mask)
        projected = (np.column_stack([xs, ys]) - center) @ eigenvectors
        return projected.max(axis=0) - projected.min(axis=0)

    baseline_extents = axis_extents(first["mask"])
    proposal_extents = axis_extents(second["mask"])
    minor = max(float(baseline_extents[1]), 1e-6)
    return {
        "baseline_area": first["area"],
        "proposal_area": second["area"],
        "intersection_pixels": intersection,
        "baseline_coverage": intersection / first["area"],
        "proposal_covered_by_baseline": intersection / second["area"],
        "proposal_to_baseline_area_ratio": second["area"] / first["area"],
        "iou": intersection / max(union, 1),
        "baseline_axis_elongation": float(baseline_extents[0]) / minor,
        "proposal_major_axis_ratio": (
            float(proposal_extents[0]) / max(float(baseline_extents[0]), 1e-6)),
        "proposal_minor_axis_ratio": float(proposal_extents[1]) / minor,
        "baseline_axis_extents": [float(value) for value in baseline_extents],
        "proposal_axis_extents": [float(value) for value in proposal_extents],
        "baseline_axis_eigenvalues": [float(value) for value in eigenvalues],
        "baseline_bbox": list(first["bbox"]),
        "proposal_bbox": list(second["bbox"]),
        "_baseline_mask": first["mask"],
        "_proposal_mask": second["mask"],
    }


def match_expansion(baseline: dict, proposals: list[dict], width: int,
                    height: int, *, minimum_baseline_coverage: float = .5,
                    minimum_area_ratio: float = 1.25,
                    maximum_minor_axis_ratio: float | None = None,
                    minimum_axis_elongation: float = 2.0,
                    other_baselines: list[dict] | None = None,
                    maximum_other_baseline_fraction: float | None = None,
                    ) -> tuple[dict | None, dict | None]:
    if not 0 < minimum_baseline_coverage <= 1:
        raise ValueError("minimum baseline coverage must be in (0, 1]")
    if not math.isfinite(minimum_area_ratio) or minimum_area_ratio <= 1:
        raise ValueError("minimum area ratio must be greater than 1")
    if (maximum_minor_axis_ratio is not None
            and (not math.isfinite(maximum_minor_axis_ratio)
                 or maximum_minor_axis_ratio <= 1)):
        raise ValueError("maximum minor axis ratio must be greater than 1")
    if not math.isfinite(minimum_axis_elongation) or minimum_axis_elongation < 1:
        raise ValueError("minimum axis elongation must be at least 1")
    if (maximum_other_baseline_fraction is not None
            and (not math.isfinite(maximum_other_baseline_fraction)
                 or not 0 <= maximum_other_baseline_fraction <= 1)):
        raise ValueError("maximum other baseline fraction must be in [0, 1]")
    ranked = []
    for proposal in proposals:
        if str(proposal.get("label", "")).casefold() != "pallet":
            continue
        metrics = expansion_metrics(baseline, proposal, width, height)
        proposal_mask = metrics["_proposal_mask"]
        metrics["maximum_other_baseline_fraction"] = max((
            int(np.count_nonzero(
                proposal_mask & _mask_record(other, width, height)["mask"]))
            / metrics["proposal_area"]
            for other in (other_baselines or [])
            if str(other.get("label", "")).casefold() == "pallet"
        ), default=0.0)
        if (metrics["baseline_coverage"] < minimum_baseline_coverage
                or metrics["proposal_to_baseline_area_ratio"] < minimum_area_ratio):
            continue
        if (maximum_minor_axis_ratio is not None
                and (metrics["baseline_axis_elongation"] < minimum_axis_elongation
                     or metrics["proposal_minor_axis_ratio"]
                     > maximum_minor_axis_ratio)):
            continue
        if (maximum_other_baseline_fraction is not None
                and metrics["maximum_other_baseline_fraction"]
                > maximum_other_baseline_fraction):
            continue
        score = float(proposal.get("score", 0) or 0)
        ranked.append((metrics["baseline_coverage"],
                       metrics["proposal_to_baseline_area_ratio"], score,
                       str(proposal.get("id")), proposal, metrics))
    if not ranked:
        return None, None
    ranked.sort(key=lambda item: item[:4], reverse=True)
    return ranked[0][4], ranked[0][5]


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, (19, 27, 34))
    panel.paste(fitted, ((size[0] - fitted.width) // 2,
                         (size[1] - fitted.height) // 2))
    return panel


def _mask_overlay(image: Image.Image, mask: np.ndarray,
                  color: tuple[int, int, int]) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    selected = mask.astype(bool)
    tint = np.asarray(color, dtype=np.float32)
    base[selected] = np.round(base[selected] * .55 + tint * .45).astype(np.uint8)
    return Image.fromarray(base, "RGB")


def _difference_overlay(image: Image.Image, baseline: np.ndarray,
                        proposal: np.ndarray) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    added = proposal & ~baseline
    missing = baseline & ~proposal
    retained = baseline & proposal
    for selected, color in ((retained, (35, 205, 130)),
                            (added, (255, 190, 35)),
                            (missing, (255, 70, 70))):
        tint = np.asarray(color, dtype=np.float32)
        base[selected] = np.round(base[selected] * .45 + tint * .55).astype(np.uint8)
    return Image.fromarray(base, "RGB")


def _crop(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(mask_a | mask_b)
    if not len(rows):
        raise ValueError("cannot crop empty masks")
    height, width = mask_a.shape
    span = max(int(columns.max() - columns.min()), int(rows.max() - rows.min()))
    pad = max(24, round(span * .2))
    return (max(0, int(columns.min()) - pad),
            max(0, int(rows.min()) - pad),
            min(width, int(columns.max()) + pad + 1),
            min(height, int(rows.max()) + pad + 1))


def _public_metrics(metrics: dict | None) -> dict | None:
    if metrics is None:
        return None
    return {
        key: (round(value, 6) if isinstance(value, float) else value)
        for key, value in metrics.items() if not key.startswith("_")
    }


def review(baseline_path: Path, audit_path: Path,
           augmented_paths: list[Path], image_root: Path, output: Path, *,
           limit: int = 16, width: int = 1920, height: int = 1536,
           minimum_baseline_coverage: float = .5,
           minimum_area_ratio: float = 1.25,
           maximum_minor_axis_ratio: float | None = None,
           minimum_axis_elongation: float = 2.0,
           maximum_other_baseline_fraction: float | None = None) -> dict:
    if limit <= 0:
        raise ValueError("limit must be positive")
    inputs = [baseline_path, audit_path, *augmented_paths]
    before = {str(path.resolve()): _sha256(path) for path in inputs}
    baseline = _mapping(json.loads(baseline_path.read_text(encoding="utf-8")))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_rows = audit.get("results")
    if not isinstance(audit_rows, list):
        raise ValueError("audit report has no results list")

    augmented_by_frame: dict[tuple[str, str], tuple[list[dict], str]] = {}
    for augmented_path in augmented_paths:
        payload = _mapping(json.loads(augmented_path.read_text(encoding="utf-8")))
        for topic, frames in payload.items():
            for frame, shapes in frames.items():
                key = (str(topic), str(frame))
                if key in augmented_by_frame:
                    raise ValueError(f"duplicate augmented frame: {key}")
                base_shapes = baseline.get(topic, {}).get(str(frame))
                if base_shapes is None:
                    raise ValueError(f"augmented frame absent from baseline: {key}")
                if shapes[:len(base_shapes)] != base_shapes:
                    raise ValueError(f"augmented frame does not preserve baseline prefix: {key}")
                augmented_by_frame[key] = (shapes[len(base_shapes):],
                                            str(augmented_path.resolve()))

    ranked = sorted(
        (row for row in audit_rows
         if row.get("size_band") in {"small", "tiny_skipped"}
         and float(row.get("change_ratio") or 0) > 0),
        key=lambda row: (-float(row.get("change_ratio") or 0),
                         str(row.get("topic")), int(row.get("frame", -1)),
                         str(row.get("shape_id"))))[:limit]

    inventories: dict[str, list[Path]] = {}
    panels = []
    public_rows = []
    font = ImageFont.load_default()
    for rank, row in enumerate(ranked, start=1):
        topic, frame = str(row["topic"]), str(row["frame"])
        shapes = baseline.get(topic, {}).get(frame)
        if shapes is None:
            raise ValueError(f"audit frame absent from baseline: {(topic, frame)}")
        matching = [shape for shape in shapes
                    if str(shape.get("id")) == str(row.get("shape_id"))]
        if len(matching) != 1:
            raise ValueError(f"audit shape identity is not unique: {(topic, frame, row.get('shape_id'))}")
        additions, source = augmented_by_frame.get((topic, frame), ([], None))
        proposal, metrics = match_expansion(
            matching[0], additions, width, height,
            minimum_baseline_coverage=minimum_baseline_coverage,
            minimum_area_ratio=minimum_area_ratio,
            maximum_minor_axis_ratio=maximum_minor_axis_ratio,
            minimum_axis_elongation=minimum_axis_elongation,
            other_baselines=[
                shape for shape in shapes
                if str(shape.get("id")) != str(row.get("shape_id"))],
            maximum_other_baseline_fraction=maximum_other_baseline_fraction)
        camera = _camera_name(topic)
        if camera not in inventories:
            inventories[camera] = sorted((image_root / camera).glob("*.png"))
        images = inventories[camera]
        index = int(frame)
        if index < 0 or index >= len(images):
            raise ValueError(f"frame outside image inventory: {(topic, frame)}")
        image_path = images[index]
        image = Image.open(image_path).convert("RGB")
        if image.size != (width, height):
            raise ValueError(f"unexpected image dimensions: {image_path}")
        baseline_record = _mask_record(matching[0], width, height)
        if proposal is None:
            proposal_mask = baseline_record["mask"]
        else:
            proposal_mask = metrics["_proposal_mask"]
        crop = _crop(baseline_record["mask"], proposal_mask)
        original = _fit(image.crop(crop), (360, 250))
        current = _fit(_mask_overlay(
            image, baseline_record["mask"], (35, 205, 130)).crop(crop), (360, 250))
        proposed = _fit(_mask_overlay(
            image, proposal_mask, (255, 165, 35)).crop(crop), (360, 250))
        difference = _fit(_difference_overlay(
            image, baseline_record["mask"], proposal_mask).crop(crop), (360, 250))
        header = 62
        panel = Image.new("RGB", (1440, 250 + header), "white")
        for x, view in zip((0, 360, 720, 1080),
                           (original, current, proposed, difference)):
            panel.paste(view, (x, header))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 7),
                  f"#{rank} {camera} frame {frame} {row['shape_id']}",
                  fill="black", font=font)
        draw.text((8, 27),
                  f"current area={baseline_record['area']}  cleanup change="
                  f"{float(row.get('change_ratio') or 0):.2%}",
                  fill="black", font=font)
        draw.text((370, 43), "current partial candidate", fill="black", font=font)
        if metrics is None:
            draw.text((730, 27), "no qualifying historical crop proposal",
                      fill="black", font=font)
        else:
            draw.text((730, 27),
                      f"proposal={proposal.get('id')} score={float(proposal.get('score', 0)):.3f}",
                      fill="black", font=font)
            draw.text((730, 43),
                      f"coverage={metrics['baseline_coverage']:.1%} area="
                      f"{metrics['proposal_to_baseline_area_ratio']:.2f}x",
                      fill="black", font=font)
        draw.text((1090, 43), "diff: green kept, yellow added, red missing",
                  fill="black", font=font)
        panels.append(panel)
        public_rows.append({
            "rank": rank,
            "topic": topic,
            "frame": frame,
            "shape_id": row.get("shape_id"),
            "geometry_audit_change_ratio": row.get("change_ratio"),
            "baseline_area": baseline_record["area"],
            "historical_crop_source": source,
            "historical_addition_count": len(additions),
            "matched_proposal_id": proposal.get("id") if proposal else None,
            "matched_proposal_score": proposal.get("score") if proposal else None,
            "match": _public_metrics(metrics),
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
        })

    if not panels:
        raise ValueError("no distant pallet audit rows selected")
    output.mkdir(parents=True, exist_ok=False)
    sheet = Image.new("RGB", (1440, sum(panel.height for panel in panels)), "white")
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    sheet_path = output / "distant-pallet-crop-recovery-review.jpg"
    sheet.save(sheet_path, quality=90, optimize=True)
    if any(_sha256(Path(path)) != digest for path, digest in before.items()):
        raise RuntimeError("an input changed during read-only review")
    report = {
        "schema_version": "labelagent.distant_pallet_crop_recovery_review.v1",
        "input_sha256": before,
        "selection": "highest_geometry_change_small_candidates",
        "selected_count": len(public_rows),
        "matched_expansion_count": sum(
            row["matched_proposal_id"] is not None for row in public_rows),
        "minimum_baseline_coverage": minimum_baseline_coverage,
        "minimum_area_ratio": minimum_area_ratio,
        "maximum_minor_axis_ratio": maximum_minor_axis_ratio,
        "minimum_axis_elongation": minimum_axis_elongation,
        "maximum_other_baseline_fraction": maximum_other_baseline_fraction,
        "new_model_requests": 0,
        "workbench_writes": 0,
        "contact_sheet": str(sheet_path.resolve()),
        "contact_sheet_sha256": _sha256(sheet_path),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "rows": public_rows,
    }
    report_path = output / "distant-pallet-crop-recovery-review.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path.resolve())
    report["report_sha256"] = _sha256(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--augmented", required=True, type=Path, action="append")
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--minimum-baseline-coverage", type=float, default=.5)
    parser.add_argument("--minimum-area-ratio", type=float, default=1.25)
    parser.add_argument("--maximum-minor-axis-ratio", type=float)
    parser.add_argument("--minimum-axis-elongation", type=float, default=2.0)
    parser.add_argument("--maximum-other-baseline-fraction", type=float)
    args = parser.parse_args()
    report = review(
        args.baseline, args.audit, args.augmented, args.image_root, args.output,
        limit=args.limit, width=args.width, height=args.height,
        minimum_baseline_coverage=args.minimum_baseline_coverage,
        minimum_area_ratio=args.minimum_area_ratio,
        maximum_minor_axis_ratio=args.maximum_minor_axis_ratio,
        minimum_axis_elongation=args.minimum_axis_elongation,
        maximum_other_baseline_fraction=args.maximum_other_baseline_fraction)
    print(json.dumps({key: report[key] for key in (
        "selected_count", "matched_expansion_count", "new_model_requests",
        "workbench_writes", "contact_sheet", "contact_sheet_sha256",
        "report_path", "report_sha256", "candidate_only", "human_truth",
        "approved_for_training")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
