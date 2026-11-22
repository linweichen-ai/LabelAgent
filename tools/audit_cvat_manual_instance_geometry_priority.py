"""Prioritize manual pallet instances for review using geometry-only risk signals."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _parts(shape: dict[str, Any]) -> list[dict[str, Any]]:
    value = shape.get("parts")
    if isinstance(value, list) and value:
        if not all(isinstance(part, dict) for part in value):
            raise ValueError("shape contains a non-object part")
        return value
    return [{"points": shape.get("points") or [], "holes": shape.get("holes") or []}]


def _metrics(shape: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    parts = _parts(shape)
    outer = [point for part in parts for point in (part.get("points") or [])]
    if len(outer) < 3:
        raise ValueError("shape has fewer than three outer points")
    if not all(isinstance(point, list) and len(point) == 2 for point in outer):
        raise ValueError("shape contains an invalid point")
    xs = [float(point[0]) for point in outer]
    ys = [float(point[1]) for point in outer]
    left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
    bbox_width, bbox_height = right - left + 1, bottom - top + 1
    bbox_area = max(bbox_width * bbox_height, 1)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    hole_count = 0
    hole_vertices = 0
    for part in parts:
        points = [tuple(map(float, point)) for point in (part.get("points") or [])]
        if len(points) >= 3:
            draw.polygon(points, fill=255)
        for hole in part.get("holes") or []:
            ring = [tuple(map(float, point)) for point in hole]
            if len(ring) >= 3:
                draw.polygon(ring, fill=0)
                hole_count += 1
                hole_vertices += len(ring)
    mask_pixels = mask.histogram()[255]
    perimeter = max(2 * (bbox_width + bbox_height), 1)
    edge_touch = left <= 1 or top <= 1 or right >= width - 2 or bottom >= height - 2
    return {
        "part_count": len(parts),
        "hole_count": hole_count,
        "outer_vertices": len(outer),
        "hole_vertices": hole_vertices,
        "bbox": [round(left, 3), round(top, 3), round(right, 3), round(bottom, 3)],
        "bbox_area": round(bbox_area, 3),
        "mask_pixels": mask_pixels,
        "fill_ratio": round(mask_pixels / bbox_area, 6),
        "outer_vertex_per_bbox_perimeter": round(len(outer) / perimeter, 6),
        "edge_touch": edge_touch,
    }


def _risk(metrics: dict[str, Any]) -> tuple[list[str], int]:
    flags: list[str] = []
    score = 0
    checks = (
        (metrics["edge_touch"], "touches_image_edge", 2),
        (metrics["part_count"] > 1, "multiple_parts", 2),
        (metrics["hole_count"] >= 2, "multiple_holes", 1),
        (metrics["outer_vertex_per_bbox_perimeter"] > 0.55,
         "high_vertex_density", 2),
        (metrics["fill_ratio"] < 0.30, "low_bbox_fill", 2),
        (metrics["mask_pixels"] < 1000, "small_mask", 2),
    )
    for condition, name, weight in checks:
        if condition:
            flags.append(name)
            score += weight
    return flags, score


def audit(candidate_manifest: Path, candidate_receipt: Path,
          diversity_audit: Path, image_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_manifest = candidate_manifest.resolve()
    candidate_receipt = candidate_receipt.resolve()
    diversity_audit = diversity_audit.resolve()
    image_root = image_root.resolve()
    candidates_value = _read(candidate_manifest)
    receipt = _read(candidate_receipt)
    diversity = _read(diversity_audit)
    manifest_sha = sha256(candidate_manifest)
    if receipt.get("private_manifest_sha256") != manifest_sha:
        raise ValueError("candidate manifest hash differs from receipt")
    if diversity.get("candidate_manifest_sha256") != manifest_sha:
        raise ValueError("diversity audit binds a different candidate manifest")
    if diversity.get("status") != "assistant_triage_only":
        raise ValueError("diversity audit status differs")
    representatives = diversity.get("representative_candidate_ids")
    candidates = candidates_value.get("candidates")
    if not isinstance(representatives, list) or not isinstance(candidates, list):
        raise ValueError("representative or candidate list is absent")
    by_id = {candidate.get("candidate_id"): candidate for candidate in candidates
             if isinstance(candidate, dict)}
    if len(by_id) != len(candidates):
        raise ValueError("candidate ids are missing or repeated")
    if len(set(representatives)) != len(representatives):
        raise ValueError("representative ids are repeated")

    rows = []
    flag_counts: Counter[str] = Counter()
    for candidate_id in representatives:
        candidate = by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(f"unknown representative: {candidate_id}")
        image_meta, shape = candidate.get("image"), candidate.get("shape")
        if not isinstance(image_meta, dict) or not isinstance(shape, dict):
            raise ValueError(f"candidate {candidate_id}: binding is incomplete")
        if (shape.get("label") != "pallet"
                or (shape.get("cvat_metadata") or {}).get("source") != "manual"):
            raise ValueError(f"candidate {candidate_id}: provenance differs")
        filename = image_meta.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"candidate {candidate_id}: unsafe image filename")
        image_path = image_root / filename
        if not image_path.is_file() or sha256(image_path) != image_meta.get("sha256"):
            raise ValueError(f"candidate {candidate_id}: image hash differs")
        with Image.open(image_path) as opened:
            width, height = opened.size
        metrics = _metrics(shape, width, height)
        flags, score = _risk(metrics)
        flag_counts.update(flags)
        rows.append({"candidate_id": candidate_id, "risk_score": score,
                     "risk_flags": flags, "metrics": metrics})
    rows.sort(key=lambda row: (-row["risk_score"], -len(row["risk_flags"]),
                               row["candidate_id"]))
    score_counts = Counter(row["risk_score"] for row in rows)
    priority_counts = {
        "high": sum(row["risk_score"] >= 4 for row in rows),
        "medium": sum(2 <= row["risk_score"] < 4 for row in rows),
        "routine": sum(row["risk_score"] < 2 for row in rows),
    }
    private = {
        "schema_version": "labelagent.cvat-manual-instance-geometry-priority.v1",
        "status": "assistant_triage_only",
        "candidate_manifest_sha256": manifest_sha,
        "candidate_receipt_sha256": sha256(candidate_receipt),
        "diversity_audit_sha256": sha256(diversity_audit),
        "representative_count": len(rows),
        "priority_counts": priority_counts,
        "risk_score_histogram": dict(sorted(score_counts.items())),
        "risk_flag_counts": dict(sorted(flag_counts.items())),
        "rows": rows,
        "automatic_decisions": False,
        "human_truth": False,
        "training_authorized": False,
        "limitations": [
            "Geometry flags prioritize review; they do not prove an annotation is wrong.",
            "Occlusion and pallet structure can legitimately create holes or multiple parts.",
        ],
    }
    public = {key: private[key] for key in (
        "schema_version", "status", "candidate_manifest_sha256",
        "candidate_receipt_sha256", "diversity_audit_sha256",
        "representative_count", "priority_counts", "risk_score_histogram",
        "risk_flag_counts", "automatic_decisions", "human_truth",
        "training_authorized", "limitations")}
    public["private_candidate_ids_published"] = False
    public["raw_images_published"] = False
    return private, public


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--candidate-receipt", required=True, type=Path)
    parser.add_argument("--diversity-audit", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--private-output", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    args = parser.parse_args()
    if args.private_output.exists() or args.receipt_output.exists():
        raise ValueError("choose fresh output paths")
    private, public = audit(args.candidates, args.candidate_receipt,
                            args.diversity_audit, args.image_root)
    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    args.private_output.write_text(json.dumps(private, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    public["private_audit_sha256"] = sha256(args.private_output)
    args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_output.write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(public, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
