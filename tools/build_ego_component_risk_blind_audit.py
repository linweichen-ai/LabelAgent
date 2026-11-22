#!/usr/bin/env python3
"""Build blinded top/middle/bottom audit pages for ego risk ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont

try:
    from tools.build_ego_component_union_review import file_sha256
except ModuleNotFoundError:  # Direct script execution.
    from build_ego_component_union_review import file_sha256  # type: ignore


BANDS = ("top", "middle", "bottom")


def _key(seed: str, component_id: str, prefix: str = "select") -> str:
    return hashlib.sha256(
        f"{prefix}|{seed}|{component_id}".encode("utf-8")).hexdigest()


def _band_for_camera_rank(rank: int, count: int) -> str | None:
    if rank <= math.ceil(count * 0.20):
        return "top"
    if math.floor(count * 0.40) < rank <= math.ceil(count * 0.60):
        return "middle"
    if rank > math.floor(count * 0.80):
        return "bottom"
    return None


def stratified_sample(rows: list[dict], per_camera_band: int,
                      seed: str,
                      excluded_component_ids: set[str] | None = None) -> list[dict]:
    if per_camera_band <= 0 or not seed:
        raise ValueError("invalid blind-audit sampling settings")
    excluded_component_ids = excluded_component_ids or set()
    cameras = sorted({row.get("camera") for row in rows})
    if any(camera not in {"camera1", "camera3"} for camera in cameras):
        raise ValueError("blind audit requires known cameras")
    counts = {camera: sum(row["camera"] == camera for row in rows)
              for camera in cameras}
    selected = []
    for camera in cameras:
        for band in BANDS:
            candidates = [
                row for row in rows
                if row["camera"] == camera
                and row.get("component_id") not in excluded_component_ids
                and _band_for_camera_rank(
                    int(row["camera_review_priority_rank"]), counts[camera]) == band
            ]
            if len(candidates) < per_camera_band:
                raise ValueError(f"insufficient {camera}/{band} candidates")
            candidates.sort(key=lambda row: (
                _key(seed, row["component_id"]), row["component_id"]))
            for row in candidates[:per_camera_band]:
                copied = dict(row)
                copied["audit_camera"] = camera
                copied["audit_band"] = band
                selected.append(copied)
    selected.sort(key=lambda row: (
        _key(seed, row["component_id"], "shuffle"), row["component_id"]))
    for index, row in enumerate(selected, 1):
        row["audit_id"] = f"A{index:03d}"
    return selected


def _write_pages(staging: Path, selected: list[dict], review_root: Path,
                 page_size: int = 20) -> list[dict]:
    columns, cell_width, image_height, label_height = 4, 500, 145, 38
    pages = []
    font = ImageFont.load_default()
    for page_index, start in enumerate(range(0, len(selected), page_size), 1):
        page_rows = selected[start:start + page_size]
        row_count = (len(page_rows) + columns - 1) // columns
        canvas = Image.new("RGB", (columns * cell_width,
                                   row_count * (image_height + label_height)),
                           (11, 18, 26))
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(page_rows):
            left = index % columns * cell_width
            top = index // columns * (image_height + label_height)
            with Image.open(review_root / row["review_image"]) as image:
                preview = image.convert("RGB")
                preview.thumbnail((cell_width - 8, image_height - 4))
            canvas.paste(preview, (left + (cell_width - preview.width) // 2,
                                   top + (image_height - preview.height) // 2))
            draw.rectangle((left, top + image_height,
                            left + cell_width, top + image_height + label_height),
                           fill=(23, 33, 44))
            draw.text((left + 5, top + image_height + 5),
                      f"{row['audit_id']}  choose: fill-ok / uncertain / reject",
                      fill=(235, 220, 120), font=font)
        target = staging / f"blind_audit_page_{page_index:02d}.jpg"
        canvas.save(target, "JPEG", quality=94)
        pages.append({
            "page": page_index,
            "first_audit_id": page_rows[0]["audit_id"],
            "last_audit_id": page_rows[-1]["audit_id"],
            "image": target.name,
            "sha256": file_sha256(target),
        })
    return pages


def build_audit(ranking_path: Path, component_review_path: Path,
                output: Path, per_camera_band: int = 10,
                seed: str = "ego-risk-blind-audit-v1",
                exclude_audit_path: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    ranking_sha = file_sha256(ranking_path)
    review_sha = file_sha256(component_review_path)
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    review = json.loads(component_review_path.read_text(encoding="utf-8"))
    for name, packet in (("ranking", ranking), ("component review", review)):
        if (packet.get("candidate_only") is not True
                or packet.get("human_truth") is not False
                or packet.get("approved_for_training") is not False
                or packet.get("production_deployed") is not False
                or packet.get("default_enabled") is not False):
            raise ValueError(f"{name} safety flags are inconsistent")
    if (ranking.get("automatic_decisions_enabled") is not False
            or ranking.get("component_review_sha256") != review_sha):
        raise ValueError("ranking and component review are not bound")
    review_rows = {row.get("component_id"): row
                   for row in review.get("rows", [])}
    if len(review_rows) != len(review.get("rows", [])):
        raise ValueError("duplicate component review identity")
    ranking_ids = {row.get("component_id") for row in ranking.get("rows", [])}
    if None in ranking_ids or len(ranking_ids) != len(ranking.get("rows", [])):
        raise ValueError("ranking component identities are missing or duplicated")
    excluded_ids: set[str] = set()
    excluded_audit_sha = None
    if exclude_audit_path is not None:
        excluded_audit_sha = file_sha256(exclude_audit_path)
        excluded_audit = json.loads(exclude_audit_path.read_text(encoding="utf-8"))
        if (excluded_audit.get("candidate_only") is not True
                or excluded_audit.get("human_truth") is not False
                or excluded_audit.get("automatic_decisions_enabled") is not False
                or excluded_audit.get("ranking_sha256") != ranking_sha):
            raise ValueError("excluded audit safety flags or ranking binding mismatch")
        excluded_rows = excluded_audit.get("mapping")
        if not isinstance(excluded_rows, list) or not excluded_rows:
            raise ValueError("excluded audit mapping is missing")
        excluded_ids = {row.get("component_id") for row in excluded_rows}
        if (None in excluded_ids or len(excluded_ids) != len(excluded_rows)
                or not excluded_ids.issubset(ranking_ids)):
            raise ValueError("excluded audit identities are invalid")
    selected = stratified_sample(
        ranking.get("rows", []), per_camera_band, seed, excluded_ids)
    if any(row["component_id"] in excluded_ids for row in selected):
        raise ValueError("blind audit overlaps excluded audit")
    for row in selected:
        source = review_rows.get(row["component_id"])
        if (source is None
                or row.get("review_image_sha256") !=
                source.get("review_image_sha256")
                or file_sha256(component_review_path.parent /
                               source.get("review_image", "")) !=
                source.get("review_image_sha256")):
            raise ValueError(f"review image binding mismatch: {row['component_id']}")
        row["review_image"] = source["review_image"]
        row["assistant_blind_decision"] = None
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        pages = _write_pages(staging, selected, component_review_path.parent)
        mapping = [{
            "audit_id": row["audit_id"],
            "component_id": row["component_id"],
            "camera": row["audit_camera"],
            "hidden_band": row["audit_band"],
            "global_rank": row["review_priority_rank"],
            "camera_rank": row["camera_review_priority_rank"],
            "score": row["review_priority_score"],
            "assistant_blind_decision": None,
            "review_image_sha256": row["review_image_sha256"],
        } for row in selected]
        packet = {
            "schema_version": "labelagent.ego_component_risk_blind_audit.v1",
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False, "production_deployed": False,
            "default_enabled": False, "automatic_decisions_enabled": False,
            "blinded_for_risk_band": True,
            "ranking_sha256": ranking_sha,
            "component_review_sha256": review_sha,
            "excluded_audit_sha256": excluded_audit_sha,
            "excluded_component_count": len(excluded_ids),
            "disjoint_from_excluded_audit": exclude_audit_path is not None,
            "seed": seed,
            "per_camera_band": per_camera_band,
            "selected_count": len(mapping),
            "expected_stratum_count": len(BANDS) * 2,
            "pages": pages,
            "mapping": mapping,
            "next_step": "Review pages without opening this mapping, then reveal bands only after decisions are frozen.",
        }
        target = staging / "blind_audit.json"
        target.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        readback = json.loads(target.read_text(encoding="utf-8"))
        if (readback["selected_count"] != len(mapping)
                or any(row["assistant_blind_decision"] is not None
                       for row in readback["mapping"])
                or any(file_sha256(staging / page["image"]) != page["sha256"]
                       for page in pages)):
            raise ValueError("blind audit readback verification failed")
        staging.replace(output)
    except Exception:
        if staging.exists() and staging.parent == output.parent:
            shutil.rmtree(staging)
        raise
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--component-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-camera-band", type=int, default=10)
    parser.add_argument("--seed", default="ego-risk-blind-audit-v1")
    parser.add_argument("--exclude-audit", type=Path)
    arguments = parser.parse_args()
    packet = build_audit(
        arguments.ranking, arguments.component_review, arguments.output,
        arguments.per_camera_band, arguments.seed, arguments.exclude_audit)
    print(json.dumps({
        "selected_count": packet["selected_count"],
        "page_count": len(packet["pages"]),
        "page_hashes": [page["sha256"] for page in packet["pages"]],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
