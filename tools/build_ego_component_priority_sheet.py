#!/usr/bin/env python3
"""Build an atomic visual sheet for top-ranked ego review components."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont

try:
    from tools.build_ego_component_union_review import file_sha256
except ModuleNotFoundError:  # Direct script execution.
    from build_ego_component_union_review import file_sha256  # type: ignore


def _write_sheet(output: Path, rows: list[dict], review_root: Path) -> Path:
    columns, cell_width, image_height, label_height = 5, 430, 145, 48
    row_count = (len(rows) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_width,
                               row_count * (image_height + label_height)),
                       (11, 18, 26))
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    for index, row in enumerate(rows):
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
        draw.text((left + 4, top + image_height + 4),
                  f"#{row['review_priority_rank']} {row['component_id']} | {row['review_priority_score']:.4f}",
                  fill=(245, 88, 220), font=font)
        reasons = ",".join(row.get("priority_reasons", [])) or "ranked_by_score"
        draw.text((left + 4, top + image_height + 24), reasons[:66],
                  fill=(230, 220, 120), font=font)
    target = output / "priority_review_sheet.jpg"
    canvas.save(target, "JPEG", quality=94)
    return target


def build_sheet(ranking_path: Path, component_review_path: Path,
                output: Path, top_count: int = 25,
                selection_mode: str = "camera_proportional") -> dict:
    if top_count <= 0:
        raise ValueError("top count must be positive")
    if selection_mode not in {"overall", "camera_proportional"}:
        raise ValueError("unknown selection mode")
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
    ranking_rows = ranking.get("rows", [])
    if (len(review_rows) != len(review.get("rows", []))
            or {row.get("component_id") for row in ranking_rows} !=
            set(review_rows)):
        raise ValueError("ranking and component sets differ")
    quotas = None
    if selection_mode == "overall":
        selected_ranking_rows = sorted(
            ranking_rows, key=lambda row: row["review_priority_rank"])[:top_count]
    else:
        cameras = sorted({row.get("camera") for row in ranking_rows})
        if any(camera not in {"camera1", "camera3"} for camera in cameras):
            raise ValueError("camera-proportional selection requires camera identities")
        counts = {camera: sum(row["camera"] == camera for row in ranking_rows)
                  for camera in cameras}
        available = min(top_count, len(ranking_rows))
        exact = {camera: available * counts[camera] / len(ranking_rows)
                 for camera in cameras}
        quotas = {camera: int(exact[camera]) for camera in cameras}
        if available >= len(cameras):
            for camera in cameras:
                quotas[camera] = max(1, quotas[camera])
        while sum(quotas.values()) < available:
            camera = max(cameras,
                         key=lambda item: (exact[item] - quotas[item], item))
            quotas[camera] += 1
        while sum(quotas.values()) > available:
            camera = min(
                (item for item in cameras if quotas[item] > 1),
                key=lambda item: (exact[item] - quotas[item], item))
            quotas[camera] -= 1
        selected_ranking_rows = []
        for camera in cameras:
            camera_rows = sorted(
                (row for row in ranking_rows if row["camera"] == camera),
                key=lambda row: row["camera_review_priority_rank"])
            selected_ranking_rows.extend(camera_rows[:quotas[camera]])
        selected_ranking_rows.sort(key=lambda row: row["review_priority_rank"])
    selected = []
    for ranking_row in selected_ranking_rows:
        source = review_rows[ranking_row["component_id"]]
        if (ranking_row.get("automatic_decision") is not None
                or ranking_row.get("review_image_sha256") !=
                source.get("review_image_sha256")):
            raise ValueError(
                f"ranking row binding mismatch: {ranking_row['component_id']}")
        image_path = component_review_path.parent / source.get("review_image", "")
        if file_sha256(image_path) != source.get("review_image_sha256"):
            raise ValueError(
                f"review image hash mismatch: {ranking_row['component_id']}")
        selected.append({
            "component_id": ranking_row["component_id"],
            "review_priority_rank": ranking_row["review_priority_rank"],
            "review_priority_score": ranking_row["review_priority_score"],
            "review_priority_band": ranking_row["review_priority_band"],
            "camera": ranking_row.get("camera"),
            "camera_review_priority_rank": ranking_row.get(
                "camera_review_priority_rank"),
            "priority_reasons": ranking_row.get("priority_reasons", []),
            "automatic_decision": None,
            "review_image": source["review_image"],
            "review_image_sha256": source["review_image_sha256"],
        })
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        sheet = _write_sheet(staging, selected, component_review_path.parent)
        packet = {
            "schema_version": "labelagent.ego_component_priority_sheet.v1",
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False, "production_deployed": False,
            "default_enabled": False, "automatic_decisions_enabled": False,
            "ranking_sha256": ranking_sha,
            "component_review_sha256": review_sha,
            "selected_count": len(selected),
            "requested_top_count": top_count,
            "selection_mode": selection_mode,
            "camera_quotas": quotas,
            "sheet": sheet.name,
            "sheet_sha256": file_sha256(sheet),
            "rows": selected,
            "next_step": "Review in rank order. This sheet cannot make or approve decisions.",
        }
        target = staging / "priority_review.json"
        target.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        readback = json.loads(target.read_text(encoding="utf-8"))
        if (readback["selected_count"] != len(selected)
                or file_sha256(staging / readback["sheet"]) !=
                readback["sheet_sha256"]
                or any(row["automatic_decision"] is not None
                       for row in readback["rows"])):
            raise ValueError("priority sheet readback verification failed")
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
    parser.add_argument("--top-count", type=int, default=25)
    parser.add_argument("--selection-mode", choices=("overall",
                        "camera_proportional"), default="camera_proportional")
    arguments = parser.parse_args()
    packet = build_sheet(arguments.ranking, arguments.component_review,
                         arguments.output, arguments.top_count,
                         arguments.selection_mode)
    print(json.dumps({
        "selected_count": packet["selected_count"],
        "sheet_sha256": packet["sheet_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
