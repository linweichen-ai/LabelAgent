#!/usr/bin/env python3
"""Build a high-legibility manual review pack for the safe camera1 tail."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont

try:
    from tools.evaluate_ego_tail_vlm_focused_pilot import build_focused_montage
    from tools.evaluate_ego_tail_vlm_guard import (
        file_sha256, select_camera1_tail)
except ModuleNotFoundError:  # Direct execution adds tools/, not the repo root.
    from evaluate_ego_tail_vlm_focused_pilot import build_focused_montage
    from evaluate_ego_tail_vlm_guard import file_sha256, select_camera1_tail


def write_contact_pages(directory: Path, rows: list[dict], page_size: int = 6
                        ) -> list[dict]:
    if page_size <= 0:
        raise ValueError("page size must be positive")
    pages = []
    columns, cell_width, cell_height = 3, 330, 370
    font = ImageFont.load_default()
    for page_index, start in enumerate(range(0, len(rows), page_size), 1):
        page_rows = rows[start:start + page_size]
        row_count = (len(page_rows) + columns - 1) // columns
        canvas = Image.new(
            "RGB", (columns * cell_width, row_count * cell_height),
            (9, 15, 22))
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(page_rows):
            left = index % columns * cell_width
            top = index // columns * cell_height
            with Image.open(directory / row["focused_image"]) as image:
                preview = image.convert("RGB")
                preview.thumbnail((316, 316), Image.Resampling.LANCZOS)
            canvas.paste(
                preview,
                (left + (cell_width - preview.width) // 2, top + 2))
            draw.text(
                (left + 7, top + 322), row["component_id"],
                fill=(245, 245, 248), font=font)
            draw.text(
                (left + 7, top + 340),
                f"rank {row['camera_rank']} | {row['policy_queue']}",
                fill=(255, 220, 40), font=font)
        name = f"focused_review_page_{page_index:02d}.jpg"
        target = directory / name
        canvas.save(target, "JPEG", quality=93)
        pages.append({
            "page": page_index,
            "first_component_index": start + 1,
            "last_component_index": start + len(page_rows),
            "image": name,
            "sha256": file_sha256(target),
        })
    return pages


def build_pack(policy_path: Path, component_review_path: Path,
               output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    component_review = json.loads(
        component_review_path.read_text(encoding="utf-8"))
    selected = select_camera1_tail(policy, component_review)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    rows = []
    try:
        for item in selected:
            source = component_review_path.parent / item["review_image"]
            if file_sha256(source) != item["review_image_sha256"]:
                raise ValueError(
                    f"review image hash mismatch: {item['component_id']}")
            payload = build_focused_montage(source)
            name = f"{item['component_id']}-focused.png"
            target = temporary / name
            target.write_bytes(payload)
            rows.append({
                **item,
                "focused_image": name,
                "focused_image_sha256": file_sha256(target),
            })
        pages = write_contact_pages(temporary, rows)
        priority_review = sum(
            row["policy_queue"] == "priority_review" for row in rows)
        deferred_manual_review = sum(
            row["policy_queue"] == "deferred_manual_review" for row in rows)
        if priority_review + deferred_manual_review != len(rows):
            raise ValueError("focused review pack contains an unknown queue")
        packet = {
            "schema_version": "labelagent.ego_tail_focused_review_pack.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_only": True,
            "assistant_review_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "production_deployed": False,
            "default_enabled": False,
            "automatic_decisions_enabled": False,
            "policy_sha256": file_sha256(policy_path),
            "component_review_sha256": file_sha256(component_review_path),
            "review_count": len(rows),
            "priority_review": priority_review,
            "deferred_manual_review": deferred_manual_review,
            "rows": rows,
            "pages": pages,
            "boundary": (
                "This pack improves reviewer legibility only. It does not "
                "change masks, truth, training approval, or queue decisions."),
        }
        manifest = temporary / "focused_review_manifest.json"
        manifest.write_text(
            json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        readback = json.loads(manifest.read_text(encoding="utf-8"))
        if readback["review_count"] != len(rows):
            raise ValueError("focused review pack readback mismatch")
        temporary.replace(output)
        return packet
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--component-review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    packet = build_pack(
        arguments.policy, arguments.component_review, arguments.output)
    print(json.dumps({key: packet[key] for key in (
        "review_count", "priority_review", "deferred_manual_review")},
        ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
