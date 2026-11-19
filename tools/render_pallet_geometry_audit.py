#!/usr/bin/env python3
"""Render a compact original/before/after sheet from a pallet audit report."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from source.agent.edit_agent import _rasterize
from source.agent.workflow import candidate_overlay, refine_pallet_candidate_geometry


def _candidate_mapping(payload: dict) -> dict:
    nested = payload.get("candidates_by_topic")
    return nested if isinstance(nested, dict) else payload


def _locate(mapping: dict, topic: str, frame: str, shape_id):
    for shape in mapping[topic][str(frame)]:
        if shape.get("id") == shape_id:
            return shape
    raise KeyError((topic, frame, shape_id))


def _fit(image: Image.Image, width: int = 440) -> Image.Image:
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def render(state: Path, report: Path, image_dir: Path, output: Path,
           *, limit: int = 4) -> None:
    payload = json.loads(state.read_text(encoding="utf-8"))
    mapping = _candidate_mapping(payload)
    audit = json.loads(report.read_text(encoding="utf-8"))
    ranked = sorted(audit["results"], key=lambda row: (
        row["holes_before"] - row["holes_after"], row["changed_pixels"]), reverse=True)
    rows = []
    for record in ranked:
        if record["changed_pixels"] <= 0:
            continue
        frame = int(record["frame"])
        source_path = image_dir / f"frame_api_{frame:03d}.jpg"
        if not source_path.exists():
            continue
        image = Image.open(source_path).convert("RGB")
        shape = _locate(mapping, record["topic"], record["frame"], record["shape_id"])
        refined, _ = refine_pallet_candidate_geometry(shape, image.width, image.height)
        mask = _rasterize(shape, image.width, image.height).astype(bool)
        ys, xs = np.where(mask)
        if not len(xs):
            continue
        pad = 28
        box = (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
               min(image.width, int(xs.max()) + pad + 1),
               min(image.height, int(ys.max()) + pad + 1))
        before = Image.open(io.BytesIO(candidate_overlay(source_path.read_bytes(), [shape]))).convert("RGB")
        after = Image.open(io.BytesIO(candidate_overlay(source_path.read_bytes(), [refined]))).convert("RGB")
        panels = [_fit(item.crop(box)) for item in (image, before, after)]
        header = 34
        row = Image.new("RGB", (sum(panel.width for panel in panels),
                                max(panel.height for panel in panels) + header), "white")
        draw = ImageDraw.Draw(row)
        font = ImageFont.load_default()
        titles = (f"Frame {frame} original",
                  f"Before: {record['holes_before']} holes",
                  f"After: {record['holes_after']} holes; {record['change_ratio']:.2%} changed")
        x = 0
        for panel, title in zip(panels, titles):
            row.paste(panel, (x, header))
            draw.text((x + 8, 10), title, fill="black", font=font)
            x += panel.width
        rows.append(row)
        if len(rows) >= limit:
            break
    if not rows:
        raise ValueError("audit report contains no renderable changed candidates")
    sheet = Image.new("RGB", (max(row.width for row in rows),
                              sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()
    render(args.state, args.audit, args.image_dir, args.output, limit=args.limit)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
