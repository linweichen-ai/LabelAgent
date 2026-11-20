#!/usr/bin/env python3
"""Create QA contact sheets from rendered document pages."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-sheet", type=int, default=9)
    args = parser.parse_args()
    pages = sorted(
        args.input.glob("page-*.png"),
        key=lambda path: int(path.stem.rsplit("-", 1)[-1]),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    label_font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 24)
    for offset in range(0, len(pages), args.per_sheet):
        subset = pages[offset:offset + args.per_sheet]
        thumbs = []
        for path in subset:
            with Image.open(path) as page:
                page = page.convert("RGB")
                page.thumbnail((520, 680), Image.Resampling.LANCZOS)
                panel = Image.new("RGB", (560, 740), "#E8E8E8")
                panel.paste(page, ((560 - page.width) // 2, 45))
                ImageDraw.Draw(panel).text((20, 10), path.stem, fill="black", font=label_font)
                thumbs.append(panel)
        sheet = Image.new("RGB", (1680, 2220), "#BFC5CA")
        for index, thumb in enumerate(thumbs):
            sheet.paste(thumb, ((index % 3) * 560, (index // 3) * 740))
        target = args.output / f"sheet-{offset // args.per_sheet + 1:02d}.jpg"
        sheet.save(target, "JPEG", quality=88, optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
