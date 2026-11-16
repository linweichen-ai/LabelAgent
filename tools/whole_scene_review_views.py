"""Whole source scene, fixed fractional context and unchanged target geometry."""
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.evaluate_target_cards import fit_panel


def context_boxes(target):
    if not isinstance(target, np.ndarray) or target.dtype != bool or target.ndim != 2 or not target.any():
        raise ValueError("A nonempty source-bound boolean mask required")
    height, width = target.shape
    yy, xx = np.nonzero(target)
    left, top, right, bottom = int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1
    region_width = max(math.ceil(width / 2), right - left)
    region_height = max(math.ceil(height / 2), bottom - top)
    x = max(0, min(math.floor((left + right - region_width) / 2), width - region_width))
    y = max(0, min(math.floor((top + bottom - region_height) / 2), height - region_height))
    px, py = max(8, math.ceil(.1 * (right - left))), max(8, math.ceil(.1 * (bottom - top)))
    tight = (max(0, left - px), max(0, top - py), min(width, right + px), min(height, bottom + py))
    return (x, y, x + region_width, y + region_height), tight


def whole_scene_views(source, target):
    """No class/ID/score or SAM label/geometry is accepted or rendered.

    Both views retain the whole scene. Magenta identifies the original target;
    cyan identifies the same class-blind half-frame region shown separately.
    """
    source = source.convert("RGB")
    if target.shape != (source.height, source.width):
        raise ValueError("Target does not bind source dimensions")
    region_box, tight_box = context_boxes(target)
    x1, y1, x2, y2 = region_box
    assert int(target[y1:y2, x1:x2].sum()) == int(target.sum())
    marked = source.copy(); draw = ImageDraw.Draw(marked)
    contours, _ = cv2.findContours(target.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = [tuple(p) for p in contour.reshape(-1, 2).tolist()]
        if len(points) >= 2:
            draw.line(points + points[:1], fill=(255, 0, 255), width=2)
    region = marked.crop(region_box)
    whole = marked.copy()
    ImageDraw.Draw(whole).rectangle((x1, y1, x2 - 1, y2 - 1), outline=(0, 220, 255), width=3)
    target_picture = Image.composite(source, Image.new("RGB", source.size, (127, 127, 127)),
                                    Image.fromarray(target.astype(np.uint8) * 255)).crop(tight_box)
    font = ImageFont.load_default(size=14)
    overview = Image.new("RGB", (1024, 1024), (127, 127, 127))
    draw = ImageDraw.Draw(overview); overview_meta = []
    for title, picture, box in (("WHOLE", whole, (32, 24, 960, 544)), ("REGION", region, (32, 600, 960, 400))):
        x, y, width, height = box
        fitted, info = fit_panel(picture, width, height)
        overview.paste(fitted, (x, y)); draw.text((x + 4, y - 20), title, font=font, fill=(240, 240, 240))
        overview_meta.append(info)
    linked = Image.new("RGB", (1536, 512), (127, 127, 127)); draw = ImageDraw.Draw(linked); linked_meta = []
    for i, (title, picture) in enumerate(zip(("WHOLE", "REGION", "TARGET"), (whole, region, target_picture))):
        fitted, info = fit_panel(picture, 480, 480)
        x = 24 + i * 504; linked.paste(fitted, (x, 16)); draw.text((x + 4, 0), title, font=font, fill=(240, 240, 240))
        linked_meta.append(info)
    return {"overview": overview, "linked": linked}, {
        "whole_scene_box": [0, 0, source.width, source.height], "region_box": list(region_box), "tight_box": list(tight_box),
        "native_target_pixels": int(target.sum()), "region_retained_target_pixels": int(target[y1:y2, x1:x2].sum()),
        "entire_scene_retained_in_both_views": True, "original_target_not_shrunk": True,
        "region_rule": "One half of source width and height, enlarged only to include the entire target bbox, centered and clamped",
        "display_resize_adds_no_source_detail": True, "categories_IDs_scores_SAM_labels_rendered": False,
        "overview_panels": overview_meta, "linked_panels": linked_meta}
