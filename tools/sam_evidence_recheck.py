"""Neutral visual evidence and conservative fresh VLM recheck decisions."""
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.evaluate_target_cards import fit_panel


CATEGORIES = frozenset(("pallet", "person", "uncertain"))


def needs_recheck(raw_categories, supported_classes):
    if len(raw_categories) != 2 or any(c not in CATEGORIES for c in raw_categories):
        raise ValueError("Two original view categories required")
    if any(c not in ("pallet", "person") for c in supported_classes):
        raise ValueError("Unsupported evidence class")
    return len(supported_classes) == 1 and tuple(raw_categories) != (supported_classes[0],) * 2


def choose_support_mask(target, masks):
    if target.dtype != bool or target.ndim != 2 or not target.any() or not masks:
        raise ValueError("Nonempty target and evidence masks required")
    if any(mask.dtype != bool or mask.shape != target.shape for mask in masks):
        raise ValueError("Exact same source geometry required")
    overlaps = [int(np.count_nonzero(target & mask)) for mask in masks]
    index = max(range(len(masks)), key=lambda i: (overlaps[i], -i))
    if 2 * overlaps[index] < int(target.sum()):
        raise ValueError("No evidence meets the frozen half-target requirement")
    return index, overlaps[index]


def rechecked_pair(previous_guarded, supported_classes, fresh_categories):
    if len(previous_guarded) != 2 or any(c not in CATEGORIES for c in previous_guarded):
        raise ValueError("Two prior guarded categories required")
    if len(fresh_categories) != 2 or any(c not in CATEGORIES for c in fresh_categories):
        raise ValueError("Two fresh valid categories required")
    matched = len(supported_classes) == 1 and tuple(fresh_categories) == (supported_classes[0],) * 2
    return {"categories": list(fresh_categories if matched else previous_guarded),
            "fresh_pair_matched_unique_SAM_support": matched}


def evidence_views(source, target, auxiliary):
    """Retain the entire original target; auxiliary pixels are a separate panel.

    No category, ID, score, expected answer or class-dependent color is accepted.
    """
    source = source.convert("RGB")
    if any(mask.dtype != bool or mask.shape != (source.height, source.width) or not mask.any()
           for mask in (target, auxiliary)):
        raise ValueError("Both masks must bind the complete source image")
    def bounds(mask, minimum):
        yy, xx = np.nonzero(mask)
        left, top, right, bottom = int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1
        px, py = max(minimum, math.ceil(.1 * (right - left))), max(minimum, math.ceil(.1 * (bottom - top)))
        return (max(0, left - px), max(0, top - py), min(source.width, right + px), min(source.height, bottom + py))
    context_box = bounds(target | auxiliary, 32)
    target_box = bounds(target, 8)
    x1, y1, x2, y2 = context_box
    target_local, aux_local = target[y1:y2, x1:x2], auxiliary[y1:y2, x1:x2]
    assert int(target_local.sum()) == int(target.sum())
    context = source.crop(context_box)
    marked = context.copy(); draw = ImageDraw.Draw(marked)
    for mask, color in ((aux_local, (0, 220, 255)), (target_local, (255, 0, 255))):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            points = [tuple(p) for p in contour.reshape(-1, 2).tolist()]
            if len(points) >= 2:
                draw.line(points + points[:1], fill=color, width=2)
    gray = Image.new("RGB", source.size, (127, 127, 127))
    target_picture = Image.composite(source, gray, Image.fromarray(target.astype(np.uint8) * 255)).crop(target_box)
    aux_picture = Image.composite(source, gray, Image.fromarray(auxiliary.astype(np.uint8) * 255)).crop(context_box)
    overview, overview_info = fit_panel(marked, 512, 512)
    linked = Image.new("RGB", (1536, 512), (127, 127, 127)); draw = ImageDraw.Draw(linked)
    font = ImageFont.load_default(size=14)
    panels = []
    for i, (title, picture) in enumerate(zip(("TARGET", "AUX", "SOURCE"), (target_picture, aux_picture, marked))):
        panel, info = fit_panel(picture, 480, 480)
        left = 24 + 504 * i; linked.paste(panel, (left, 16))
        draw.text((left + 4, 0), title, fill=(240, 240, 240), font=font)
        panels.append(info)
    return {"overview": overview, "linked": linked}, {
        "context_box": list(context_box), "target_box": list(target_box),
        "original_target_pixels": int(target.sum()), "retained_target_pixels": int(target_local.sum()),
        "auxiliary_pixels": int(auxiliary.sum()), "original_target_not_shrunk": True,
        "category_ID_score_rendered": False, "titles": ["TARGET", "AUX", "SOURCE"],
        "overview": overview_info, "linked_panels": panels}
