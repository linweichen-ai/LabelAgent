"""An image-only point menu; proposal validity never grants mask acceptance.

This changes both the view count and response interface from native coordinate
planning. It is a separate proposal experiment, not a single-factor ablation.
"""
import copy
import hashlib

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from native_resume_policy import PLAN_SCHEMA as _NATIVE_SCHEMA
from native_resume_policy import planning_views as _native_views
from visible_base_policy import png


PLAN_PROMPT = (
    "The first photo is an unmarked original view. The second is the SAME photo "
    "with a seed locator and a finite menu of numbered points. The arrow or corner "
    "marks locate source seed pixels; they are not evidence for a category. "
    "Name the actual visible physical object containing the seed. Resolve ONE "
    "identifiable physical unit, not a stack or an adjacent occluder. If its "
    "complete visible structure is identifiable, select two or three distinct "
    "positive point IDs on different visible structural parts of that same unit. "
    "You may select zero to four distinct negative point IDs on visible adjacent "
    "or occluding objects. At least one positive must remain on the seed. A point "
    "cannot be both positive and negative. Return only IDs shown in the second "
    "image, such as P01; do not return or invent coordinates. Do not select points "
    "just to fill a list, infer hidden parts, or copy the task as a category. "
    "If only an edge, multiple units, or uncertainty can be identified, report the "
    "appropriate instance_scope with both ID lists empty. If the offered points "
    "cannot represent the same identifiable unit, abstain with empty lists. Return "
    "object, instance_scope, positive_ids, negative_ids, evidence."
)

MAX_CATALOG_POINTS = 23
MIN_DISPLAY_SPACING = 36.0
_FIELDS = {"object", "instance_scope", "positive_ids", "negative_ids", "evidence"}


def _catalog(mask, binding):
    """Three seed-derived locations, then a fixed crop-relative 5-by-4 grid."""
    binary = (np.asarray(mask) != 0).astype("uint8")
    ys, xs = np.where(binary)
    if not len(xs):
        raise ValueError("An empty seed cannot define a point menu")
    box = binding["source_box"]
    width, height = box[2] - box[0], box[3] - box[1]
    scale = 512 / width
    distance = cv2.distanceTransform(np.pad(binary, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    ay, ax = map(int, np.unravel_index(distance.argmax(), distance.shape))
    proposed = [(ax, ay, "seed_interior")]
    extent = int(xs.max()) - int(xs.min())
    for name, eligible in (
        ("seed_left", xs <= int(xs.min()) + extent * .25),
        ("seed_right", xs >= int(xs.max()) - extent * .25),
    ):
        indexes = np.flatnonzero(eligible)
        best = int(indexes[np.argmax(distance[ys[indexes], xs[indexes]])])
        proposed.append((int(xs[best]), int(ys[best]), name))
    for yp in (12, 36, 64, 88):
        for xp in (10, 30, 50, 70, 90):
            proposed.append((round(box[0] + width * xp / 100),
                             round(box[1] + height * yp / 100), "fixed_grid"))
    catalog = []
    spacing = max(3.0, MIN_DISPLAY_SPACING / scale)
    for x, y, origin in proposed:
        if not box[0] <= x < box[2] or not box[1] <= y < box[3]:
            continue
        if any((x - point["x"]) ** 2 + (y - point["y"]) ** 2 < spacing ** 2 for point in catalog):
            continue
        catalog.append({"id": f"P{len(catalog) + 1:02d}", "x": x, "y": y,
                        "display_xy": [round((x - box[0]) * scale), round((y - box[1]) * scale)],
                        "origin": origin})
    if not catalog or len(catalog) > MAX_CATALOG_POINTS:
        raise ValueError("Invalid deterministic point catalog")
    return catalog


def _font():
    for name in ("C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, 13)
        except OSError:
            pass
    return ImageFont.load_default()


def _draw_ids(image, catalog):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    font = _font()
    placed = []
    for point in catalog:
        x, y = point["display_xy"]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 240, 70), outline=(15, 15, 15), width=1)
        text = point["id"]
        bounds = draw.textbbox((0, 0), text, font=font)
        width, height = bounds[2] - bounds[0] + 4, bounds[3] - bounds[1] + 4
        options = [(x + 5, y + 4), (x - width - 5, y + 4),
                   (x + 5, y - height - 4), (x - width - 5, y - height - 4)]
        options = [(min(max(2, left), image.width - width - 2),
                    min(max(2, top), image.height - height - 2)) for left, top in options]
        def overlap(position):
            left, top = position
            return sum(max(0, min(left + width, r) - max(left, l)) *
                       max(0, min(top + height, b) - max(top, t)) for l, t, r, b in placed)
        left, top = min(options, key=overlap)
        rect = (left, top, left + width, top + height)
        draw.rectangle(rect, fill=(20, 20, 20))
        draw.text((left + 2 - bounds[0], top + 2 - bounds[1]), text, font=font, fill=(255, 255, 255))
        placed.append(rect)
    return result


def make_views(source, mask):
    """Return pointer/corners ID views, one clean view, and exact point binding.

    Each request uses ``[clean, views[cue]]`` in that order. Original detail views
    are intentionally omitted; the original source photo and seed are unchanged.
    """
    native, _details, original_binding = _native_views(source, mask)
    binding = copy.deepcopy(original_binding)
    box = binding["source_box"]
    clean = source.crop(tuple(box)).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    catalog = _catalog(mask, binding)
    views = {name: _draw_ids(view, catalog) for name, view in native.items()}
    binding.update(
        coordinate_system="fixed_source_pixel_point_ids",
        point_mapping="selected ID -> exact recorded source integer pixel; no normalized-coordinate round trip",
        final_rounding="none on selected IDs; menu locations fixed before the request",
        point_catalog=catalog,
        source_size=list(source.size),
        source_rgb_pixel_sha256=hashlib.sha256(source.convert("RGB").tobytes()).hexdigest(),
        seed_binary_sha256=hashlib.sha256((np.asarray(mask) != 0).astype("uint8").tobytes()).hexdigest(),
        menu_policy="seed_distance_peak_left_right_then_fixed_5x4_grid",
        menu_max_points=MAX_CATALOG_POINTS,
        min_display_spacing=MIN_DISPLAY_SPACING,
        request_image_order=["clean_same_crop", "same_crop_with_seed_locator_and_point_ids"],
        request_image_count=2,
        detail_views_used=False,
        no_reference_points_or_area_budget=True,
        no_automatic_point_completion_or_replacement=True,
        expected_class_rendered=False,
        not_single_factor_ablation=True,
        required_downstream_gates=["neutral_category", "subject_identity", "single_base",
                                   "complete_visible_extent", "complete_native_mask", "original_area_count_budget",
                                   "original_instance_purity_support", "incumbent_preservation"],
    )
    return views, clean, binding


def schema_for(binding):
    ids = [point["id"] for point in binding["point_catalog"]]
    if not ids or len(ids) != len(set(ids)) or len(ids) > MAX_CATALOG_POINTS:
        raise ValueError("Invalid point catalog IDs")
    return {
        "type": "object",
        "properties": {
            "object": copy.deepcopy(_NATIVE_SCHEMA["properties"]["object"]),
            "instance_scope": copy.deepcopy(_NATIVE_SCHEMA["properties"]["instance_scope"]),
            "positive_ids": {"type": "array", "maxItems": 3, "uniqueItems": True,
                             "items": {"type": "string", "enum": ids}},
            "negative_ids": {"type": "array", "maxItems": 4, "uniqueItems": True,
                             "items": {"type": "string", "enum": ids}},
            "evidence": copy.deepcopy(_NATIVE_SCHEMA["properties"]["evidence"]),
        },
        "required": ["object", "instance_scope", "positive_ids", "negative_ids", "evidence"],
        "additionalProperties": False,
    }


def bind_ids(decision, binding, seed):
    """Strictly map selected IDs; never infer, snap, replace or complete a choice."""
    if type(decision) is not dict or set(decision) != _FIELDS:
        return None, "schema: decision fields missing or unexpected"
    for field, limit in (("object", 160), ("evidence", 180)):
        value = decision[field]
        if type(value) is not str or not value.strip() or len(value) > limit:
            return None, "schema: invalid " + field
    scope = decision["instance_scope"]
    if type(scope) is not str or scope not in _NATIVE_SCHEMA["properties"]["instance_scope"]["enum"]:
        return None, "schema: invalid instance_scope"
    if binding.get("coordinate_system") != "fixed_source_pixel_point_ids":
        return None, "binding: wrong or unspecified point ID convention"
    catalog = binding.get("point_catalog", [])
    if type(catalog) is not list or not catalog or len(catalog) > MAX_CATALOG_POINTS:
        return None, "binding: invalid point catalog"
    if any(type(point) is not dict or type(point.get("id")) is not str for point in catalog):
        return None, "binding: invalid catalog entry"
    by_id = {point["id"]: point for point in catalog}
    if len(by_id) != len(catalog):
        return None, "binding: duplicate catalog ID"
    for field, maximum in (("positive_ids", 3), ("negative_ids", 4)):
        choices = decision[field]
        if type(choices) is not list or len(choices) > maximum or any(type(v) is not str for v in choices):
            return None, "schema: invalid ID list or count"
        if len(choices) != len(set(choices)):
            return None, "duplicate point IDs"
        if any(value not in by_id for value in choices):
            return None, "unknown point ID"
    if set(decision["positive_ids"]) & set(decision["negative_ids"]):
        return None, "positive/negative ID conflict"
    if scope != "single_visible_object":
        if decision["positive_ids"] or decision["negative_ids"]:
            return None, "schema: abstention must have empty ID lists"
        return None, "planner did not identify a single visible object"
    if not 2 <= len(decision["positive_ids"]) <= 3:
        return None, "schema: single-object plan requires two or three positive IDs"
    mask = np.asarray(seed)
    box = binding.get("source_box")
    if (type(box) not in (list, tuple) or len(box) != 4
            or any(type(value) is not int for value in box) or mask.ndim != 2
            or not 0 <= box[0] < box[2] <= mask.shape[1]
            or not 0 <= box[1] < box[3] <= mask.shape[0]):
        return None, "binding: invalid source bounds or seed dimensions"
    rows = []
    for label, choices in ((1, decision["positive_ids"]), (0, decision["negative_ids"])):
        for choice in choices:
            point = by_id[choice]
            x, y = point.get("x"), point.get("y")
            if type(x) is not int or type(y) is not int or not box[0] <= x < box[2] or not box[1] <= y < box[3]:
                return None, "binding: catalog point outside bound source photo"
            if any((x - row["x"]) ** 2 + (y - row["y"]) ** 2 < 9 for row in rows):
                return None, "duplicate/conflicting prompt points"
            rows.append({"x": x, "y": y, "label": label})
    binary = (mask != 0).astype("uint8")
    dilated = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    if not any(row["label"] == 1 and dilated[row["y"], row["x"]] for row in rows):
        return None, "no positive bound to original seed"
    if any(row["label"] == 0 and binary[row["y"], row["x"]] for row in rows):
        return None, "negative conflicts with original seed"
    return rows, None
