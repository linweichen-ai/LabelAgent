"""Strict, source-bound coordinate repair for a new, independent experiment.

The source images are the unchanged native overview/detail views. Normalized
coordinates denote source crop edges, not indices into a resized image. This
module proposes prompts only; it never grants semantic or instance acceptance.
"""
import copy

import cv2
import numpy as np

from visible_base_native_policy import planning_views as _native_views
from visible_base_policy import PLAN_SCHEMA as _ORIGINAL_SCHEMA, png


PLAN_PROMPT = (
    "Name the actual visible physical object containing the marked source pixels. "
    "The marker identifies a seed region and may cover only an edge; it is not "
    "evidence for any category. The first image is a real square source photo. "
    "The next two photos show original left and right local detail without category "
    "labels. Resolve ONE identifiable physical unit, not a whole stack or an "
    "adjacent occluder. If its complete visible structure can be identified, propose "
    "two or three positive points on distinct visible structural parts of that same "
    "unit and up to four negative points on visible adjacent or occluding objects. "
    "All points refer to the FIRST image using normalized integer edge coordinates "
    "0..1000: 0 is its top/left edge, 500 is the center, and 1000 is its bottom/right "
    "edge. Use interior coordinates for points; a point exactly on the bottom or "
    "right edge is outside the photo. Never use coordinates from a detail photo. "
    "At least one positive must remain on the seed region. Do not invent hidden "
    "parts or copy the task as a category. If only an edge, multiple units or "
    "uncertainty can be identified, give empty point lists. Return object, "
    "instance_scope, positive_points, negative_points, evidence."
)

PLAN_SCHEMA = copy.deepcopy(_ORIGINAL_SCHEMA)
for _key in ("positive_points", "negative_points"):
    for _axis in ("x", "y"):
        PLAN_SCHEMA["properties"][_key]["items"]["properties"][_axis]["maximum"] = 1000


def planning_views(source, mask):
    """Retain prior source pixels and mark the corrected coordinate convention."""
    views, details, binding = _native_views(source, mask)
    binding = dict(binding)
    binding.update(
        coordinate_system="normalized_edge_0_1000",
        point_mapping="source_left_or_top + normalized * source_extent / 1000",
        edge_interval="left/top inclusive; right/bottom exclusive",
        final_rounding="Python round once in source pixels; reject if outside",
        no_point_clamping_or_snapping=True,
    )
    return views, details, binding


def validate_decision(decision):
    """Validate locally even if a model endpoint does not enforce its schema."""
    fields = set(PLAN_SCHEMA["required"])
    if type(decision) is not dict or set(decision) != fields:
        return False, "schema: decision fields missing or unexpected"
    for field, limit in (("object", 160), ("evidence", 180)):
        value = decision[field]
        if type(value) is not str or not value.strip() or len(value) > limit:
            return False, "schema: invalid " + field
    scope = decision["instance_scope"]
    if type(scope) is not str or scope not in PLAN_SCHEMA["properties"]["instance_scope"]["enum"]:
        return False, "schema: invalid instance_scope"
    for field, maximum in (("positive_points", 3), ("negative_points", 4)):
        points = decision[field]
        if type(points) is not list or len(points) > maximum:
            return False, "schema: invalid point array or point count"
        for point in points:
            if type(point) is not dict or set(point) != {"x", "y"}:
                return False, "schema: point fields missing or unexpected"
            for axis in ("x", "y"):
                value = point[axis]
                if type(value) is not int or not 0 <= value <= 1000:
                    return False, "schema: coordinates must be integers in 0..1000"
    if scope == "single_visible_object" and not 2 <= len(decision["positive_points"]) <= 3:
        return False, "schema: single-object plan requires two or three positive points"
    if scope != "single_visible_object" and (decision["positive_points"] or decision["negative_points"]):
        return False, "schema: abstention must have empty point lists"
    return True, None


def bind_points(decision, binding, seed):
    valid, why = validate_decision(decision)
    if not valid:
        return None, why
    if decision["instance_scope"] != "single_visible_object":
        return None, "planner did not identify a single visible object"
    if binding.get("coordinate_system") != "normalized_edge_0_1000":
        return None, "binding: wrong or unspecified coordinate convention"
    box = binding.get("source_box")
    mask = np.asarray(seed)
    if (type(box) not in (list, tuple) or len(box) != 4
            or any(type(value) is not int for value in box)
            or mask.ndim != 2
            or not 0 <= box[0] < box[2] <= mask.shape[1]
            or not 0 <= box[1] < box[3] <= mask.shape[0]):
        return None, "binding: invalid source bounds or seed dimensions"
    rows = []
    for label, points in ((1, decision["positive_points"]), (0, decision["negative_points"])):
        for point in points:
            x = box[0] + point["x"] * (box[2] - box[0]) / 1000
            y = box[1] + point["y"] * (box[3] - box[1]) / 1000
            if not box[0] <= x < box[2] or not box[1] <= y < box[3]:
                return None, "planner point outside bound source photo"
            x, y = round(x), round(y)
            if not box[0] <= x < box[2] or not box[1] <= y < box[3]:
                return None, "rounding escapes bound source photo"
            if any((x - row["x"]) ** 2 + (y - row["y"]) ** 2 < 9 for row in rows):
                return None, "duplicate/conflicting prompt points"
            rows.append({"x": x, "y": y, "label": label})
    # Preserve the prior radius-three seed-binding tolerance exactly. Do not
    # move any point toward a seed, reference point, or area-budget boundary.
    binary = (mask != 0).astype("uint8")
    dilated = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    if not any(row["label"] == 1 and dilated[row["y"], row["x"]] for row in rows):
        return None, "no positive bound to original seed"
    if any(row["label"] == 0 and binary[row["y"], row["x"]] for row in rows):
        return None, "negative conflicts with original seed"
    return rows, None
