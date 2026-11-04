"""Bound cached SAM class support for an isolated annotation review proposal.

SAM predictions are corroborating model evidence, never independent truth.
This policy abstains and never invents a replacement class.
"""
from dataclasses import asdict, dataclass

import numpy as np
from source.agent.edit_agent import _rasterize, valid_shape


POLICY = "single-mask-half-target-class-support-v1"
CLASSES = ("pallet", "person")


def cached_class_masks(receipt, *, image_sha256, frame, camera, size, category):
    """Reject stale-image, different-frame/prompt and invalid-geometry caches."""
    if category not in CLASSES:
        raise ValueError("Unsupported evidence class")
    expected = {"input_sha256": image_sha256, "frame": frame, "camera": camera,
                "prompt": category, "label": category, "status": "ok"}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Cache does not bind this source, frame and class")
    results = receipt["response"]["results"]
    if type(receipt.get("masks")) is not int or receipt["masks"] != len(results):
        raise ValueError("Cache result count differs")
    width, height = size
    masks = []
    for result in results:
        shape = {"id": 1, "label": category, "parts": result.get("parts", [])}
        if not shape["parts"] or not valid_shape(shape, width, height):
            raise ValueError("Invalid cached polygon geometry")
        masks.append(_rasterize(shape, width, height).astype(bool))
    return masks


@dataclass(frozen=True)
class ClassSupport:
    target_pixels: int
    pallet_max_overlap_pixels: int
    person_max_overlap_pixels: int
    supported_classes: tuple


def mask_support(target, class_masks):
    """Use the maximum single-mask target coverage, with a fixed inclusive1/2.

    Background objects and separate fragments cannot accumulate into support.
    No target label, ID, expected class, scores or tuning parameter is accepted.
    """
    if not isinstance(target, np.ndarray) or target.dtype != bool or target.ndim != 2 or not target.any():
        raise ValueError("A nonempty2D boolean target mask is required")
    if set(class_masks) != set(CLASSES):
        raise ValueError("Both class evidence sets are required")
    area = int(target.sum())
    hits = {}
    for category in CLASSES:
        overlaps = []
        for mask in class_masks[category]:
            if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != target.shape:
                raise ValueError("Evidence masks must share the exact source geometry")
            overlaps.append(int(np.count_nonzero(target & mask)))
        hits[category] = max(overlaps, default=0)
    supported = tuple(category for category in CLASSES if 2 * hits[category] >= area)
    return ClassSupport(area, hits["pallet"], hits["person"], supported)


def guard_proposal(category, support):
    if category not in (*CLASSES, "uncertain") or not isinstance(support, ClassSupport):
        raise ValueError("Supported category and ClassSupport required")
    accepted = category in CLASSES and support.supported_classes == (category,)
    after = category if accepted else "uncertain"
    return {"policy": POLICY, "raw_category": category, "category": after,
            "changed": after != category, "support": asdict(support),
            "reason": "unique_matching_class_support" if accepted else "insufficient_conflicting_or_uncertain"}
