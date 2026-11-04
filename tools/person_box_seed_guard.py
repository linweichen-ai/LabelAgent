"""Unpadded per-person boxes as conservative model-derived guard evidence."""
import cv2
import numpy as np
from tools.person_guard_envelope import person_envelope_mask


def person_box_seed_hits(shapes, pixels, width, height):
    """Use the existing box geometry and fixed six-pixel prediction dilation."""
    points = np.asarray(pixels)
    if (points.ndim != 2 or points.shape[1] != 2 or points.dtype.kind not in 'iu'
            or np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height)):
        raise ValueError('Aligned in-image integer pixels required')
    mask = person_envelope_mask(shapes, width, height, 'bounding_box')
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    return mask[points[:, 1], points[:, 0]] != 0
