"""Bounded same-scan metric neighbourhood of predicted person points.

This is one Euclidean dilation from the original seeds. It never propagates
through newly protected points and never edits a candidate or reads labels.
"""
import math

import numpy as np
from scipy.spatial import cKDTree


def expand_person_seeds(points, seeds, radius_m=.15):
    xyz = np.asarray(points)
    mask = np.asarray(seeds)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.dtype.kind not in 'fiu'
            or not np.isfinite(xyz).all() or mask.dtype != np.bool_
            or mask.shape != (len(xyz),)):
        raise ValueError('Finite XYZ and aligned boolean original seeds required')
    if (isinstance(radius_m, (bool, np.bool_)) or not isinstance(radius_m, (int, float))
            or not math.isfinite(radius_m) or radius_m <= 0):
        raise ValueError('Positive finite metric radius required')
    protected = mask.copy()
    if mask.any():
        distances, _ = cKDTree(np.asarray(xyz[mask], dtype=np.float64)).query(
            np.asarray(xyz, dtype=np.float64), k=1, workers=1)
        protected |= distances <= radius_m
    return protected
