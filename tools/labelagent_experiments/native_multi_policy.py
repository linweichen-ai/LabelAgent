"""Image/prediction-only anchors and lossless native mask placement."""
import cv2
import numpy as np


def anchors(seed):
    seed = np.asarray(seed, dtype=bool)
    if seed.ndim != 2 or not seed.any():
        return []
    distance = cv2.distanceTransform(seed.astype('uint8'), cv2.DIST_L2, 5)
    first = np.unravel_index(np.argmax(distance), seed.shape)
    ys = np.where(seed)[0]
    cutoff = int(np.quantile(ys, .75, method='nearest'))
    lower = distance.copy()
    lower[:cutoff] = 0
    second = np.unravel_index(np.argmax(lower), seed.shape)
    result = []
    for name, (y, x) in [('interior_peak', first), ('lower_quartile_peak', second)]:
        if seed[y, x] and not any(p['x'] == x and p['y'] == y for p in result):
            result.append({'origin': name, 'x': int(x), 'y': int(y), 'label': 1})
    return result


def place(raw, box, shape=(1536, 1920)):
    raw = np.asarray(raw).squeeze()
    x1, y1, x2, y2 = box
    if raw.shape != (y2-y1, x2-x1) or not np.isin(raw, [0, 1]).all():
        raise ValueError('Native candidate must be an exact binary crop')
    if not (0 <= x1 < x2 <= shape[1] and 0 <= y1 < y2 <= shape[0]):
        raise ValueError('Crop outside source image')
    full = np.zeros(shape, dtype='uint8')
    full[y1:y2, x1:x2] = raw != 0
    assert int(full.sum()) == int((raw != 0).sum())
    return full
