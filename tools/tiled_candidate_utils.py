"""Image-grid proposal augmentation; no reference labels or 3D inputs."""
from copy import deepcopy
import math

import numpy as np

from source.agent.edit_agent import _rasterize, valid_shape


def tile_grid(width, height, fraction=.6):
    """Four fixed overlapping crops, expressed as exclusive PIL bounds."""
    if width < 2 or height < 2 or not .5 <= fraction < 1:
        raise ValueError('Invalid grid dimensions or crop fraction')
    tw, th = math.ceil(width*fraction), math.ceil(height*fraction)
    return list(dict.fromkeys((x, y, x+tw, y+th)
        for y in (0, height-th) for x in (0, width-tw)))


def remap_tile_result(result, bounds, width, height, identifier, edge_margin=2):
    """Reject internal-edge truncation, then map every part/hole to the image."""
    x0, y0, x1, y1 = bounds
    tw, th = x1-x0, y1-y0
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height) or edge_margin < 0:
        raise ValueError('Invalid crop bounds or edge margin')
    parts = deepcopy(result.get('parts', []))
    if not parts:
        raise ValueError('A tile result requires polygon parts')
    local = {'id': 1, 'label': 'pallet', 'points': parts[0]['points'],
        'holes': parts[0].get('holes', []), 'parts': parts}
    if not valid_shape(local, tw, th):
        raise ValueError('Invalid local tile geometry')
    points = np.array([point for part in parts for point in part['points']], dtype=float)
    low, high = points.min(axis=0), points.max(axis=0)
    if ((x0 > 0 and low[0] <= edge_margin) or
        (y0 > 0 and low[1] <= edge_margin) or
        (x1 < width and high[0] >= tw-1-edge_margin) or
        (y1 < height and high[1] >= th-1-edge_margin)):
        return None
    for part in parts:
        part['points'] = [[x+x0, y+y0] for x, y in part['points']]
        part['holes'] = [[[x+x0, y+y0] for x, y in ring] for ring in part.get('holes', [])]
    score = float(result['score'])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Invalid model confidence')
    shape = {'id': identifier, 'label': 'pallet', 'points': deepcopy(parts[0]['points']),
        'holes': deepcopy(parts[0]['holes']), 'parts': parts, 'score': score,
        'source': 'real_SAM3_fixed_grid_unconfirmed'}
    if not valid_shape({**shape, 'id': 1}, width, height):
        raise ValueError('Invalid mapped tile geometry')
    return shape


def _mask_record(shape, width, height):
    mask = _rasterize(shape, width, height)
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError('Empty candidate raster')
    x0, x1 = int(columns.min()), int(columns.max())+1
    y0, y1 = int(rows.min()), int(rows.max())+1
    return {'bbox': (x0, y0, x1, y1), 'mask': mask[y0:y1, x0:x1].copy(), 'area': len(rows)}


def _iou(first, second):
    a, b = first['bbox'], second['bbox']
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.
    ma = first['mask'][y0-a[1]:y1-a[1], x0-a[0]:x1-a[0]]
    mb = second['mask'][y0-b[1]:y1-b[1], x0-b[0]:x1-b[0]]
    intersection = int(np.count_nonzero(ma & mb))
    return intersection/(first['area']+second['area']-intersection)


def augment_candidates(baseline, proposals, width, height, minimum_score=.8, addition_ratio=.5, nms_iou=.8):
    """Keep the baseline byte-equivalent as objects; append bounded novel masks."""
    if not 0 <= minimum_score <= 1 or not 0 < addition_ratio <= 1 or not 0 < nms_iou <= 1:
        raise ValueError('Invalid augmentation settings')
    for index, shape in enumerate([*baseline, *proposals], 1):
        if not valid_shape({**shape, 'id': index}, width, height):
            raise ValueError('Invalid candidate geometry')
    if any(shape.get('label') != 'pallet' for shape in proposals):
        raise ValueError('This experiment augments pallet candidates only')
    pallet = [shape for shape in baseline if shape['label'] == 'pallet']
    existing = [_mask_record(shape, width, height) for shape in pallet]
    limit = max(1, math.floor(len(pallet)*addition_ratio))
    ranked = []
    for index, shape in enumerate(proposals):
        score = float(shape.get('score', -1))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid proposal confidence')
        if score < minimum_score:
            continue
        record = _mask_record(shape, width, height)
        overlap = max((_iou(record, old) for old in existing), default=0.)
        if overlap >= nms_iou:
            continue
        ranked.append((score*(1-overlap), score, -index, shape, record))
    ranked.sort(key=lambda item: item[:3], reverse=True)
    additions = []
    for _, _, _, shape, record in ranked:
        if any(_iou(record, old) >= nms_iou for old in existing):
            continue
        if shape['id'] in {value['id'] for value in [*baseline, *additions]}:
            raise ValueError('Proposal ID collides with an existing ID')
        additions.append(deepcopy(shape))
        existing.append(record)
        if len(additions) >= limit:
            break
    return deepcopy(baseline)+additions, {'baseline_pallets': len(pallet), 'tile_proposals': len(proposals),
        'addition_budget': limit, 'added_pallets': len(additions), 'baseline_candidates_preserved': True}
