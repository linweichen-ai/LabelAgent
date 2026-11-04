"""Image-grid proposal augmentation; no reference labels or 3D inputs."""
from copy import deepcopy
import math

import numpy as np

from source.agent.edit_agent import _rasterize, valid_shape


def tile_grid(width, height, fraction=.6, divisions=2):
    """Fixed overlapping image-only crops, in exclusive PIL coordinates.

    The default remains the original four crops. Denser grids require explicit
    arguments, complete coverage and distinct bounds for every requested tile.
    """
    if (not isinstance(divisions, int) or isinstance(divisions, bool) or not 2 <= divisions <= 8
            or width < divisions or height < divisions
            or not math.isfinite(fraction) or not 1/divisions <= fraction < 1):
        raise ValueError('Invalid grid dimensions or crop fraction')
    tw, th = math.ceil(width*fraction), math.ceil(height*fraction)
    xs = [round(i*(width-tw)/(divisions-1)) for i in range(divisions)]
    ys = [round(i*(height-th)/(divisions-1)) for i in range(divisions)]
    bounds = [(x,y,x+tw,y+th) for y in ys for x in xs]
    if len(set(bounds)) != divisions**2:
        raise ValueError('Image is too small for distinct requested crops')
    return bounds


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


def proposal_priorities(first_scale, second_scale, width, height, mode='confidence', baseline=None):
    """Use confidence or geometric agreement with the other image-crop scale."""
    if mode not in ('confidence','consensus','novel_consensus'):
        raise ValueError('Unknown image-only proposal priority')
    all_proposals=first_scale+second_scale
    identifiers=[shape['id'] for shape in all_proposals]
    if len(set(identifiers))!=len(identifiers):
        raise ValueError('Cross-scale proposal IDs must be unique')
    for index,shape in enumerate(all_proposals,1):
        score=float(shape.get('score',-1))
        if (shape.get('label')!='pallet' or not math.isfinite(score) or not 0<=score<=1
                or not valid_shape({**shape,'id':index},width,height)):
            raise ValueError('Invalid image-only priority input')
    if mode=='confidence':
        return {shape['id']:float(shape['score']) for shape in all_proposals}
    if mode=='novel_consensus' and baseline is None:
        raise ValueError('Novel consensus priorities require the original image predictions')
    baseline_records=[_mask_record(shape,width,height) for shape in (baseline or []) if shape['label']=='pallet']
    records=[[_mask_record(shape,width,height) for shape in group] for group in (first_scale,second_scale)]
    priorities={}
    for scale,group in enumerate((first_scale,second_scale)):
        for shape,record in zip(group,records[scale]):
            agreement=max((_iou(record,other) for other in records[1-scale]),default=0.)
            novelty=1-max((_iou(record,old) for old in baseline_records),default=0.) if mode=='novel_consensus' else 1.
            priorities[shape['id']]=float(shape['score'])*novelty*(.5+.5*agreement)
    return priorities


def augment_candidates(baseline, proposals, width, height, minimum_score=.8, addition_ratio=.5, nms_iou=.8, maximum_person_overlap=None, ranking_scores=None, dynamic_diversity=False):
    """Keep the baseline byte-equivalent as objects; append bounded novel masks."""
    if not 0 <= minimum_score <= 1 or not 0 < addition_ratio <= 1 or not 0 < nms_iou <= 1:
        raise ValueError('Invalid augmentation settings')
    if maximum_person_overlap is not None and (not math.isfinite(maximum_person_overlap)
            or not 0 <= maximum_person_overlap <= 1):
        raise ValueError('Invalid predicted-person overlap limit')
    if ranking_scores is not None:
        identifiers=[shape['id'] for shape in proposals]
        if len(set(identifiers))!=len(identifiers) or set(ranking_scores)!=set(identifiers):
            raise ValueError('Priority scores must bind every unique proposal ID exactly')
        if any(not math.isfinite(float(value)) or not 0<=float(value)<=1 for value in ranking_scores.values()):
            raise ValueError('Invalid image-only priority score')
    for index, shape in enumerate([*baseline, *proposals], 1):
        if not valid_shape({**shape, 'id': index}, width, height):
            raise ValueError('Invalid candidate geometry')
    if any(shape.get('label') != 'pallet' for shape in proposals):
        raise ValueError('This experiment augments pallet candidates only')
    pallet = [shape for shape in baseline if shape['label'] == 'pallet']
    existing = [_mask_record(shape, width, height) for shape in pallet]
    limit = max(1, math.floor(len(pallet)*addition_ratio))
    person_mask = None
    if maximum_person_overlap is not None:
        person_mask = np.zeros((height,width),dtype=bool)
        for shape in baseline:
            if shape['label'] == 'person':
                person_mask |= _rasterize(shape,width,height) != 0
    rejected_by_person = 0
    ranked = []
    for index, shape in enumerate(proposals):
        score = float(shape.get('score', -1))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid proposal confidence')
        if score < minimum_score:
            continue
        record = _mask_record(shape, width, height)
        if person_mask is not None:
            x0,y0,x1,y1 = record['bbox']
            fraction = np.count_nonzero(record['mask'] & person_mask[y0:y1,x0:x1])/record['area']
            if fraction > maximum_person_overlap:
                rejected_by_person += 1
                continue
        overlap = max((_iou(record, old) for old in existing), default=0.)
        if overlap >= nms_iou:
            continue
        priority=score*(1-overlap) if ranking_scores is None else float(ranking_scores[shape['id']])
        ranked.append((priority, score, -index, shape, record))
    ranked.sort(key=lambda item: item[:3], reverse=True)
    additions = []
    while ranked:
        if dynamic_diversity:
            def current_priority(item):
                _,score,order,shape,record=item
                base=score if ranking_scores is None else float(ranking_scores[shape['id']])
                return (base*(1-max((_iou(record,old) for old in existing),default=0.)),score,order)
            selected=max(range(len(ranked)),key=lambda index:current_priority(ranked[index]))
        else:
            selected=0
        _, _, _, shape, record=ranked.pop(selected)
        if any(_iou(record, old) >= nms_iou for old in existing):
            continue
        if shape['id'] in {value['id'] for value in [*baseline, *additions]}:
            raise ValueError('Proposal ID collides with an existing ID')
        additions.append(deepcopy(shape))
        existing.append(record)
        if len(additions) >= limit:
            break
    info = {'baseline_pallets': len(pallet), 'tile_proposals': len(proposals),
        'addition_budget': limit, 'added_pallets': len(additions), 'baseline_candidates_preserved': True}
    if person_mask is not None:
        info.update(maximum_predicted_person_overlap=maximum_person_overlap,
                    proposals_rejected_by_predicted_person=rejected_by_person)
    if ranking_scores is not None:
        info['ranking_source']='explicit_image_only_priorities; original model scores retained'
    if dynamic_diversity:
        info['dynamic_diversity']='base priority*(1-max_mask_IoU_to_baseline_and_already_accepted_additions)'
    return deepcopy(baseline)+additions, info


def augment_preserving_selected(original, selected, proposals, width, height,
        minimum_score=.5, addition_ratio=.5, nms_iou=.8, maximum_person_overlap=None,
        ranking_scores=None,dynamic_diversity=False):
    """Keep previously selected masks, using only the original remaining budget."""
    if selected[:len(original)] != original or len(selected) < len(original):
        raise ValueError('Selected candidates must retain the entire original prefix')
    if any(shape['label'] != 'pallet' for shape in selected[len(original):]):
        raise ValueError('Previously selected additions must be pallet proposals')
    if not 0 < addition_ratio <= 1:
        raise ValueError('Invalid original addition budget')
    original_pallets = sum(shape['label']=='pallet' for shape in original)
    budget = max(1, math.floor(original_pallets*addition_ratio))
    already_added = len(selected)-len(original)
    if already_added > budget:
        raise ValueError('Previously selected candidates already exceed the original budget')
    remaining = budget-already_added
    # A prefix of the same greedy ranking has identical choices at any smaller
    # budget; no reference geometry is involved in choosing the retained masks.
    augmented, proposal_info = augment_candidates(selected, proposals, width, height,
        minimum_score=minimum_score, addition_ratio=1., nms_iou=nms_iou,
        maximum_person_overlap=maximum_person_overlap,ranking_scores=ranking_scores,
        dynamic_diversity=dynamic_diversity)
    additions = augmented[len(selected):][:remaining]
    info={'baseline_candidates_preserved': True,
        'previously_selected_candidates_preserved': True, 'original_pallets': original_pallets,
        'original_addition_budget': budget, 'previous_additions': already_added,
        'remaining_addition_budget': remaining, 'added_pallets': len(additions)}
    for key in ('maximum_predicted_person_overlap','proposals_rejected_by_predicted_person','ranking_source','dynamic_diversity'):
        if key in proposal_info:info[key]=proposal_info[key]
    return deepcopy(selected)+additions,info
