"""Source-bound target enlargement, class-blind inspection, then classification.

The target is a proposed region, not an approved instance mask. Model answers
and geometric consistency are different evidence and must stay separate.
"""
import base64
import hashlib
import json
import math

import numpy as np
from tools.evaluate_target_cards import fit_panel
from tools.grounded_target_views import target_anchor
from tools.evaluate_vlm_grounding import build_payload, validated_response

INSPECT = (
    'First locate the selected physical surface before assigning any object class. '
    'Image 1 is an unmarked enlargement of the selected region. Image 2 is its '
    'unmarked surrounding photograph, for physical continuity only. '
    'The selection seed in image 1 is ({x},{y}) on a 0..1000 coordinate scale. '
    'Inspect real source pixels at that position, not a different nearby component. '
    'If only a plain uniform field is visible, say empty and x=y=-1. '
    'If the physical surface cannot be located, say unclear and x=y=-1. '
    'Otherwise say visible, give a point on that same physical surface in IMAGE 1 '
    '(0..1000 from left/top), and briefly describe its visible material, color and '
    'structure. Do not name an object category yet. Border padding is not an object.'
)
CLASSIFY = (
    'Image 1 is the unmarked enlargement of the selected target, not an example. '
    'Image 2 is the original surrounding photograph with location guides. '
    'Classify ONLY the physical surface at the selection seed ({x},{y}) in IMAGE 1 '
    '(0..1000 from left/top). Use image 2 to understand directly connected structure; '
    'do not switch to a more prominent nearby object. The guides are not physical '
    'evidence. A plain uniform source field contains no object, regardless of guides. '
    'A separate category-blind inspection reported: {inspection}. This report may '
    'be wrong: check the pixels yourself. A pallet is a manufactured load-bearing '
    'transport platform, including a visible deck, block, runner or base. A person '
    'is a human body or part with visible anatomical, clothing or body continuity. '
    'Return uncertain if the selected surface has no distinguishable physical '
    'object or no evidence of either class. Return category and a short reason '
    'describing the selected source surface, not the guides or neighboring cargo.'
)

def digest(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def target_enlargement(source, mask):
    source = source.convert('RGB')
    if mask.dtype != bool or mask.shape != (source.height, source.width) or not mask.any():
        raise ValueError('Source-bound nonempty boolean selection required')
    yy, xx = np.nonzero(mask)
    box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
    cropmask = mask[box[1]:box[3], box[0]:box[2]]
    panel, fit = fit_panel(source.crop(tuple(box)), 512, 512)
    ax, ay = target_anchor(mask)
    # Pixel-center transform, shared with the inverse used for the diagnostic.
    px = fit['offset'][0] + ((ax - box[0] + .5) * fit['display_size'][0] / cropmask.shape[1]) - .5
    py = fit['offset'][1] + ((ay - box[1] + .5) * fit['display_size'][1] / cropmask.shape[0]) - .5
    seed = [round(px / 511 * 1000), round(py / 511 * 1000)]
    metadata = dict(source_box=box, fit=fit, canvas_size=[512, 512], seed=seed,
                    mask_shape=list(cropmask.shape),
                    mask_packbits_b64=base64.b64encode(np.packbits(cropmask).tobytes()).decode(),
                    original_target_pixels=int(mask.sum()), retained_target_pixels=int(cropmask.sum()),
                    no_markers_in_target_image=True, bbox_is_not_single_object_isolation=True)
    return panel, metadata

def geometry_evidence(grounding, metadata):
    x, y = grounding['x'], grounding['y']
    if grounding['visibility'] != 'visible' or not (0 <= x <= 1000 and 0 <= y <= 1000):
        return dict(point_inside_target=False, point_inside_display=False,
                    geometry_only_not_human_localization_truth=True)
    px, py = x / 1000 * 511, y / 1000 * 511
    ox, oy = metadata['fit']['offset']; dw, dh = metadata['fit']['display_size']
    h, w = metadata['mask_shape']
    ix, iy = math.floor((px - ox + .5) * w / dw), math.floor((py - oy + .5) * h / dh)
    inside = 0 <= ix < w and 0 <= iy < h
    packed = np.frombuffer(base64.b64decode(metadata['mask_packbits_b64']), dtype=np.uint8)
    mask = np.unpackbits(packed)[:h*w].reshape(h, w)
    return dict(point_inside_target=bool(inside and mask[iy, ix]), point_inside_display=inside,
                source_point=[metadata['source_box'][0]+ix, metadata['source_box'][1]+iy],
                geometry_only_not_human_localization_truth=True)

def _base(fixture, assets, refs, prompt):
    neutral = dict(categories=fixture['categories'], temperature=0, max_tokens=192,
                   prompts={'query': prompt})
    body, _ = build_payload(neutral, assets, {'views': {'query': refs}}, 'query', fixture['model'])
    # Multiple images must use the server default, not its single-image override.
    return body

def grounding_payload(fixture, assets, case):
    x, y = fixture['geometry'][case['id']]['seed']
    body = _base(fixture, assets, case['localization'], INSPECT.format(x=x, y=y))
    schema = {'type': 'object', 'properties': {
        'visibility': {'type': 'string', 'enum': ['empty', 'unclear', 'visible']},
        'x': {'type': 'integer', 'minimum': -1, 'maximum': 1000},
        'y': {'type': 'integer', 'minimum': -1, 'maximum': 1000},
        'features': {'type': 'string', 'pattern': '[ -~]{1,100}'}},
        'required': ['visibility', 'x', 'y', 'features'], 'additionalProperties': False}
    body['response_format']['json_schema'].update(name='source_target_inspection', schema=schema)
    return body, digest(body)

def classify_payload(fixture, assets, case, view, grounding):
    x, y = fixture['geometry'][case['id']]['seed']
    # Only the literal model inspection enters the second prompt, never truth.
    inspection = json.dumps({k: grounding[k] for k in ('visibility', 'x', 'y', 'features')}, separators=(',', ':'))
    body = _base(fixture, assets, case['views'][view], CLASSIFY.format(x=x, y=y, inspection=inspection))
    body['response_format']['json_schema']['schema']['properties']['reason'] = {'type':'string','pattern':'[ -~]{1,100}'}
    return body, digest(body)

def validated_grounding(raw):
    meta = raw.get('local_vlm', {})
    choice = raw['choices'][0]
    if meta.get('backend') != 'transformers_local_cuda' or meta.get('image_count') != 2 or choice.get('finish_reason') != 'stop':
        raise ValueError('Complete real CUDA two-image inspection required')
    p = json.loads(choice['message']['content'])
    if set(p) != {'visibility', 'x', 'y', 'features'} or p['visibility'] not in ('visible','empty','unclear'):
        raise ValueError('Invalid inspection fields')
    if any(type(p[k]) is not int or not -1 <= p[k] <= 1000 for k in ('x','y')):
        raise ValueError('Invalid inspection coordinates')
    if p['visibility'] != 'visible' and (p['x'], p['y']) != (-1, -1):
        raise ValueError('Empty/unclear inspection must abstain on coordinates')
    if p['visibility'] == 'visible' and min(p['x'], p['y']) < 0:
        raise ValueError('Visible inspection needs nonnegative coordinates')
    if not isinstance(p['features'],str) or not 1 <= len(p['features']) <= 100 or not all(32 <= ord(c) <= 126 for c in p['features']):
        raise ValueError('Invalid inspection description')
    return p
