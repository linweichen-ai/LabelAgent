"""Person-only guard evidence; never create or augment pallet annotations."""
from copy import deepcopy
import math

from tools.crop_response_binding import check_crop_receipt
from source.agent.edit_agent import valid_shape

PURPOSE = 'predicted_person_guard_seed_not_annotation'


def check_person_guard_receipt(receipt, expected):
    if (expected.get('query') != 'person' or receipt.get('label') != 'person'
            or receipt.get('purpose') != PURPOSE):
        raise ValueError('A native person-guard response is required')
    check_crop_receipt(receipt, expected)


def guard_pending_request(pending, response_exists):
    """Missing successful responses must not be silently inferred again."""
    if pending is not None and not response_exists and pending.get('status') != 'rejected_busy_before_activation':
        raise ValueError('Unresolved model request; inspect server before repeating accepted inference')


def remap_person_guard_result(result, bounds, width, height, identifier):
    if (len(bounds) != 4 or any(type(v) is not int for v in bounds)
            or type(width) is not int or type(height) is not int):
        raise ValueError('Integer image and crop dimensions required')
    x0, y0, x1, y1 = bounds
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError('Invalid person crop bounds')
    try:
        score = float(result['score'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Model confidence required') from error
    if isinstance(result['score'], bool) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Invalid model confidence')
    parts = deepcopy(result.get('parts', []))
    local = {'id': 1, 'label': 'person', 'parts': parts}
    if not parts or not valid_shape(local, x1-x0, y1-y0):
        raise ValueError('Invalid local person geometry')
    # Crop-edge fragments are valid guard seeds, not complete instance claims.
    for part in parts:
        part['points'] = [[x+x0, y+y0] for x, y in part['points']]
        part['holes'] = [[[x+x0, y+y0] for x, y in ring] for ring in part.get('holes', [])]
    shape = {'id': identifier, 'label': 'person', 'points': deepcopy(parts[0]['points']),
             'holes': deepcopy(parts[0]['holes']), 'parts': parts, 'score': score,
             'source': 'real_SAM3_person_crop_guard_unconfirmed', 'purpose': PURPOSE}
    if not valid_shape({**shape, 'id': 1}, width, height):
        raise ValueError('Invalid mapped person geometry')
    return shape
