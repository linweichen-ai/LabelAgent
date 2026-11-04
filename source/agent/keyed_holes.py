"""Transport handles preserve model judgements and exact original hole identity."""
from copy import deepcopy


def reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def decode_keyed_assessments(result, holes):
    result = deepcopy(result)
    values = result.get('hole_assessment')
    expected = {f'h{i}': h['id'] for i, h in enumerate(holes)}
    if len(set(expected.values())) != len(holes):
        raise ValueError('Input hole IDs must be unique')
    if not isinstance(values, dict) or set(values) != set(expected):
        raise ValueError('Keyed hole assessment must cover every required handle exactly once')
    assessments = []
    for handle, identifier in expected.items():
        value = values[handle]
        if not isinstance(value, dict) or set(value) != {'semantic', 'reason'}:
            raise ValueError('Invalid keyed assessment fields: ' + handle)
        if value['semantic'] not in ('ground','non_ground') or not isinstance(value['reason'],str) or not value['reason'].strip():
            raise ValueError('Missing model semantic/evidence: ' + handle)
        assessments.append({'hole_id':identifier, **value})
    result['hole_assessment'] = assessments
    return result
