"""Emit source-evidence text before a model-generated class suffix."""
import hashlib
import json
import re
from tools.source_last import source_last_payload as BASE_PAYLOAD
from tools.evaluate_vlm_grounding import validated_response as BASE_VALIDATOR

OLD_RETURN = 'Return category and one short reason describing the selected physical surface.'
NEW_RETURN = (
    'First describe the selected physical surface actually visible in clean image 2. '
    'Then give its class. Return one JSON field named result containing '
    '"short visible-structure description :: class". The class suffix must be '
    'pallet, person or uncertain. The description must precede the explicit class suffix.'
)
PATTERN = r'[ -~]{1,100} :: (pallet|person|uncertain)'
SCHEMA = dict(type='object', properties={'result':dict(type='string',pattern=PATTERN)},
              required=['result'], additionalProperties=False)


def evidence_first_payload(fixture, assets, case, view, guard=False):
    body, digest = BASE_PAYLOAD(fixture, assets, case, view, guard)
    if guard: return body, digest
    result = json.loads(json.dumps(body))
    text = result['messages'][0]['content'][0]['text']
    if not text.endswith(OLD_RETURN) or text.count(OLD_RETURN) != 1:
        raise ValueError('Expected unchanged incumbent return-format sentence')
    result['messages'][0]['content'][0]['text'] = text[:-len(OLD_RETURN)] + NEW_RETURN
    result['response_format']['json_schema']['schema'] = json.loads(json.dumps(SCHEMA))
    digest = hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return result, digest


def decode_result(parsed, categories):
    if set(parsed) != {'result'} or not isinstance(parsed['result'],str) or not re.fullmatch(PATTERN,parsed['result']):
        raise ValueError('Invalid evidence-first result')
    reason, category = parsed['result'].rsplit(' :: ',1)
    if not 1<=len(reason)<=100 or category not in categories:
        raise ValueError('Invalid model-generated class suffix')
    # Extract the explicit suffix exactly; never infer a class from evidence.
    return dict(reason=reason,category=category)


def validated_evidence_first(raw, categories, image_count):
    parsed = json.loads(raw['choices'][0]['message']['content'])
    if set(parsed) != {'result'}:
        return BASE_VALIDATOR(raw,categories,image_count)  # Baseline and guard only.
    meta = raw.get('local_vlm',{})
    if meta.get('backend')!='transformers_local_cuda' or meta.get('image_count')!=image_count:
        raise ValueError('Real CUDA/image-count confirmation required')
    if raw['choices'][0].get('finish_reason')!='stop':
        raise ValueError('Unfinished evidence-first output')
    return decode_result(parsed,categories)
