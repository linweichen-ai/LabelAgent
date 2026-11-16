"""Category-blind whole-source observation without location guides or seeds."""
import hashlib,json
from tools.evaluate_vlm_grounding import build_payload

CATEGORIES=['flat_field','structured_scene','uncertain']
PROMPT=(
    'Inspect this unmarked photograph as a WHOLE. Report its visual content, '
    'without guessing a selected object or assigning an object class. '
    'flat_field means the photograph contains only a plain field or a smooth '
    'brightness/color gradient, with no distinguishable physical objects or '
    'object boundaries. structured_scene means visible physical objects with '
    'distinguishable boundaries, parts or surface structure are present. '
    'Ignore neutral padding outside the photograph. Use uncertain if the '
    'photographic content cannot be distinguished. Give one short reason about '
    'what is actually visible. Do not invent an object because a region was selected.'
)

def scene_payload(fixture,assets,case):
    ref=case['views']['overview'][0]
    neutral=dict(categories=CATEGORIES,temperature=0,max_tokens=192,prompts={'scene':PROMPT})
    body,_=build_payload(neutral,assets,{'views':{'scene':[ref]}},'scene',fixture['model'])
    body['image_token_budget']=256
    body['response_format']['json_schema']['schema']['properties']['reason']={'type':'string','pattern':'[ -~]{1,100}'}
    return body,hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def reviewed_category(raw_category,scene_category):
    if scene_category not in CATEGORIES:raise ValueError('Unknown scene observation')
    return raw_category if scene_category=='structured_scene' else 'uncertain'
