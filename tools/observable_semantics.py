"""Bare-VLM evidence decisions with logically consistent conditional decoding."""
import hashlib
import json
from tools.ascii_target_recheck import ascii_card_payload
from tools.evaluate_vlm_grounding import validated_response

FIELDS=('content','human','pallet','reason','category')

def observable_schema():
    branches=[]
    for content,human,pallet,category in (
        ('empty_or_markers','not_supported','not_supported','uncertain'),
        ('photo','supported','not_supported','person'),
        ('photo','not_supported','supported','pallet'),
        ('photo','not_supported','not_supported','uncertain'),
        ('photo','supported','supported','uncertain')):
        values=dict(content=content,human=human,pallet=pallet,category=category)
        properties={k:({'type':'string','pattern':'[ -~]{1,100}'} if k=='reason' else {'type':'string','const':values[k]}) for k in FIELDS}
        branches.append({'type':'object','properties':properties,'required':list(FIELDS),'additionalProperties':False})
    return {'oneOf':branches}

PROMPT='This image MAY show no real target. Determine visible content before selecting a class. Ignore orange arrows, magenta corners and other drawn guides as physical evidence. An empty uniform area, even with drawn markers, is empty_or_markers. Otherwise use photo. Human evidence is supported ONLY if the selected part has visible human anatomy or clear clothing/body continuity; a curved dark silhouette alone is not evidence. Pallet evidence is supported ONLY for a manufactured transport base with visible deck, support block or runner; a flat background, a shadow, or cargo alone is not evidence. Judge the part at the arrow tip or inside the corner box, not a different nearby object. Blurry photographs can still contain a real visible part, but do not invent missing anatomy or platform structure. Use not_supported when a cue is absent or not identifiable. Describe only visible evidence, then choose the consistent category. Empty, neither type supported, or both types supported means uncertain. Return fields in this exact order: content, human, pallet, reason, category.'

def observable_payload(fixture,assets,case,view,guard=False):
    body,_=ascii_card_payload(fixture,assets,case,view,guard)
    if not guard:
        body['response_format']['json_schema']={'name':'observable_target','schema':observable_schema(),'strict':True}
    digest=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return body,digest

def validated_observable(raw,categories,image_count):
    meta=raw['local_vlm']
    if meta.get('requested_image_token_budget') is None:return validated_response(raw,categories,image_count)
    if meta.get('backend')!='transformers_local_cuda' or meta.get('image_count')!=image_count or raw['choices'][0]['finish_reason']!='stop':
        raise ValueError('Complete real local CUDA response required')
    parsed=json.loads(raw['choices'][0]['message']['content'])
    if tuple(parsed)!=FIELDS:raise ValueError('Actual evidence field order differs')
    if parsed['content'] not in ('photo','empty_or_markers') or any(parsed[k] not in ('supported','not_supported') for k in ('human','pallet')):
        raise ValueError('Unsupported evidence observation')
    reason=parsed['reason']
    if not isinstance(reason,str) or not 1<=len(reason)<=100 or any(not 32<=ord(c)<=126 for c in reason):raise ValueError('Invalid evidence text')
    h,p=(parsed[k]=='supported' for k in ('human','pallet'))
    expected='uncertain' if parsed['content']=='empty_or_markers' or h==p else 'person' if h else 'pallet'
    if parsed['content']=='empty_or_markers' and (h or p):raise ValueError('Marker-only image cannot have physical-class support')
    if parsed['category']!=expected:raise ValueError('Class contradicts the model evidence; never rewrite it')
    return parsed
