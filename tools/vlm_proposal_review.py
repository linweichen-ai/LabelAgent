"""A VLM reviews its own untrusted class proposal; no image-statistic/SAM filter."""
import hashlib
import json
from tools.ascii_target_recheck import ascii_card_payload

REVIEW='A previous VLM proposed category {draft} for the marked part. This is an UNTRUSTED draft, not a reference label. Inspect ONLY the photograph at the orange arrow tip or inside magenta corners and check whether visible evidence supports that draft. Ignore drawn guides as physical objects. Judge the visible part, not whether the entire object is shown. A partial pallet deck, support block or runner can support pallet; a visible human body part with anatomical or clothing continuity can support person. Cargo alone, a curved dark patch, a shadow or reflection cannot verify a class. A uniform blank remains empty even with an arrow or box. Use supported only when the draft fits identifiable photographic evidence; use unsupported for a clearly wrong/empty target and uncertain if identity cannot be established. Do not agree merely because a draft was provided. Write reason before verdict.'

def review_payload(fixture,assets,case,view,draft):
    if draft not in ('pallet','person'):raise ValueError('Review only an actual non-abstaining VLM proposal')
    body,_=ascii_card_payload(fixture,assets,case,view)
    body['messages'][0]['content'][0]['text']=REVIEW.format(draft=draft)
    schema={'type':'object','properties':{'reason':{'type':'string','pattern':'[ -~]{1,100}'},'verdict':{'type':'string','enum':['supported','unsupported','uncertain']}},'required':['reason','verdict'],'additionalProperties':False}
    body['response_format']['json_schema']={'name':'untrusted_VLM_proposal_review','schema':schema,'strict':True}
    return body,hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def validated_review(raw):
    meta=raw['local_vlm'];choice=raw['choices'][0]
    if meta.get('backend')!='transformers_local_cuda' or meta.get('image_count')!=1 or choice['finish_reason']!='stop':raise ValueError('Complete real VLM review required')
    parsed=json.loads(choice['message']['content'])
    if list(parsed)!=['reason','verdict'] or parsed['verdict'] not in ('supported','unsupported','uncertain'):raise ValueError('Invalid review order or decision')
    if not isinstance(parsed['reason'],str) or not 1<=len(parsed['reason'])<=100 or any(not 32<=ord(c)<=126 for c in parsed['reason']):raise ValueError('Invalid review evidence')
    return parsed

def reviewed_category(draft,verdict):
    if draft not in ('pallet','person') or verdict not in ('supported','unsupported','uncertain'):raise ValueError('Unsupported model decision')
    return draft if verdict=='supported' else 'uncertain'
