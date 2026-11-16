"""Two images use the validated global budget, not the single-image override."""
import hashlib
import json
from tools.source_companion import unmarked_companion, PREFIX
from tools.ascii_target_recheck import ascii_card_payload

def source_companion_payload(fixture,assets,case,view,guard=False):
    if not guard and len(case['views'][view])!=2:
        raise ValueError('One clean source plus one marked query required')
    body,_=ascii_card_payload(fixture,assets,case,view,guard)
    if not guard:body.pop('image_token_budget')
    return body,hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
