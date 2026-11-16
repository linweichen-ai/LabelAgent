"""Class-symmetric disagreement trigger and measured encoding escalation."""
from copy import deepcopy
import hashlib
import json
from tools.ascii_target_recheck import ascii_card_payload

def resolution_needed(categories):
    if len(categories)!=2 or any(c not in ('pallet','person','uncertain') for c in categories):
        raise ValueError('Two valid category observations required')
    return categories[0]!=categories[1]

def resolution_payload(fixture,assets,case,view,guard=False):
    body,_=ascii_card_payload(fixture,assets,case,view,guard)
    if not guard:body['image_token_budget']=1024
    digest=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return body,digest
