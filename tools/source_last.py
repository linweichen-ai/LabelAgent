"""Counterbalance image order while preserving the complete-contour query.

The full JPEG bytes, geometry, class definition and decoding parameters stay
unchanged. Only image order and the corresponding textual image indices swap.
"""
import hashlib
import json
import re
from tools.contour_visibility import classify_payload
from tools.ascii_target_recheck import ascii_card_payload

def digest(body):
    return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def swap_image_roles(body):
    # Copy so that cached baseline bodies cannot be mutated by a comparison.
    result=json.loads(json.dumps(body))
    messages=result['messages']
    if len(messages)!=1 or messages[0]['role']!='user':
        raise ValueError('One user query required')
    content=messages[0]['content']
    if len(content)!=3 or [r['type'] for r in content]!=['text','image_url','image_url']:
        raise ValueError('One prompt and exactly two images required')
    prompt=content[0]['text']
    refs=re.findall(r'\bimage ([12])\b',prompt,flags=re.IGNORECASE)
    if set(refs)!={'1','2'}:
        raise ValueError('Both source/locator image roles must be explicit')
    content[0]['text']=re.sub(r'\b(image )([12])\b',lambda m:m.group(1)+str(3-int(m.group(2))),prompt,flags=re.IGNORECASE)
    content[1],content[2]=content[2],content[1]
    return result

def source_last_payload(fixture,assets,case,view,guard=False):
    if guard:return ascii_card_payload(fixture,assets,case,view,True)
    baseline,_=classify_payload(fixture,assets,case,view,None)
    body=swap_image_roles(baseline)
    return body,digest(body)
