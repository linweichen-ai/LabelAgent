"""Reference-derived text; classification receives only the held-out query image."""
import base64
import hashlib
import json
from tools.ascii_target_recheck import ascii_card_payload

def descriptor_payload(raw,model):
    schema={'type':'object','properties':{name:{'type':'string','pattern':'[ -~]{1,240}'} for name in ('pallet_cues','person_cues','uncertain_rule')},'required':['pallet_cues','person_cues','uncertain_rule'],'additionalProperties':False}
    return {'model':model,'temperature':0,'max_tokens':384,'image_token_budget':1024,
            'messages':[{'role':'user','content':[{'type':'text','text':'These four panels are approved, labelled examples: two pallet and two person. There is NO query target in this image. Extract short visible physical cues for both classes and a conservative rule for an empty or indistinct future query. Describe structures, not panel positions, class frequencies or identities. Do not classify an unseen query. Use concise English text.'},
                {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(raw).decode()}}]}],
            'response_format':{'type':'json_schema','json_schema':{'name':'approved_reference_cues','schema':schema,'strict':True}}}

def reason_first_payload(fixture,assets,case,view,guard=False):
    body,_=ascii_card_payload(fixture,assets,case,view,guard)
    if not guard:
        schema=body['response_format']['json_schema']['schema'];old=schema['properties']
        schema['properties']={'reason':old['reason'],'category':old['category']};schema['required']=['reason','category']
    digest=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return body,digest
