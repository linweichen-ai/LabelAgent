"""Change only the observer label vocabulary and matching legend words."""
import json,hashlib
from tools.source_scene import scene_payload as original
CATEGORIES=['no','yes','uncertain']
def scene_payload(f,a,c):
    body,_=original(f,a,c)
    text=body['messages'][0]['content'][0]['text']
    body['messages'][0]['content'][0]['text']=text.replace('flat_field means','no means').replace('structured_scene means','yes means')
    body['response_format']['json_schema']['schema']['properties']['category']['enum']=CATEGORIES
    return body,hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def reviewed_category(raw,scene):
    if scene not in CATEGORIES:raise ValueError('Invalid observer label')
    return raw if scene=='yes' else 'uncertain'
