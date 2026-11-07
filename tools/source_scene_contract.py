"""Expose the same decoding schema and its field mapping in the model prompt."""
import json,hashlib
from tools.source_scene_yesno import scene_payload as original,CATEGORIES,reviewed_category
PREFIX=' Return one JSON object matching the following output schema. The category field must contain the observation label defined above; the reason field must give visible source evidence supporting that same choice. Output schema: '
def scene_payload(f,a,c):
    body,_=original(f,a,c)
    schema=body['response_format']['json_schema']['schema']
    body['messages'][0]['content'][0]['text']+=PREFIX+json.dumps(schema,sort_keys=True,separators=(',',':'))
    return body,hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
