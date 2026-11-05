"""Separate neutral observation from target-category choices; fail closed."""
import re,json,hashlib,io
from pathlib import Path
from PIL import Image
CAPTION_PROMPT=('Describe ONLY the physical object or surface at the center of this actual photo. '
 'State visible material, structure and color in one short English sentence. '
 'If the image is uniform and contains no visible object or structure, describe it as a uniform image. '
 'Do not infer objects from the task, background, a locator or a class list. '
 'Return a JSON object with exactly one key object, a concrete visible description.')
CAPTION_SCHEMA={'type':'object','properties':{'object':{'type':'string','minLength':1,'maxLength':200}},'required':['object'],'additionalProperties':False}
def pallet_evidence(caption):
 text=caption.get('object','')
 return (isinstance(text,str) and bool(re.search(r'\bpallets?\b|\bskids?\b|托盘',text,re.I))
  and not re.search(r'\b(blank|uniform|uncertain|no objects?|no pallets?|outline|contour|locator)\b',text,re.I))
def classify_neutral(workflow,config,image,cache):
 cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
 binding={'image_sha256':hashlib.sha256(image).hexdigest(),'prompt':CAPTION_PROMPT,'schema':CAPTION_SCHEMA,'model':config['llm_model']}
 key=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest();file=cache/(key+'.json')
 if file.exists():row=json.loads(file.read_text(encoding='utf-8'));return row
 decision=workflow.chat(dict(config,llm_max_tokens=256),CAPTION_PROMPT,image,response_schema=CAPTION_SCHEMA)
 if set(decision)!={'object'} or not isinstance(decision['object'],str) or not decision['object'].strip():raise ValueError('Invalid neutral visual observation')
 row={**binding,'decision':decision,'pallet_evidence':pallet_evidence(decision),'real_model_request':True}
 file.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');file.with_suffix('.png').write_bytes(image)
 return row
