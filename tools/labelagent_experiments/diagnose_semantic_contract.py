"""Paired actual image-model calls isolate schema/prompt semantics."""
import json,base64,time,urllib.request,urllib.error,hashlib,io
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/labelagent_semantic_recovery_20261105/semantic_contract'
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
CATEGORIES=['single_visible_pallet_base','multiple_pallet_group','cargo_or_other','uncertain']
OLD=('Identify ONLY the selected target, not other nearby objects. The image is either selected original pixels on gray background, or original context with one thin target outline. '
 'Ignore the gray background, outline and all unselected neighbors. Describe visible target material and structure. '
 'Use single_visible_pallet_base for one actual pallet deck and its feet, including visible parts under occlusion; multiple_pallet_group only if the SELECTED target itself spans multiple distinct pallet decks. '
 'Boxes, people, robot and floor are cargo_or_other; blank or unresolved evidence is uncertain. Never assume a pallet is present. A colored line is never the object.')
SCHEMA={'type':'object','properties':{'object':{'type':'string','minLength':1,'maxLength':120},'category':{'type':'string','enum':CATEGORIES}},'required':['object','category'],'additionalProperties':False}
CAPTION={'type':'object','properties':{'object':{'type':'string','minLength':1,'maxLength':200}},'required':['object'],'additionalProperties':False}
def request(image,prompt,schema):
 b=io.BytesIO();image.save(b,'PNG');raw=b.getvalue()
 payload={'model':'labelagent-vlm','temperature':0,'max_tokens':256,'messages':[{'role':'user','content':[{'type':'text','text':prompt},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(raw).decode()}}]}],'response_format':{'type':'json_schema','json_schema':{'name':'evidence','schema':schema,'strict':True}}}
 for _ in range(100):
  try:
   with urllib.request.urlopen('http://127.0.0.1:8765/api/run/status',timeout=15) as f:running=json.load(f).get('status')=='running'
   if running:time.sleep(3);continue
   req=urllib.request.Request('http://127.0.0.1:18868/v1/chat/completions',json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=240) as f:r=json.load(f)
   return {'decision':json.loads(r['choices'][0]['message']['content']),'usage':r.get('usage'),'runtime':r.get('local_vlm'),'image_sha256':hashlib.sha256(raw).hexdigest(),'prompt':prompt,'schema':schema}
  except urllib.error.HTTPError as e:
   body=e.read().decode()
   if e.code!=503:raise RuntimeError(body)
   time.sleep(3)
 raise RuntimeError('Shared model busy; no other service stopped')
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'completed.json').exists():raise FileExistsError('Preserve completed controls')
 raw=Image.open(ROOT/'outputs/labelagent_preserved_model_comparison_20261105/inputs/camera0_frame76.png').convert('RGB')
 cases={'gray':Image.new('RGB',(256,256),'gray'),'white':Image.new('RGB',(256,256),'white'),
  'raw_pallet':raw.crop((260,100,480,370)), 'raw_floor':raw.crop((250,850,700,1200)), 'raw_robot':raw.crop((641,398,929,665))}
 rows=[]
 for name,im in cases.items():
  im.save(OUT/(name+'.png'))
  for mode,prompt,schema in [('old_schema_without_text_contract',OLD,SCHEMA),
   ('explicit_schema_contract',OLD+' Return exactly a JSON object with keys object and category. object is a concrete description of visible pixels, never a category token. category is one of '+json.dumps(CATEGORIES)+'. The required response schema is '+json.dumps(SCHEMA),SCHEMA),
   ('neutral_caption_only','Describe the actual visible colors, material, and physical object in this image. If no object or structure is visible, say uniform blank image. Do not assume any object exists. Return JSON with exactly one key object containing a short concrete description.',CAPTION)]:
   file=OUT/(name+'_'+mode+'.json')
   r=json.loads(file.read_text(encoding='utf-8')) if file.exists() else request(im,prompt,schema)
   save(file,r);row={'case':name,'mode':mode,**r};rows.append(row);save(OUT/'progress.json',rows)
   print(json.dumps({'case':name,'mode':mode,'decision':r['decision']},ensure_ascii=False),flush=True)
 save(OUT/'completed.json',{'status':'complete','rows':rows,'same_model_and_image_per_ablation':True,'synthetic_controls_are_functional_not_accuracy':True})
if __name__=='__main__':main()
