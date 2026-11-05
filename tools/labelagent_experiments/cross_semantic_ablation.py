"""Cross prompt and enum independently; do not alter the shared backend."""
from pathlib import Path
from PIL import Image
import json
from diagnose_semantic_contract import request,save,OLD,SCHEMA,CAPTION
from semantic_evidence_gate import CAPTION_PROMPT
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'outputs/labelagent_semantic_recovery_20261105/cross_ablation';out.mkdir(exist_ok=True)
if (out/'completed.json').exists():raise FileExistsError('Keep controls')
rows=[]
for color in ['gray','white']:
 im=Image.new('RGB',(256,256),color)
 for mode,prompt,schema in [('old_prompt_no_category_enum',OLD+' Return JSON with exactly one key object containing the visible description.',CAPTION),('neutral_prompt_with_category_enum',CAPTION_PROMPT,SCHEMA)]:
  file=out/(color+'_'+mode+'.json');r=json.loads(file.read_text(encoding='utf-8')) if file.exists() else request(im,prompt,schema);save(file,r)
  rows.append({'case':color,'mode':mode,**r});save(out/'progress.json',rows);print(json.dumps({'case':color,'mode':mode,'decision':r['decision']},ensure_ascii=False),flush=True)
save(out/'completed.json',{'status':'complete','rows':rows,'purpose':'diagnostic2x2 prompt/decoder crossing, not classifier accuracy'})
