"""Freeze a separate subject-veto subset; no new pixels or reference access."""
import json,shutil,copy,hashlib,sys
from pathlib import Path
from semantic_subject_guard import compatible_pallet_subject
ROOT=Path(__file__).resolve().parents[2];parent=ROOT/'outputs/labelagent_semantic_recovery_20261105';source=parent/'instance_generation_contextual';out=parent/'instance_generation_subject_guarded'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert read(source/'completed.json')['status']=='complete'
if out.exists():raise FileExistsError('Preserve candidate variants')
shutil.copytree(source,out);selection=read(source/'C_selection.json');verified=read(source/'verification_progress.json');bykey={r['row']['key']:r for r in verified['rows']};veto=[]
for row in verified['rows']:
 row['before_subject_guard']=row['passed'];row['passed']=row['passed'] and all(compatible_pallet_subject(v['decision']) for v in row['views'])
for case,rows in selection['selected'].items():
 kept=[]
 for row in rows:
  evidence=bykey[row['key']]
  if all(compatible_pallet_subject(v['decision']) for v in evidence['views']):kept.append(row)
  else:veto.append({'case':case,'key':row['key'],'captions':[v['decision'] for v in evidence['views']],'reason':'main subject contradicts pallet identity or is explicitly uncertain; reject entire proposal'})
 selection['selected'][case]=kept;save(out/case/'C_children.json',[r['shape'] for r in kept])
# Area is recalculated from whole unchanged masks; never fabricate a smaller mask.
cfg=read(ROOT/'task_workspaces/labelagent_semantic_recovery_20261105/config.json');sys.path.insert(0,cfg['source_snapshot']);import numpy as np
from source.agent.edit_agent import _rasterize
for case,rows in selection['selected'].items():
 base=read(ROOT/'outputs/labelagent_preserved_model_comparison_20261105/A_current'/case/'result.json')['shapes'];masks=[_rasterize(s,1920,1536) for s in base if s['label']=='pallet']
 union=np.maximum.reduce(masks)
 for r in rows:union|=_rasterize(r['shape'],1920,1536)
 selection['summary'][case].update(new_count=len(rows),new_union_pixels=int(union.sum()))
selection['subject_vetoes']=veto;save(out/'C_selection.json',selection);save(out/'verification_progress.json',verified)
protocol=read(source/'generation_protocol.json');protocol.update(subject_guard='whole candidate rejected on explicit pallet jack/forklift subject or uncertainty; no new inference, no reference input',source_candidate_bank_sha256=hashlib.sha256((source/'C_selection.json').read_bytes()).hexdigest());save(out/'generation_protocol.json',protocol)
done=read(source/'completed.json');done.update(fresh_model_requests=0,subject_vetoes=len(veto),inherited_real_caption_requests=180,cases={case:{'B':0,'C':len(rows)} for case,rows in selection['selected'].items()});save(out/'completed.json',done);print(json.dumps(done),flush=True)
