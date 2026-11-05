"""Bounded first-round crop preflight on two formerly nonconvergent groups."""
import sys,json,copy,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'task_workspaces/labelagent_semantic_recovery_20261105';out=ROOT/'outputs/labelagent_semantic_recovery_20261105/spatial_preflight'
cfg=json.loads((work/'config.json').read_text(encoding='utf-8'));sys.path.insert(0,cfg['source_snapshot'])
import cv2,numpy as np
from PIL import Image
from source.agent import workflow
from source.agent.edit_agent import _rasterize
from ground_action_review_v5 import approved_gap_sam_operation
from spatial_gap_recovery import spatial_sam,choose_early_hard_fills
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
out.mkdir(parents=True,exist_ok=True)
if (out/'completed.json').exists():raise FileExistsError('Keep completed preflight')
traces={}
def trace(r):traces[r['id']]=copy.deepcopy(r);save(out/'model_trace.json',list(traces.values()))
def sam(*a,**kw):
 for _ in range(100):
  with urllib.request.urlopen('http://127.0.0.1:8765/api/run/status',timeout=10) as f:busy=json.load(f).get('status')=='running'
  if busy:time.sleep(3);continue
  try:return workflow.sam_request(*a,**kw)
  except ValueError as error:
   if 'Another model request' not in str(error):raise
   time.sleep(3)
 raise RuntimeError('Existing model remained busy')
old=ROOT/'outputs/labelagent_preserved_model_comparison_20261105';rows=[]
with workflow.trace_calls(trace):
 for arm in ['B_sam7','C_hybrid']:
  case='camera1_frame76';image=(old/'inputs'/(case+'.png')).read_bytes();w,h=Image.open(old/'inputs'/(case+'.png')).size
  shapes=read(old/'seed_stages'/case/'all_arms.json')[arm];audit=read(old/arm/case/'result.json')['agent']['audit'][0];assess=audit['decision']['hole_assessment'];holes=workflow.candidate_holes(shapes,['floor'],w,h)
  operations=[{'operation':'fill_gap','shape_ids':[g['shape_id']],'hole_id':g['id'],'reason':'Previously image-approved unchanged gap'} for g in holes if g['id'] in {a['hole_id'] for a in assess if a['semantic']=='ground'}]
  schedule=choose_early_hard_fills(holes,assess,operations,shapes,w,h,approved_gap_sam_operation)
  op=schedule[-1];prior=_rasterize(next(s for s in shapes if s['id']==op['shape_ids'][0]),w,h)
  for fill in schedule[:-1]:prior|=next(g['mask'] for g in holes if g['id']==fill['hole_id'])
  gaps=[g for g in holes if g['id'] in op['reviewed_gap_ids']]
  result=spatial_sam({'sam3_url':cfg['sam_url'],'sam_spatial_cache':str(out/'cache')},sam,image,prior,gaps,op['points']);raw=_rasterize(result,w,h)
  coverage=[{'id':g['id'],'area':g['area'],'covered':int((g['mask']&raw).sum()),'full':bool(np.all(raw[g['mask']!=0]))} for g in gaps]
  row={'arm':arm,'case':case,'individually_counted_fills':[o['hole_id'] for o in schedule[:-1]],'ops':len(schedule),'SAM_complete':sum(r['full'] for r in coverage),'SAM_tasks':len(gaps),'rows':coverage,'receipt':result['spatial_receipt'],'old_pending':len(audit['tool_results'][0]['unresolved_gap_ids'])}
  rows.append(row);save(out/'progress.json',rows);print(json.dumps(row),flush=True)
save(out/'completed.json',{'status':'complete','rows':rows,'annotation_writes':0,'new_agent_runs':0,'purpose':'source-image-only preflight; existing semantic approvals reused, not fresh human truth'})
