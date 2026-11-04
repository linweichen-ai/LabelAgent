"""Matched actual SAM requests; only the dense prior is omitted."""
import sys,json,copy,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'task_workspaces/labelagent_recovery_20261105';out=ROOT/'outputs/labelagent_recovery_20261105/gap_no_prior_preflight'
sys.path.insert(0,str(work/'source_snapshot'))
from source.agent import workflow
from source.agent.edit_agent import _rasterize
from PIL import Image
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
out.mkdir(parents=True,exist_ok=True)
if (out/'completed.json').exists():raise FileExistsError('Preserve completed preflight')
cfg=read(work/'config.json');config={'sam3_url':cfg['sam_url']};old=ROOT/'outputs/labelagent_efficiency_partial_comparison_20261104';traces={}
def trace(row):traces[row['id']]=copy.deepcopy(row);save(out/'model_trace.json',list(traces.values()))
def sam(image,points):
 for _ in range(100):
  with urllib.request.urlopen('http://127.0.0.1:8765/api/run/status',timeout=10) as f:busy=json.load(f).get('status')=='running'
  if busy:time.sleep(3);continue
  try:return workflow.sam_request(config,'/segment_points',image,points=points)
  except ValueError as error:
   if 'Another model request' not in str(error):raise
   time.sleep(3)
 raise RuntimeError('Existing model service remained busy')
rows=[]
with workflow.trace_calls(trace):
 for cam in [0,1]:
  case=f'camera{cam}_frame76';image=(old/'inputs'/(case+'.png')).read_bytes();w,h=Image.open(old/'inputs'/(case+'.png')).size;seeds=read(old/'seed_stages'/case/'all_arms.json')
  for arm in ['B_sam7','C_hybrid']:
   audit=read(old/arm/case/'result.json')['agent']['audit'][0];op=audit['decision']['operations'][0];ids=op['reviewed_gap_ids'];holes=workflow.candidate_holes(seeds[arm],['floor'],w,h)
   file=out/(case+'_'+arm+'_response.json');response=read(file) if file.exists() else sam(image,op['points']);save(file,response)
   mask=_rasterize({'parts':response['parts']},w,h);Image.fromarray(mask*255).save(out/(case+'_'+arm+'_raw_mask.png'))
   coverage=[]
   for identifier in ids:
    gap=next(g for g in holes if g['id']==identifier);target=int(gap['mask'].sum());hit=int((mask&gap['mask']).sum());coverage.append({'hole_id':identifier,'target':target,'covered':hit,'fully_covered':target==hit})
   row={'case':case,'arm':arm,'original_ground_tasks':len(ids),'old_whole_image_complete_gaps':len(audit['tool_results'][0].get('resolved_gap_ids',[])),
    'no_prior_complete_gaps':sum(r['fully_covered'] for r in coverage),'warnings':response.get('warnings',[]),'rows':coverage,'points':op['points']};rows.append(row);save(out/'progress.json',{'rows':rows});print(json.dumps({k:row[k] for k in ['case','arm','original_ground_tasks','old_whole_image_complete_gaps','no_prior_complete_gaps','warnings']},ensure_ascii=False),flush=True)
save(out/'completed.json',{'status':'complete','rows':rows,'fresh_reference_scoring':False,'annotation_writes':0,'comparison':'same original image and original points; archived real dense-prior response vs actual no-prior response; edit bounds and full-region criterion unchanged'})
