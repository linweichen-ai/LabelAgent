"""Fixed-cohort point-ID experiment; previous sealed runs are read-only."""
from pathlib import Path
import json,hashlib,shutil,time,urllib.request,datetime
ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'task_workspaces/labelagent_point_ids_20261107'
OUT=ROOT/'outputs/labelagent_point_ids_20261107'
PREVIOUS=ROOT/'outputs/labelagent_native_resume_20261107'
INC=ROOT/'outputs/labelagent_context_comparison_20261105'
work,out=WORK,OUT
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.pending');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def get(url):
 with urllib.request.urlopen(url,timeout=15) as f:return json.load(f)
def cooperative(fn,*a,**kw):
 for _ in range(1500):
  if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
  if get('http://127.0.0.1:8765/api/run/status').get('status')=='running':time.sleep(.2);continue
  try:return fn(*a,**kw)
  except ValueError as e:
   if 'Another model request' not in str(e):raise
   time.sleep(.2)
 raise RuntimeError('Shared model busy; keep responses and resume only missing requests')
def prepare():
 if (WORK/'config.json').exists():raise FileExistsError('Experiment already prepared')
 WORK.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 oldwork=ROOT/'task_workspaces/labelagent_native_resume_20261107'
 shutil.copytree(oldwork/'source_snapshot',WORK/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 cfg=read(oldwork/'config.json');cfg.update(source_snapshot=str(WORK/'source_snapshot'),output=str(ROOT/'outputs/labelagent_point_ids_comparison_20261107'),optimization_plan=str(OUT/'optimization_plan.json'));save(WORK/'config.json',cfg)
 plan=read(PREVIOUS/'optimization_plan.json');plan.update(experiment='fixed_image_point_ID_selection',point_planning='choose fixed image-derived point IDs on a clean photo plus same source locator/directory view; no numerical model coordinates',cohort='same eight image-only seeds as native-coordinate experiment',real_planning_requests_max=16,synthetic_planning_controls_max=4,VLM_proposals_max=20,SAM_generation_max=16,blank_diagnostics_max=2,changed_factors=['point representation','view count and layout'],not_single_factor_ablation=True,reference_points_for_generation=False,model_output_limit=512,temperature=0,original_acceptance_unchanged=True,planner_controls='diagnostic and fail-closed on PAL hallucination; never substitute for original downstream gates',temporal_audit='separate read-only same-camera RGB diagnostic, not proposal masks')
 save(OUT/'optimization_plan.json',plan);shutil.copy2(PREVIOUS/'incumbent_binding.json',OUT/'incumbent_binding.json')
 state=get('http://127.0.0.1:8765/api/state');protected={str(p):sha(p) for p in [Path(cfg['reference_annotations']),Path(cfg['reference_description']),Path(cfg['package'])/'model/model.safetensors',INC/'inputs/camera0_frame76.png',INC/'inputs/camera1_frame76.png']}
 save(OUT/'protected_before.json',{'files':protected,'workbench':{k:state.get(k) for k in ('path','truth','reference_frames','topic')}})
 save(OUT/'runtime_before.json',{'VLM':get('http://127.0.0.1:18868/health'),'workbench':get('http://127.0.0.1:8765/api/run/status'),'at':datetime.datetime.now().astimezone().isoformat()})
 print(json.dumps({'prepared':str(OUT),'new_Mask2Former_forwards':0,'real_planning_limit':16,'Agent_rounds':3}),flush=True)
if __name__=='__main__':prepare()
