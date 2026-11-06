"""Isolated continuation; never mutate the sealed visible-base attempt."""
from pathlib import Path
import json, hashlib, shutil, time, urllib.request, datetime, copy

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'task_workspaces/labelagent_native_resume_20261107'
OUT=ROOT/'outputs/labelagent_native_resume_20261107'
OLD=ROOT/'outputs/labelagent_visible_base_20261107'
INC=ROOT/'outputs/labelagent_context_comparison_20261105'
work,out=WORK,OUT
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 temp=p.with_name(p.name+'.pending');temp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def get(url):
 with urllib.request.urlopen(url,timeout=15) as f:return json.load(f)
def prepare():
 if (WORK/'config.json').exists():raise FileExistsError('Preparation already frozen')
 WORK.mkdir(parents=True,exist_ok=True);OUT.mkdir()
 previous=ROOT/'task_workspaces/labelagent_visible_base_20261107'
 shutil.copytree(previous/'source_snapshot',WORK/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 cfg=read(previous/'config.json');cfg.update(source_snapshot=str(WORK/'source_snapshot'),output=str(ROOT/'outputs/labelagent_native_resume_comparison_20261107'),optimization_plan=str(OUT/'optimization_plan.json'))
 save(WORK/'config.json',cfg)
 plan=read(OLD/'optimization_plan.json');plan.update(experiment='native_coordinate_resume',predecessor_native_inference_responses=0,new_Mask2Former_forwards=0,point_planning='real square photo, explicit half-open edge coordinates 0..1000; local schema validation; no clamping or snapping',VLM_output_max_tokens=512,VLM_prompt_profile='compact',VLM_proposals_max=16,SAM_generation_max=16,blank_diagnostics_max=2,no_semantic_gate_override=True)
 save(OUT/'optimization_plan.json',plan);shutil.copy2(OLD/'incumbent_binding.json',OUT/'incumbent_binding.json')
 save(OUT/'runtime_before.json',{'workbench':get('http://127.0.0.1:8765/api/run/status'),'VLM':get('http://127.0.0.1:18868/health'),'checked_at':datetime.datetime.now().astimezone().isoformat()})
 protected={str(p):sha(p) for p in [Path(cfg['reference_annotations']),Path(cfg['reference_description']),Path(cfg['package'])/'model/model.safetensors',INC/'inputs/camera0_frame76.png',INC/'inputs/camera1_frame76.png']}
 state=get('http://127.0.0.1:8765/api/state');save(OUT/'protected_before.json',{'files':protected,'workbench':{k:state.get(k) for k in ('path','truth','reference_frames','topic')}})
 print(json.dumps({'prepared':str(OUT),'model':read(OUT/'runtime_before.json')['VLM'].get('model_config_sha256')}),flush=True)

def cooperative_call(fn,*args,**kwargs):
 # Model contention never consumes an Agent round and never stops another client.
 for _ in range(100):
  if (OUT/'STOP').exists():raise RuntimeError('Explicit experiment STOP')
  if get('http://127.0.0.1:8765/api/run/status').get('status')=='running':time.sleep(3);continue
  try:return fn(*args,**kwargs)
  except ValueError as e:
   if 'Another model request' not in str(e):raise
   time.sleep(3)
 raise RuntimeError('Shared model busy; completed responses preserved for missing-only resume')

if __name__=='__main__':prepare()
