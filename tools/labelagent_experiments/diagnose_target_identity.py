"""Bound new image-only observations; do not turn a diagnosis into acceptance."""
import sys,json,hashlib,copy,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'outputs/labelagent_target_identity_20261106';work=ROOT/'task_workspaces/labelagent_target_identity_20261106'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 cfg=read(work/'config.json');sys.path.insert(0,cfg['source_snapshot']);from source.agent import workflow
 from semantic_evidence_gate import classify_neutral
 from semantic_subject_guard import compatible_pallet_subject
 from target_identity_policy import EXTENT_PROMPT,EXTENT_SCHEMA
 folder=out/'target_12of16_diagnostic';assert not (folder/'completed.json').exists();config={'llm_url':cfg['vlm_url'],'llm_model':cfg['vlm_model'],'llm_key':'local','llm_max_tokens':256};original=workflow._chat;trace={}
 def chat(*a,**kw):
  for _ in range(100):
   try:return original(*a,**kw)
   except ValueError as e:
    if 'Another model request' not in str(e):raise
    time.sleep(3)
  raise RuntimeError('Shared model busy')
 workflow._chat=chat
 def log(r):trace[r['id']]=copy.deepcopy(r);save(folder/'model_trace.json',list(trace.values()))
 rows=[]
 with workflow.trace_calls(log):
  for name in ['near','context']:
   r=classify_neutral(workflow,config,(folder/(name+'.png')).read_bytes(),folder/'neutral_cache');rows.append({'view':name,'decision':r['decision'],'unchanged_subject_gate_passed':compatible_pallet_subject(r['decision'])});save(folder/'progress.json',rows);print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
  result=workflow.chat(config,EXTENT_PROMPT,(folder/'extent_photo.png').read_bytes(),extra_images=[(folder/'extent_binary.png').read_bytes()],response_schema=EXTENT_SCHEMA)
  save(folder/'extent_observation.json',{'decision':result,'prompt':EXTENT_PROMPT,'schema':EXTENT_SCHEMA,'photo_sha256':hashlib.sha256((folder/'extent_photo.png').read_bytes()).hexdigest(),'mask_sha256':hashlib.sha256((folder/'extent_binary.png').read_bytes()).hexdigest(),'diagnostic_only':True})
 save(folder/'completed.json',{'status':'complete','new_neutral_views':rows,'extent_observation':result,'old_judgments_preserved':True,'no_adoption_from_diagnostic':True,'model_trace_records':len(trace)});print(json.dumps(read(folder/'completed.json'),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
