"""Open object naming and visible structural points, with no reference positions."""
import sys,json,hashlib,copy,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'task_workspaces/labelagent_visible_base_20261107';out=ROOT/'outputs/labelagent_visible_base_20261107'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 cfg=read(work/'config.json');sys.path.insert(0,cfg['source_snapshot']);import numpy as np
 from PIL import Image
 from source.agent import workflow
 from source.agent.edit_agent import _rasterize
 from instance_separation_geometry import exact_binary_shape
 from visible_base_policy import planning_views,bind_points,PLAN_PROMPT,PLAN_SCHEMA,png
 old=ROOT/'outputs/labelagent_target_identity_20261106';previous=ROOT/'outputs/labelagent_instance_separation_20261105';seeds=read(old/'seam_generation/all_hypotheses.json');strip=next(r for r in read(previous/'seam_generation/all_hypotheses.json') if r['key'].startswith('fb1434b8'));seeds.append(strip);prompt=read(Path(strip['source_prediction']).parent/'prompt.json');parent=next(r for r in read(previous/'parents/completed.json')['rows'] if r['key']==prompt['source_parent']);mask=(np.asarray(Image.open(Path(parent['folder'])/'mask.png'))>0).astype('uint8');shape=exact_binary_shape(mask);assert shape is not None;seeds.append({'case':strip['case'],'shape':shape,'key':'coarse_'+parent['key'],'source':'old_image_only_parent','source_prediction':str(Path(parent['folder'])/'mask.png')});assert len(seeds)==8
 dest=out/'planning';dest.mkdir(exist_ok=True)
 if (dest/'completed.json').exists():raise FileExistsError('Frozen point plans')
 save(dest/'frozen_plan.json',{'cohort':[{'case':r['case'],'key':r['key'],'source':r['source']} for r in seeds],'prompt':PLAN_PROMPT,'schema':PLAN_SCHEMA,'policy_sha256':hashlib.sha256(Path(__file__).with_name('visible_base_policy.py').read_bytes()).hexdigest(),'reference_inputs':False,'complete_physical_object_identity_not_granted_by_points':True,'model_output_max_tokens':512,'image_views_per_request':3})
 config={'llm_url':cfg['vlm_url'],'llm_model':cfg['vlm_model'],'llm_key':'local','llm_max_tokens':512};original=workflow._chat;trace={};rows=[];bindings=[]
 def cooperative(*a,**kw):
  for _ in range(100):
   with urllib.request.urlopen('http://127.0.0.1:8765/api/run/status',timeout=15) as f:busy=json.load(f).get('status')=='running'
   if busy:time.sleep(3);continue
   try:return original(*a,**kw)
   except ValueError as e:
    if 'Another model request' not in str(e):raise
    time.sleep(3)
  raise RuntimeError('Shared model busy')
 workflow._chat=cooperative
 def log(r):trace[r['id']]=copy.deepcopy(r);save(dest/'model_trace.json',list(trace.values()))
 with workflow.trace_calls(log):
  for i,seed in enumerate(seeds):
   source=Image.open(ROOT/'outputs/labelagent_context_comparison_20261105/inputs'/(seed['case']+'.png')).convert('RGB');m=_rasterize(seed['shape'],1920,1536);views,details,binding=planning_views(source,m);folder=dest/f'seed{i+1:02d}';folder.mkdir(exist_ok=True);Image.fromarray(m*255).save(folder/'seed_mask.png');bindings.append({'index':i+1,'case':seed['case'],'key':seed['key'],'source':seed['source'],'folder':str(folder),'binding':binding});save(folder/'binding.json',bindings[-1]);source.crop(tuple(binding['source_box'])).save(folder/'SAM_source_input.png')
   for j,im in enumerate(details):im.save(folder/f'detail{j}.png')
   for cue,im in views.items():
    im.save(folder/(cue+'.png'));row={'index':i+1,'case':seed['case'],'key':seed['key'],'cue':cue,'folder':str(folder),'binding':binding,'source_prediction':seed['source_prediction']}
    try:
     decision=workflow.chat(config,PLAN_PROMPT,png(im),extra_images=[png(d) for d in details],response_schema=PLAN_SCHEMA);points,why=bind_points(decision,binding,m);row.update(decision=decision,points=points,rejection=why,valid_prompt=points is not None)
    except ValueError as e:row.update(error=str(e),points=None,valid_prompt=False,rejection='model transport/schema failure retained')
    save(folder/(cue+'_decision.json'),row);rows.append(row);save(dest/'progress.json',rows);print(json.dumps({'seed':i+1,'cue':cue,'object':row.get('decision',{}).get('object'),'scope':row.get('decision',{}).get('instance_scope'),'valid_prompt':row['valid_prompt'],'rejection':row.get('rejection')},ensure_ascii=False),flush=True)
 save(dest/'cohort.json',bindings);save(dest/'completed.json',{'status':'complete','cases':len(seeds),'point_plan_requests':len(rows),'valid_prompts':sum(r['valid_prompt'] for r in rows),'model_trace_records':len(trace),'reference_inputs':False,'no_annotation_writes':True});print(json.dumps(read(dest/'completed.json')),flush=True)
if __name__=='__main__':main()
