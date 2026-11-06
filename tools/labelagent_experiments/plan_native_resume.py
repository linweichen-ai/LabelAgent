"""Run the previously unexecuted native-input experiment with explicit budgets."""
import sys,hashlib,copy,urllib.request,json
from pathlib import Path
from native_resume_common import *

def main():
 cfg=read(WORK/'config.json');sys.path.insert(0,cfg['source_snapshot'])
 import numpy as np
 from PIL import Image
 from source.agent import workflow
 from native_resume_policy import planning_views,bind_points,PLAN_PROMPT,PLAN_SCHEMA,png
 dest=OUT/'planning';dest.mkdir(exist_ok=True)
 if (dest/'completed.json').exists():raise FileExistsError('Completed plans immutable')
 seeds=read(OLD/'native_inputs/cohort.json')
 config={'llm_url':cfg['vlm_url'],'llm_model':cfg['vlm_model'],'llm_key':'local','llm_max_tokens':512,'llm_prompt_profile':'compact'}
 frozen={'cohort':[{'index':r['index'],'case':r['case'],'key':r['key'],'source':r['source']} for r in seeds],'prompt':PLAN_PROMPT,'schema':PLAN_SCHEMA,'policy_sha256':sha(Path(__file__).with_name('native_resume_policy.py')),'reference_inputs':False,'model_output_max_tokens':512,'llm_prompt_profile':'compact','image_views_per_request':3,'seed_must_remain_complete':True}
 if (dest/'frozen_plan.json').exists():assert read(dest/'frozen_plan.json')==frozen
 else:save(dest/'frozen_plan.json',frozen)
 trace={r['id']:r for r in read(dest/'model_trace.json')} if (dest/'model_trace.json').exists() else {}
 wire=read(dest/'request_receipts.json') if (dest/'request_receipts.json').exists() else []
 rawopen=urllib.request.urlopen
 def observed(req,*a,**kw):
  if isinstance(req,urllib.request.Request) and req.full_url.endswith('/chat/completions'):
   body=json.loads(req.data);assert body['max_tokens']==512 and body['temperature']==0
   images=[v['image_url']['url'] for v in body['messages'][0]['content'] if v['type']=='image_url'];assert len(images)==3
   wire.append({'max_tokens':body['max_tokens'],'temperature':0,'input_images':len(images),'image_data_url_sha256':[hashlib.sha256(v.encode()).hexdigest() for v in images],'model':body['model'],'response_schema_sha256':hashlib.sha256(json.dumps(body['response_format'],sort_keys=True).encode()).hexdigest()})
   save(dest/'request_receipts.json',wire)
  return rawopen(req,*a,**kw)
 urllib.request.urlopen=observed
 original=workflow._chat
 workflow._chat=lambda *a,**kw:cooperative_call(original,*a,**kw)
 def log(r):trace[r['id']]=copy.deepcopy(r);save(dest/'model_trace.json',list(trace.values()))
 rows=[];bindings=[]
 with workflow.trace_calls(log):
  for seed in seeds:
   source=Image.open(INC/'inputs'/(seed['case']+'.png')).convert('RGB');m=(np.asarray(Image.open(Path(seed['folder'])/'seed_mask.png'))>0).astype('uint8')
   views,details,binding=planning_views(source,m);folder=dest/f"seed{seed['index']:02d}";folder.mkdir(exist_ok=True)
   record={**seed,'folder':str(folder),'binding':binding};bindings.append(record);save(folder/'binding.json',record)
   Image.fromarray(m*255).save(folder/'seed_mask.png');source.crop(tuple(binding['source_box'])).save(folder/'SAM_source_input.png')
   for j,im in enumerate(details):im.save(folder/f'detail{j}.png')
   for cue,im in views.items():
    file=folder/(cue+'_decision.json');im.save(folder/(cue+'.png'))
    if file.exists():row=read(file)
    else:
     row={'index':seed['index'],'case':seed['case'],'key':seed['key'],'cue':cue,'folder':str(folder),'binding':binding,'source_prediction':seed.get('source_prediction')}
     try:
      decision=workflow.chat(config,PLAN_PROMPT,png(im),extra_images=[png(d) for d in details],response_schema=PLAN_SCHEMA)
      points,why=bind_points(decision,binding,m);row.update(decision=decision,points=points,rejection=why,valid_prompt=points is not None)
     except ValueError as e:row.update(error=str(e),points=None,valid_prompt=False,rejection='model transport/schema failure retained')
     save(file,row)
    rows.append(row);save(dest/'progress.json',rows)
    print(json.dumps({'seed':seed['index'],'cue':cue,'object':row.get('decision',{}).get('object'),'scope':row.get('decision',{}).get('instance_scope'),'valid':row['valid_prompt'],'reason':row.get('rejection')},ensure_ascii=False),flush=True)
 save(dest/'cohort.json',bindings)
 save(dest/'completed.json',{'status':'complete','cases':len(seeds),'point_plan_responses':len(rows),'valid_prompts':sum(r['valid_prompt'] for r in rows),'model_trace_records':len(trace),'request_transmissions':len(wire),'request_max_tokens_verified':all(r['max_tokens']==512 for r in wire),'reference_inputs':False,'annotation_writes':0})
 print(json.dumps(read(dest/'completed.json')),flush=True)
if __name__=='__main__':main()
