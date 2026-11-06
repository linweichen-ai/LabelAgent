"""Fixed image point catalog; model IDs are never completed or repaired."""
import sys,copy,urllib.request,hashlib,json
from pathlib import Path
from point_ids_common import *

def main():
 cfg=read(WORK/'config.json');sys.path.insert(0,cfg['source_snapshot'])
 import numpy as np
 from PIL import Image
 from source.agent import workflow
 from point_id_policy import make_views,schema_for,bind_ids,PLAN_PROMPT,png
 from semantic_subject_guard import compatible_pallet_subject
 dest=OUT/'planning';controls=OUT/'planner_controls';dest.mkdir(exist_ok=True);controls.mkdir(exist_ok=True)
 if (dest/'completed.json').exists():raise FileExistsError('Completed planner is immutable')
 seeds=read(PREVIOUS/'planning/cohort.json');assert len(seeds)==8
 config={'llm_url':cfg['vlm_url'],'llm_model':cfg['vlm_model'],'llm_key':'local','llm_max_tokens':512,'llm_prompt_profile':'compact'}
 prepared=[];bindings=[];inputs=[]
 for seed in seeds:
  source=Image.open(INC/'inputs'/(seed['case']+'.png')).convert('RGB');mask=(np.asarray(Image.open(Path(seed['folder'])/'seed_mask.png'))>0).astype('uint8')
  variants=[('real',source,dest/f"seed{seed['index']:02d}")]
  if seed['index'] in [1,7]:variants.append(('synthetic_gray' if seed['index']==1 else 'synthetic_white',Image.new('RGB',source.size,(127,127,127) if seed['index']==1 else (255,255,255)),controls/f"seed{seed['index']:02d}"))
  for kind,photo,folder in variants:
   marked,clean,binding=make_views(photo,mask);folder.mkdir(exist_ok=True);Image.fromarray(mask*255).save(folder/'seed_mask.png');photo.crop(tuple(binding['source_box'])).save(folder/'SAM_source_input.png');clean.save(folder/'clean.png')
   record={**seed,'folder':str(folder),'binding':binding,'kind':kind};save(folder/'binding.json',record)
   if kind=='real':bindings.append(record)
   for cue,image in marked.items():
    image.save(folder/(cue+'.png'));prepared.append((record,cue,clean,image,schema_for(binding),mask))
   for file in sorted(folder.glob('*.png')):inputs.append({'path':str(file),'sha256':sha(file)})
 frozen={'cohort':[{'index':r['index'],'case':r['case'],'key':r['key']} for r in seeds],'prompt':PLAN_PROMPT,'policy_sha256':sha(Path(__file__).with_name('point_id_policy.py')),'input_sha256':inputs,'max_tokens':512,'images_per_request':2,'real_requests':16,'synthetic_control_requests':4,'reference_inputs':False,'points_repaired_or_supplemented':False,'two_changes_not_single_factor_ablation':True}
 if (dest/'frozen_plan.json').exists():assert read(dest/'frozen_plan.json')==frozen
 else:save(dest/'frozen_plan.json',frozen)
 trace={r['id']:r for r in read(dest/'model_trace.json')} if (dest/'model_trace.json').exists() else {}
 receipts=read(dest/'request_receipts.json') if (dest/'request_receipts.json').exists() else [];context={};rawopen=urllib.request.urlopen;original=workflow._chat
 def observed(req,*a,**kw):
  if isinstance(req,urllib.request.Request) and req.full_url.endswith('/chat/completions'):
   payload=json.loads(req.data);assert payload['max_tokens']==512 and payload['temperature']==0
   urls=[x['image_url']['url'] for x in payload['messages'][0]['content'] if x['type']=='image_url'];assert len(urls)==2
   row={**context,'max_tokens':payload['max_tokens'],'temperature':0,'image_data_url_sha256':[hashlib.sha256(v.encode()).hexdigest() for v in urls],'input_images':len(urls),'schema_sha256':hashlib.sha256(json.dumps(payload['response_format'],sort_keys=True).encode()).hexdigest()};receipts.append(row)
   try:
    result=rawopen(req,*a,**kw);row['http_status']=getattr(result,'status',200);save(dest/'request_receipts.json',receipts);return result
   except Exception as e:
    row['http_status']=getattr(e,'code',None);row['transport_error_type']=type(e).__name__;save(dest/'request_receipts.json',receipts);raise
  return rawopen(req,*a,**kw)
 urllib.request.urlopen=observed;workflow._chat=lambda *a,**kw:cooperative(original,*a,**kw)
 def log(r):trace[r['id']]={**copy.deepcopy(r),'experiment_context':dict(context)};save(dest/'model_trace.json',list(trace.values()))
 rows=[];blankrows=[]
 # Fixed real/blank order is declared in prepared; never choose the next target from scores.
 with workflow.trace_calls(log):
  for record,cue,clean,marked,schema,mask in prepared:
   context.clear();context.update(kind=record['kind'],seed=record['index'],cue=cue);folder=Path(record['folder']);file=folder/(cue+'_decision.json')
   if file.exists():row=read(file)
   else:
    row={'index':record['index'],'case':record['case'],'key':record['key'],'cue':cue,'folder':str(folder),'binding':record['binding'],'kind':record['kind']}
    try:
     decision=workflow.chat(config,PLAN_PROMPT,png(clean),extra_images=[png(marked)],response_schema=schema);points,why=bind_ids(decision,record['binding'],mask);row.update(decision=decision,points=points,rejection=why,valid_prompt=points is not None)
    except ValueError as e:row.update(error=str(e),points=None,valid_prompt=False,rejection='model transport/schema failure retained')
    save(file,row)
   if row['kind']=='real':rows.append(row);save(dest/'progress.json',rows)
   else:
    row['claims_pallet']=compatible_pallet_subject(row.get('decision',{}));row['abstains_with_empty_points']=row.get('decision',{}).get('instance_scope')!='single_visible_object' and not row.get('decision',{}).get('positive_ids') and not row.get('decision',{}).get('negative_ids');blankrows.append(row);save(controls/'progress.json',blankrows)
   print(json.dumps({'kind':row['kind'],'seed':row['index'],'cue':cue,'object':row.get('decision',{}).get('object'),'scope':row.get('decision',{}).get('instance_scope'),'valid':row['valid_prompt'],'reason':row['rejection']},ensure_ascii=False),flush=True)
 assert len(rows)==16 and len(blankrows)==4
 save(dest/'cohort.json',bindings);save(controls/'completed.json',{'status':'complete','responses':4,'PAL_hallucinations':sum(r['claims_pallet'] for r in blankrows),'all_abstain_with_empty_points':all(r['abstains_with_empty_points'] for r in blankrows),'PAL_guard_passed':not any(r['claims_pallet'] for r in blankrows),'synthetic_functional_diagnostic_not_accuracy':True})
 save(dest/'completed.json',{'status':'complete','cases':8,'point_plan_responses':len(rows),'synthetic_control_responses':len(blankrows),'valid_prompts':sum(r['valid_prompt'] for r in rows),'model_trace_records':len(trace),'request_transmissions':len(receipts),'request_max_tokens_verified':all(r['max_tokens']==512 for r in receipts),'reference_inputs':False,'annotation_writes':0})
 print(json.dumps(read(dest/'completed.json')),flush=True)
if __name__=='__main__':main()
