"""Apply existing neutral, single-base and extent prompts to complete masks."""
import sys,copy,json,time,urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
from run_rgb_layers import ROOT,OUT,INC,read,save
def main():
 if (OUT/'semantic_review.json').exists():raise FileExistsError('Review immutable')
 cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');sys.path.insert(0,cfg['source_snapshot'])
 from source.agent import workflow
 from semantic_evidence_gate import classify_neutral
 from semantic_subject_guard import compatible_pallet_subject
 from target_identity_policy import centered_photos,extent_views,EXTENT_PROMPT,EXTENT_SCHEMA
 from instance_identity_consistency import target_views,INSTANCE_PROMPT,INSTANCE_SCHEMA,png,single_physical_base
 config={'llm_url':cfg['vlm_url'],'llm_model':cfg['vlm_model'],'llm_key':'local','llm_prompt_profile':'compact','llm_max_tokens':256}
 original=workflow._chat;trace=[]
 def cooperative(*a,**kw):
  for _ in range(1500):
   if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
   try:return original(*a,**kw)
   except ValueError as e:
    if 'Another model request' not in str(e):raise
    time.sleep(.2)
  raise RuntimeError('Model busy')
 workflow._chat=cooperative
 def log(r):trace.append(copy.deepcopy(r));save(OUT/'semantic_trace.json',trace)
 rows=[]
 with workflow.trace_calls(log):
  for row in read(OUT/'support_area_audit.json')['candidates']:
   if not row['valid']:continue
   m=np.load(Path(row['folder'])/'complete_full_mask.npy',allow_pickle=False);image=Image.open(INC/'inputs'/(row['case']+'.png')).convert('RGB');views,binding=centered_photos(image,m);neutral=[]
   for name,photo in views.items():neutral.append(classify_neutral(workflow,config,png(photo),OUT/'neutral_cache'))
   neutral_ok=all(compatible_pallet_subject(r['decision']) for r in neutral);structure=[];extent=None
   if neutral_ok:
    for name,photo in target_views(image,m)[0].items():structure.append(workflow.chat(config,INSTANCE_PROMPT,png(photo),response_schema=INSTANCE_SCHEMA))
   single_ok=bool(structure) and all(single_physical_base(d) for d in structure)
   if single_ok:
    pointed,binary,_=extent_views(image,m);extent=workflow.chat(config,EXTENT_PROMPT,png(pointed),extra_images=[png(binary)],response_schema=EXTENT_SCHEMA)
   result={'key':row['key'],'neutral':[r['decision'] for r in neutral],'neutral_passed':neutral_ok,'structure':structure,'single_passed':single_ok,'extent':extent,'extent_passed':bool(extent and extent.get('target_extent') in ['whole_visible_object','identifiable_occluded_object'] and compatible_pallet_subject({'object':extent.get('object','')})),'view_binding':binding,'original_prompts_unchanged':True}
   rows.append(result);save(OUT/'semantic_progress.json',rows);print(json.dumps(result,ensure_ascii=False),flush=True)
 save(OUT/'semantic_review.json',{'status':'complete','rows':rows,'actual_model_trace_records':len(trace),'adopted':False,'fresh_controls_run':0,'controls_scope':'old gates retained; no candidate eligible for adoption due original instance-support failure'})
if __name__=='__main__':main()
