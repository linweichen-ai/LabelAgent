"""Image-only single-instance gate before frozen final append predictions."""
import json,sys,io,hashlib,copy,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'task_workspaces/labelagent_native_resume_20261107';out=ROOT/'outputs/labelagent_native_resume_20261107';dest=out/'instance_generation'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 cfg=read(work/'config.json');sys.path.insert(0,cfg['source_snapshot']);import numpy as np
 from PIL import Image
 from source.agent import workflow
 from source.agent.edit_agent import _rasterize
 from semantic_evidence_gate import classify_neutral
 from semantic_subject_guard import compatible_pallet_subject
 from target_identity_policy import centered_photos,extent_views,EXTENT_PROMPT,EXTENT_SCHEMA
 from instance_identity_consistency import target_views,INSTANCE_PROMPT,INSTANCE_SCHEMA,png,single_physical_base,repeated_control_rejected
 from incumbent_append_policy import append_complete_masks
 if (dest/'completed.json').exists():raise FileExistsError('Preserve frozen reviewed candidates')
 assert read(out/'query_generation/completed.json')['status']==read(out/'seam_generation_all/completed.json')['status']=='complete'
 dest.mkdir(parents=True,exist_ok=True);config={'llm_url':cfg['vlm_url'],'llm_model':cfg['vlm_model'],'llm_key':'local','llm_prompt_profile':'compact','llm_max_tokens':256};traces={};original_chat=workflow._chat
 def cooperative(*a,**kw):
  for _ in range(100):
   with urllib.request.urlopen('http://127.0.0.1:8765/api/run/status',timeout=15) as f:busy=json.load(f).get('status')=='running'
   if busy:time.sleep(3);continue
   try:return original_chat(*a,**kw)
   except ValueError as e:
    if 'Another model request' not in str(e):raise
    time.sleep(3)
  raise RuntimeError('Shared model busy')
 workflow._chat=cooperative
 def log(r):traces[r['id']]=copy.deepcopy(r);save(dest/'model_trace.json',list(traces.values()))
 def structure(source,mask,folder):
  views,binding=target_views(source,mask);rows=[]
  for name,image in views.items():
   raw=png(image);key=hashlib.sha256(raw+INSTANCE_PROMPT.encode()+json.dumps(INSTANCE_SCHEMA,sort_keys=True).encode()).hexdigest();file=dest/'structure_cache'/(key+'.json');row=read(file) if file.exists() else {'decision':workflow.chat(config,INSTANCE_PROMPT,raw,response_schema=INSTANCE_SCHEMA),'source_image_sha256':hashlib.sha256(raw).hexdigest(),'schema':INSTANCE_SCHEMA,'prompt_sha256':hashlib.sha256(INSTANCE_PROMPT.encode()).hexdigest()};save(file,row);file.with_suffix('.png').write_bytes(raw);rows.append({'cue':name,**row,'file':str(file)})
  save(folder/'structure_evidence.json',{'binding':binding,'rows':rows});return rows,binding
 control_source=ROOT/'outputs/labelagent_semantic_recovery_20261105/neutral_controls';controls=[];pool={};base={};original={};people={};images={};inc=ROOT/'outputs/labelagent_context_comparison_20261105'
 with workflow.trace_calls(log):
  for name,expected in [('gray','not_single'),('white','not_single'),('raw_robot','not_single'),('pallet_stack','repeated'),('wood_pallet_base','single')]:
   image=Image.open(control_source/(name+'.png')).convert('RGB');mask=np.ones((image.height,image.width),np.uint8);folder=dest/'functional_controls'/name;folder.mkdir(parents=True,exist_ok=True);image.save(folder/'source.png');rows,binding=structure(image,mask,folder)
   if expected=='single':passed=all(single_physical_base(r['decision']) for r in rows)
   elif expected=='repeated':passed=all(repeated_control_rejected(r['decision']) for r in rows)
   else:passed=all(not single_physical_base(r['decision']) for r in rows)
   controls.append({'name':name,'expected_functional_status':expected,'decisions':[r['decision'] for r in rows],'passed':passed,'not_independent_accuracy':True});save(dest/'functional_controls/progress.json',controls);print(json.dumps({'functional_case':name,'passed':passed,'decisions':[r['decision'] for r in rows]},ensure_ascii=False),flush=True)
  controls_passed=all(r['passed'] for r in controls);save(dest/'functional_controls/completed.json',{'status':'complete','rows':controls,'all_passed':controls_passed,'source':'exposed old functional crops, not new human truth'})
  raw=read(out/'query_generation/all_hypotheses.json')+read(out/'seam_generation_all/old_and_new_hypotheses.json')
  for cam in [0,1]:
   case=f'camera{cam}_frame76';images[case]=Image.open(inc/'inputs'/(case+'.png')).convert('RGB');ss=read(inc/'C_hybrid'/case/'result.json')['shapes'];base[case]=[_rasterize(s,1920,1536) for s in ss if s['label']=='pallet'];people[case]=[_rasterize(s,1920,1536) for s in ss if s['label']=='person'];original[case]=[_rasterize(s,1920,1536) for s in read(inc/'A_current'/case/'result.json')['shapes'] if s['label']=='pallet'];unique={}
   for r in raw:
    if r['case']!=case:continue
    m=_rasterize(r['shape'],1920,1536);r={**r,'mask':m};key=r['key']
    if key not in unique or r['score']>unique[key]['score']:unique[key]=r
   pool[case]=list(unique.values());save(dest/case/'all_hypotheses.json',[{k:v for k,v in r.items() if k!='mask'} for r in pool[case]])
  verified={};held=[]
  for _ in range(100):
   selected,rejected,summary=append_complete_masks(pool,base,original,people);pending=[r for rows in selected.values() for r in rows if r['key'] not in verified]
   if not pending:break
   for row in pending:
    image=images[row['case']];m=_rasterize(row['shape'],1920,1536);views,center_binding=centered_photos(image,m);neutral=[]
    for name,photo in views.items():neutral.append(classify_neutral(workflow,config,png(photo),dest/'neutral_cache'))
    neutral_ok=all(compatible_pallet_subject(r['decision']) for r in neutral);structure_rows=[];binding=None
    if neutral_ok:structure_rows,binding=structure(image,m,dest/'candidate_evidence'/row['key'])
    single_ok=bool(structure_rows) and all(single_physical_base(r['decision']) for r in structure_rows);extent=None;extent_ok=False
    if neutral_ok and single_ok:
     pointed,binary,_=extent_views(image,m);extent=workflow.chat(config,EXTENT_PROMPT,png(pointed),extra_images=[png(binary)],response_schema=EXTENT_SCHEMA);extent_ok=extent.get('target_extent') in ['whole_visible_object','identifiable_occluded_object'] and compatible_pallet_subject({'object':extent.get('object','')})
    passed=neutral_ok and single_ok and controls_passed and extent_ok
    verified[row['key']]={'row':row,'neutral':[r['decision'] for r in neutral],'structure':[r['decision'] for r in structure_rows],'binding':binding,'neutral_passed':neutral_ok,'single_instance_passed':single_ok,'functional_controls_passed':controls_passed,'passed':passed,'center_binding':center_binding,'extent':extent,'extent_passed':extent_ok}
    if not passed:pool[row['case']]=[r for r in pool[row['case']] if r['key']!=row['key']];held.append({'case':row['case'],'key':row['key'],'reason':'single-instance evidence/controls not all passed','neutral':neutral_ok,'single':single_ok,'controls':controls_passed})
    save(dest/'verification_progress.json',{'rows':list(verified.values())});print(json.dumps({'candidate':row['key'][:8],'neutral':neutral_ok,'single':single_ok,'controls':controls_passed,'admitted':passed}),flush=True)
  else:raise RuntimeError('Selection did not stabilize')
 for case,rows in selected.items():save(dest/case/'additional_children.json',[r['shape'] for r in rows])
 save(dest/'selection.json',{'selected':selected,'rejected_whole_masks':rejected,'held_semantics':held,'summary':summary,'no_reference_inputs':True});save(dest/'generation_protocol.json',{'raw_trained_queries_source':str(out/'query_generation/completed.json'),'seam_SAM_source':str(out/'seam_generation_all/completed.json'),'single_instance_controls':str(dest/'functional_controls/completed.json'),'old_neutral_subject_gate_unchanged':True,'neutral_input_geometry':'foreground-anchor centered complete candidate, two unmarked contexts','complete_visible_object_extent_required':True,'single_base_gate_required':True,'whole_incumbent_retained':True,'same_original_global_area_count_budget':True,'no_budget_trimming':True,'reference_inputs':False});save(dest/'completed.json',{'status':'complete','model_trace_records':len(traces),'controls_passed':controls_passed,'new_instances':{c:len(rows) for c,rows in selected.items()},'summary':summary,'reference_inputs':False,'annotation_writes':0});print(json.dumps(read(dest/'completed.json')),flush=True)
if __name__=='__main__':main()
