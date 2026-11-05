"""Corrected complete-response variant; all SAM forwards reuse exact saved calls."""
import json,sys,hashlib,copy,shutil,io,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'task_workspaces/labelagent_projection_recovery_20261105';parent=ROOT/'outputs/labelagent_projection_recovery_20261105';out=parent/'instance_generation_wire_exact'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 if out.exists():raise FileExistsError('Preserve corrected generation')
 out.mkdir();cfg=read(work/'config.json');sys.path.insert(0,cfg['source_snapshot']);import numpy as np
 from PIL import Image
 from source.agent import workflow
 from source.agent.edit_agent import _rasterize
 from wire_geometry_policy import map_complete_response
 from incumbent_append_policy import append_complete_masks
 from semantic_evidence_gate import classify_neutral
 from semantic_subject_guard import compatible_pallet_subject
 for source in [parent/'instance_generation/visual_cache',parent/'instance_generation_blue/visual_cache']:shutil.copytree(source,out/'visual_cache',dirs_exist_ok=True)
 protocol={'reason':'earlier raster-contour reconstruction lost18 and added3 pixels in a selected SAM mask; earlier full comparison retained and rejected by exact-raster gate','mapping':'all original response polygon parts and holes offset into full original image; no contour reconstruction or part removal','new_SAM_forwards':0,'source_real_SAM_requests':48,'no_references_for_selection':True,'incumbent_all27_PAL_masks_retained':True,'all_acceptance_limits_unchanged':True,'same_neutral_evidence_and_subject_guard':True,'wire_policy_sha256':hashlib.sha256(Path(__file__).with_name('wire_geometry_policy.py').read_bytes()).hexdigest()};save(out/'generation_protocol.json',protocol)
 config={'sam3_url':cfg['sam_url'],'llm_url':cfg['vlm_url'],'llm_model':cfg['vlm_model'],'llm_key':'local','llm_prompt_profile':'compact','llm_max_tokens':256};traces={};original_chat=workflow._chat
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
 def log(r):traces[r['id']]=copy.deepcopy(r);save(out/'model_trace.json',list(traces.values()))
 images={};base={};original={};people={};pool={};mapped=[];invalid=[];inc=ROOT/'outputs/labelagent_context_comparison_20261105'
 for cam in [0,1]:
  case=f'camera{cam}_frame76';images[case]=Image.open(inc/'inputs'/(case+'.png')).convert('RGB');ss=read(inc/'C_hybrid'/case/'result.json')['shapes'];base[case]=[_rasterize(s,1920,1536) for s in ss if s['label']=='pallet'];people[case]=[_rasterize(s,1920,1536) for s in ss if s['label']=='person'];original[case]=[_rasterize(s,1920,1536) for s in read(inc/'A_current'/case/'result.json')['shapes'] if s['label']=='pallet'];pool[case]={}
  for r in read(parent/'instance_generation'/case/'all_hypotheses.json'):
   if r['source']=='unchanged_raw_trained_grid':pool[case][r['key']]={**r,'mask':_rasterize(r['shape'],1920,1536)}
 for source in [parent/'instance_generation',parent/'instance_generation_blue']:
  for r in read(source/'sam_progress.json'):
   if not r['technically_valid']:continue
   folder=Path(r['folder']);response=read(folder/'response.json');prompt=read(folder/'prompt.json');case=r['case']
   try:shape,mask,proof=map_complete_response(response,prompt['source_box'],1920,1536)
   except ValueError as e:invalid.append({'source':str(folder),'reason':str(e)});continue
   key=hashlib.sha256(case.encode()+mask.tobytes()).hexdigest();pool[case][key]={'case':case,'key':key,'shape':shape,'mask':mask,'score':float(response['score']),'source':'new_true_SAM_instance','source_prediction':str(folder/'response.json')};mapped.append({'case':case,'key':key,'source':str(folder/'response.json'),**proof})
 save(out/'all_wire_mapping_checks.json',{'passed':True,'rows':mapped,'invalid_complete_outputs_rejected':invalid})
 for case,rows in pool.items():save(out/case/'all_hypotheses.json',[{k:v for k,v in r.items() if k!='mask'} for r in rows.values()])
 def png(im):
  b=io.BytesIO();im.save(b,'PNG');return b.getvalue()
 banks={c:list(rows.values()) for c,rows in pool.items()};verified={};held=[]
 with workflow.trace_calls(log):
  for _ in range(100):
   selected,rejected,summary=append_complete_masks(banks,base,original,people);pending=[r for rows in selected.values() for r in rows if r['key'] not in verified]
   if not pending:break
   for r in pending:
    mask=_rasterize(r['shape'],1920,1536);yy,xx=np.where(mask);margin=max(32,int(max(xx.max()-xx.min(),yy.max()-yy.min())*.5));boxes=[[max(0,int(xx.min())-8),max(0,int(yy.min())-8),min(1920,int(xx.max())+9),min(1536,int(yy.max())+9)],[max(0,int(xx.min())-margin),max(0,int(yy.min())-margin),min(1920,int(xx.max())+margin+1),min(1536,int(yy.max())+margin+1)]];views=[]
    for b in boxes:
     photo=images[r['case']].crop(tuple(b));scale=max(1,512/max(photo.size));photo=photo.resize((round(photo.width*scale),round(photo.height*scale)),Image.Resampling.LANCZOS);views.append(classify_neutral(workflow,config,png(photo),out/'visual_cache'))
    passed=all(compatible_pallet_subject(v['decision']) for v in views);verified[r['key']]={'row':r,'views':views,'passed':passed,'source_boxes':boxes}
    if not passed:banks[r['case']]=[x for x in banks[r['case']] if x['key']!=r['key']];held.append({'case':r['case'],'key':r['key'],'captions':[v['decision'] for v in views]})
    save(out/'verification_progress.json',{'rows':list(verified.values())})
  else:raise RuntimeError('Selection did not stabilize')
 for case,rows in selected.items():save(out/case/'additional_children.json',[r['shape'] for r in rows])
 save(out/'selection.json',{'selected':selected,'rejected_whole_masks':rejected,'held_semantic':held,'summary':summary,'reference_inputs':False});save(out/'completed.json',{'status':'complete','fresh_SAM_requests':0,'source_SAM_requests':48,'fresh_VLM_trace_records':len(traces),'new_instances':{c:len(rows) for c,rows in selected.items()},'summary':summary,'whole_original_response_pixels_verified':True,'all_incumbent_masks_retained':True,'annotation_writes':0});print(json.dumps(read(out/'completed.json')),flush=True)
if __name__=='__main__':main()
