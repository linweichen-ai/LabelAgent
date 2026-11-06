"""Object-text proposals avoid manually repairing invalid VLM coordinates."""
import sys,json,hashlib,copy,time,urllib.request
from pathlib import Path
from plan_visible_bases import ROOT,work,out,read,save
PROMPT='a single pallet base, including its visible deck and support feet, excluding cargo and neighboring objects'
def main():
 cfg=read(work/'config.json');sys.path.insert(0,cfg['source_snapshot']);import numpy as np
 from PIL import Image
 from source.agent import workflow
 from source.agent.edit_agent import _rasterize
 from wire_geometry_policy import map_complete_response
 from visible_base_policy import touches_crop_border,png
 dest=out/'text_generation';dest.mkdir(exist_ok=True)
 if (dest/'completed.json').exists():raise FileExistsError('Frozen text proposals')
 cohort=read(out/'native_inputs/cohort.json');config={'sam3_url':cfg['sam_url']};trace={};rows=[];hyps=[]
 save(dest/'frozen_plan.json',{'source_cohort_sha256':hashlib.sha256((out/'native_inputs/cohort.json').read_bytes()).hexdigest(),'fixed_text_prompt':PROMPT,'image_only_source_regions':8,'requests_max':8,'no_VLM_coordinate_repair':True,'no_reference_positions':True,'no_mask_prior':True,'complete_mask_required_not_touching_crop_border':True,'no_budget_clipping':True,'busy_retry_interval_seconds':.2})
 def log(r):trace[r['id']]=copy.deepcopy(r);save(dest/'model_trace.json',list(trace.values()))
 with workflow.trace_calls(log):
  for seed in cohort:
   box=seed['binding']['source_box'];image=Image.open(Path(seed['folder'])/'SAM_source_input.png').convert('RGB');folder=dest/seed['case']/f"seed{seed['index']:02d}";folder.mkdir(parents=True,exist_ok=True);raw=png(image);(folder/'input.png').write_bytes(raw);save(folder/'prompt.json',{'source_box':box,'text':PROMPT,'seed_key':seed['key'],'mask_prior':None})
   for _ in range(900):
    with urllib.request.urlopen('http://127.0.0.1:8765/api/run/status',timeout=15) as f:busy=json.load(f).get('status')=='running'
    if busy:time.sleep(3);continue
    try:response=workflow.sam_request(config,'/segment_text',raw,prompt=PROMPT);break
    except ValueError as e:
     if 'Another model request' in str(e):time.sleep(.2);continue
     response={'rejected_reason':str(e)};break
   else:raise RuntimeError('Shared model busy')
   save(folder/'response.json',response);valid=0;rejections=[]
   for j,result in enumerate(response.get('results',[])):
    candidate=dict(result)
    if not candidate.get('parts') and candidate.get('points'):candidate['parts']=[{'points':candidate['points'],'holes':candidate.get('holes',[])}]
    try:
     if result.get('warnings') or response.get('warnings'):raise ValueError('Warned output retained but not admitted')
     shape,m,proof=map_complete_response(candidate,box,1920,1536)
     if touches_crop_border(_rasterize(candidate,image.width,image.height)):raise ValueError('Complete raw output reaches crop border')
    except ValueError as e:rejections.append({'result_index':j,'reason':str(e)})
    else:
     key=hashlib.sha256(seed['case'].encode()+m.tobytes()).hexdigest();hyps.append({'key':key,'case':seed['case'],'shape':shape,'score':float(result.get('score',0)),'source':'visible_structure_text_SAM','source_prediction':str(folder/'response.json'),'source_result_index':j,'source_box':box,'wire_proof':proof,'seed_key':seed['key']});valid+=1
   rows.append({'case':seed['case'],'index':seed['index'],'folder':str(folder),'raw_results':len(response.get('results',[])),'valid_outputs':valid,'rejections':rejections,'error':response.get('rejected_reason')});save(dest/'progress.json',rows);print(json.dumps({'seed':seed['index'],'raw_results':rows[-1]['raw_results'],'valid_outputs':valid,'rejections':rejections},ensure_ascii=False),flush=True)
 save(dest/'all_hypotheses.json',hyps);save(dest/'completed.json',{'status':'complete','real_scene_SAM_requests':len(rows),'valid_complete_outputs':len(hyps),'model_trace_records':len(trace),'reference_inputs':False,'no_output_clipping':True,'fixed_prompt':PROMPT});print(json.dumps(read(dest/'completed.json')),flush=True)
if __name__=='__main__':main()
