"""Segment complete single-object hypotheses; never intersect with a budget mask."""
import json,sys,hashlib,copy,time,urllib.request
from pathlib import Path
from plan_visible_bases import ROOT,work,out,read,save
def main():
 cfg=read(work/'config.json');sys.path.insert(0,cfg['source_snapshot']);import numpy as np
 from PIL import Image
 from source.agent import workflow
 from source.agent.edit_agent import _rasterize
 from wire_geometry_policy import map_complete_response
 from visible_base_policy import touches_crop_border,png
 dest=out/'seam_generation_native';dest.mkdir(exist_ok=True)
 if (dest/'completed.json').exists():raise FileExistsError('Frozen segmentation')
 assert read(out/'planning_native/completed.json')['status']=='complete';plans=read(out/'planning_native/progress.json');config={'sam3_url':cfg['sam_url']};trace={};rows=[];hyps=[];blanks=[];requests=0
 save(dest/'frozen_plan.json',{'planning_completed_sha256':hashlib.sha256((out/'planning_native/completed.json').read_bytes()).hexdigest(),'mask_prior':None,'output_clipping':False,'reject_complete_outputs_touching_crop_border':True,'reference_inputs':False})
 def log(r):trace[r['id']]=copy.deepcopy(r);save(dest/'model_trace.json',list(trace.values()))
 def sam(raw,points):
  nonlocal requests
  for _ in range(100):
   with urllib.request.urlopen('http://127.0.0.1:8765/api/run/status',timeout=15) as f:busy=json.load(f).get('status')=='running'
   if busy:time.sleep(3);continue
   try:requests+=1;return workflow.sam_request(config,'/segment_points',raw,points=points)
   except ValueError as e:
    if 'Another model request' in str(e):time.sleep(3);continue
    return {'rejected_reason':str(e)}
  raise RuntimeError('Shared model busy')
 with workflow.trace_calls(log):
  for plan in plans:
   if not plan['valid_prompt']:continue
   box=plan['binding']['source_box'];image=Image.open(Path(plan['folder'])/'SAM_source_input.png').convert('RGB');points=[dict(p,x=p['x']-box[0],y=p['y']-box[1]) for p in plan['points']];folder=dest/plan['case']/(f"seed{plan['index']:02d}_"+plan['cue']);folder.mkdir(parents=True,exist_ok=True);raw=png(image);(folder/'input.png').write_bytes(raw);save(folder/'prompt.json',{'source_box':box,'points':points,'source_full_points':plan['points'],'mask_prior':None,'planner_decision':plan['decision'],'seed_key':plan['key']});response=sam(raw,points);save(folder/'response.json',response);valid=bool(response.get('parts') and not response.get('warnings') and not response.get('rejected_reason'));why=response.get('rejected_reason')
   if valid:
    try:
     shape,m,proof=map_complete_response(response,box,1920,1536)
     if touches_crop_border(_rasterize(response,image.width,image.height)):raise ValueError('Complete raw output reaches crop border; reject possible truncation')
    except ValueError as e:valid=False;why=str(e)
    else:
     key=hashlib.sha256(plan['case'].encode()+m.tobytes()).hexdigest();hyps.append({'key':key,'case':plan['case'],'shape':shape,'score':float(response.get('score',0)),'source':'visible_structure_SAM','source_prediction':str(folder/'response.json'),'source_box':box,'wire_proof':proof,'seed_key':plan['key'],'planner_scope':plan['decision']['instance_scope']})
   rows.append({'case':plan['case'],'index':plan['index'],'cue':plan['cue'],'folder':str(folder),'technically_valid':valid,'rejection':why,'warnings':response.get('warnings',[]),'points':len(points)});save(dest/'progress.json',rows);print(json.dumps({'seed':plan['index'],'cue':plan['cue'],'valid':valid,'rejection':why},ensure_ascii=False),flush=True)
   if len(blanks)<2:
    blank=png(Image.new('RGB',image.size,(127,127,127)));control=sam(blank,points);(folder/'synthetic_blank_input.png').write_bytes(blank);save(folder/'synthetic_blank_response.json',control);blanks.append({'case':plan['case'],'folder':str(folder),'nonempty_blank_mask':bool(control.get('parts')),'same_points_no_prior':True,'functional_control_not_real_scene_result':True})
 save(dest/'all_hypotheses.json',hyps);save(dest/'blank_controls.json',{'rows':blanks,'not_accuracy':True});save(dest/'completed.json',{'status':'complete','real_scene_SAM_requests':len(rows),'synthetic_control_requests':len(blanks),'valid_complete_outputs':len(hyps),'model_trace_records':len(trace),'no_prior_mask':True,'no_output_budget_trimming':True,'reference_inputs':False});print(json.dumps(read(dest/'completed.json')),flush=True)
if __name__=='__main__':main()
