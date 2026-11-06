"""Visible seam conditioned multi-point SAM with no sliced mask as output/prior."""
import sys,json,hashlib,copy,io,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'task_workspaces/labelagent_target_identity_20261106';out=ROOT/'outputs/labelagent_target_identity_20261106'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 cfg=read(work/'config.json');sys.path.insert(0,cfg['source_snapshot']);import numpy as np
 from PIL import Image
 from source.agent import workflow
 from source.agent.edit_agent import _rasterize
 from target_identity_policy import deck_foot_prompts as visible_seam_prompts
 from wire_geometry_policy import map_complete_response
 dest=out/'seam_generation';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'completed.json').exists():raise FileExistsError('Preserve seam trials')
 parents=read(ROOT/'outputs/labelagent_instance_separation_20261105/parents/completed.json')['rows'];config={'sam3_url':cfg['sam_url']};inc=ROOT/'outputs/labelagent_context_comparison_20261105';traces={};rows=[];hypotheses=[];controls=[]
 def log(r):traces[r['id']]=copy.deepcopy(r);save(dest/'model_trace.json',list(traces.values()))
 def sam(image,points):
  for _ in range(100):
   with urllib.request.urlopen('http://127.0.0.1:8765/api/run/status',timeout=15) as f:busy=json.load(f).get('status')=='running'
   if busy:time.sleep(3);continue
   try:return workflow.sam_request(config,'/segment_points',image,points=points)
   except ValueError as e:
    if 'Another model request' in str(e):time.sleep(3);continue
    if any(w in str(e) for w in ['无法排除','提示点','没有找到目标']):return {'rejected_reason':str(e)}
    raise
  raise RuntimeError('Shared model remained busy')
 def png(im):
  b=io.BytesIO();im.save(b,'PNG');return b.getvalue()
 save(dest/'frozen_plan.json',{'parent_plan_sha256':hashlib.sha256((out/'parents/completed.json').read_bytes()).hexdigest(),'max_intervals_per_parent':2,'max_actual_points_per_request':7,'mask_prior':None,'no_sliced_mask_as_output':True,'synthetic_blank_controls':2,'pixel_exact_response_mapping':True,'no_reference_inputs':True,'deck_foot_policy_sha256':hashlib.sha256(Path(__file__).with_name('target_identity_policy.py').read_bytes()).hexdigest()})
 with workflow.trace_calls(log):
  for parent in parents:
   image=Image.open(inc/'inputs'/(parent['case']+'.png')).convert('RGB');mask=np.asarray(Image.open(Path(parent['folder'])/'mask.png'))>0;p={**parent,'mask':mask.astype('uint8')};prompts,evidence=visible_seam_prompts(image,p);save(out/'parent_plans'/parent['case']/(parent['key']+'.json'),{'prompts':prompts,**evidence});x1,y1,x2,y2=p['box'];box=[max(0,x1-64),max(0,y1-64),min(1920,x2+64),min(1536,y2+64)];crop=image.crop(tuple(box));raw=png(crop)
   for j,prompt in enumerate(prompts):
    points=[dict(q,x=q['x']-box[0],y=q['y']-box[1]) for q in prompt['points']];assert 1<=len(points)<=7
    folder=dest/parent['case']/(parent['key']+'_'+str(j));folder.mkdir(parents=True,exist_ok=True);(folder/'input.png').write_bytes(raw);save(folder/'prompt.json',{'source_box':box,'points':points,'source_full_points':prompt['points'],'seam_interval':prompt['seam_interval'],'mask_prior':None,'source_parent':parent['key']});response=sam(raw,points);save(folder/'response.json',response)
    row={'case':parent['case'],'parent_key':parent['key'],'folder':str(folder),'points':len(points),'technically_valid':bool(response.get('parts') and not response.get('warnings')),'rejection':response.get('rejected_reason'),'warnings':response.get('warnings',[])}
    if row['technically_valid']:
     try:shape,actual,proof=map_complete_response(response,box,1920,1536)
     except ValueError as e:row.update(technically_valid=False,rejection=str(e))
     else:
      key=hashlib.sha256(parent['case'].encode()+actual.tobytes()).hexdigest();hypotheses.append({'key':key,'case':parent['case'],'shape':shape,'score':float(response['score']),'parent_key':parent['key'],'source':'seam_guided_SAM','source_prediction':str(folder/'response.json'),'source_box':box,'wire_proof':proof});row['key']=key
    rows.append(row);save(dest/'progress.json',rows);print(json.dumps({'case':row['case'],'visible_seam':prompt['seam_interval'],'points':len(points),'valid':row['technically_valid']}),flush=True)
    if len(controls)<2:
     blank=png(Image.new('RGB',crop.size,(127,127,127)));control=sam(blank,points);file=folder/'synthetic_blank_response.json';save(file,control);(folder/'synthetic_blank_input.png').write_bytes(blank);iou=None
     if response.get('parts') and control.get('parts'):
      a=_rasterize(response,crop.width,crop.height);b=_rasterize(control,crop.width,crop.height);iou=int((a&b).sum())/max(1,int((a|b).sum()))
     controls.append({'case':parent['case'],'folder':str(folder),'same_points_no_prior':True,'blank_response_rejected':bool(control.get('rejected_reason') or control.get('warnings')),'mask_IoU_with_blank':iou,'functional_control_not_real_scene_result':True})
 save(dest/'all_hypotheses.json',hypotheses);save(dest/'blank_controls.json',{'rows':controls,'not_accuracy':True});save(dest/'completed.json',{'status':'complete','real_scene_SAM_requests':len(rows),'synthetic_control_requests':len(controls),'valid_complete_outputs':len(hypotheses),'no_prior_mask':True,'no_output_budget_trimming':True,'reference_inputs':False});print(json.dumps(read(dest/'completed.json')),flush=True)
if __name__=='__main__':main()
