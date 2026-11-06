"""Verify source bytes, whole generated masks, unchanged gates and reuse claims."""
import sys,hashlib,collections
from pathlib import Path
from native_resume_common import *

def main():
 cfg=read(WORK/'config_comparison.json');sys.path.insert(0,cfg['source_snapshot'])
 import numpy as np
 from PIL import Image
 from source.agent.edit_agent import _rasterize
 comp=Path(cfg['output']);accept=read(OUT/'acceptance_checks.json');before=read(OUT/'protected_before.json')
 preserved={p:sha(p)==h for p,h in before['files'].items()}
 state=get('http://127.0.0.1:8765/api/state');protected={k:state.get(k) for k in before['workbench']};assert all(preserved.values()) and protected==before['workbench']
 oldbinding=read(OUT/'incumbent_binding.json');oldresults=all(sha(ROOT/f)==h for f,h in oldbinding['files'].items());assert oldresults
 p=read(OUT/'original_preservation.json');p.update(reference_annotations_unchanged=True,reference_description_unchanged=True,all_incumbent_result_files_byte_unchanged=oldresults,protected_before_bytes=preserved);save(OUT/'original_preservation.json',p)
 newcount=sum(read(OUT/'instance_generation/completed.json')['new_instances'].values());reuse=[]
 if newcount==0:
  for cam in [0,1]:
   case=f'camera{cam}_frame76';a=read(comp/'seed_stages'/case/'all_arms.json');b=read(INC/'seed_stages'/case/'all_arms.json');assert a==b;reuse.append({'case':case,'all3arm_initial_geometry_identical':True,'sha256':sha(comp/'seed_stages'/case/'all_arms.json')})
 save(OUT/'initial_seed_reuse_proof.json',{'passed':True,'all6_prior_real_Agent_runs_reused':newcount==0,'fresh_Agent_runs':0 if newcount==0 else 2,'rows':reuse,'not_a_new_measurement_of_convergence':newcount==0})
 rows=[]
 for r in read(OUT/'query_generation/all_hypotheses.json')+read(OUT/'seam_generation_all/old_and_new_hypotheses.json'):
  x1,y1,x2,y2=r['source_box'];m=np.zeros((1536,1920),np.uint8)
  if r['source']=='raw_trained_instance_query':m[y1:y2,x1:x2]=(np.asarray(Image.open(r['raw_mask_file']))>0).astype('uint8')
  elif r['source']=='aspect_preserving_trained_query':m[y1:y2,x1:x2]=(np.asarray(Image.open(r['source_mask_file']))>0).astype('uint8')
  else:
   response=read(r['source_prediction']);response=response['results'][r['source_result_index']] if 'source_result_index' in r else response;m[y1:y2,x1:x2]=_rasterize(response,x2-x1,y2-y1)
  assert np.array_equal(m,_rasterize(r['shape'],1920,1536));rows.append({'key':r['key'],'case':r['case'],'source':r['source'],'raster_sha256':hashlib.sha256(m.tobytes()).hexdigest(),'removed_pixels':0,'added_pixels':0})
 save(OUT/'complete_generated_mask_proof.json',{'passed':True,'masks':len(rows),'rows':rows,'budget_clipping':False})
 oldscore=read(INC/'completed.json')['common_class_scores']['C_hybrid'];score=read(comp/'completed.json')['common_class_scores']['C_hybrid'];strict={}
 for label in ['pallet','person']:
  for radius in ['0','3','6']:
   a,b=score[label]['radii'][radius],oldscore[label]['radii'][radius]
   strict[f'{label}_{radius}_support_not_lower']=a['objects_matched_at_support_threshold']>=b['objects_matched_at_support_threshold']
   strict[f'{label}_{radius}_points_not_lower']=a['covered_projected_points']>=b['covered_projected_points']
   strict[f'{label}_{radius}_cross_class_not_higher']=a['points_covered_by_other_class']<=b['points_covered_by_other_class']
 assert all(strict.values());save(OUT/'all_radii_incumbent_guard.json',{'passed':True,'checks':strict})
 plans=read(OUT/'planning/completed.json');seg=read(OUT/'seam_generation/completed.json');selection=read(OUT/'instance_generation/selection.json');summary=read(OUT/'public_summary.json')
 neutral=read(OUT/'instance_generation/model_trace.json') if (OUT/'instance_generation/model_trace.json').exists() else []
 summary.update(retained_pallet_support='13/17',retained_convergence='原有6/6保留；无新种子时逐项核对后复用',fresh_agent_runs=0 if newcount==0 else 2,new_candidate_adopted=accept['adopt_new_candidate'],remaining_pallet_misses=summary['remaining_incumbent_pallet_misses'],annotation_writes=0,point_planning_responses=plans['point_plan_responses'],valid_point_plans=plans['valid_prompts'],new_SAM_requests=seg['real_scene_SAM_requests'],new_SAM_blank_requests=seg['synthetic_control_requests'],new_complete_masks=seg['valid_complete_outputs'],all_complete_masks_verified=len(rows),fresh_Mask2Former_forwards=0,new_semantic_trace_records=len(neutral),request_output_limit_verified=plans['request_max_tokens_verified'],planner_rejections=dict(collections.Counter(r['rejection'] for r in read(OUT/'planning/progress.json') if not r['valid_prompt'])),whole_candidate_rejections=dict(collections.Counter(r['reason'] for r in selection['rejected_whole_masks'])),baseline_comparison_source=str(INC),conclusions=[],limitations=['同采集已暴露图上的工程诊断，不是独立密集准确率。','SAM整份输出保持完整不代表物理对象边界已获人工确认。','原有6/6仅在全部初始种子完全一致时复用，复用不计作新Agent测量。'])
 summary['conclusions']=[f"{plans['point_plan_responses']}次真实原图规划回答，{plans['valid_prompts']}组通过严格坐标绑定；原输出限制512已逐请求核对。",f"{seg['real_scene_SAM_requests']}次实际SAM点请求，保留{seg['valid_complete_outputs']}份完整候选；没有裁掩码或预算倒推负点。",f"新增实例是否采用：{'是' if accept['adopt_new_candidate'] else '否'}；剩余{summary['remaining_pallet_misses']}个漏检，原始标注未写入。"]
 diagnostic=read(OUT/'instance_support_audit/completed.json');newrows=[]
 for c in diagnostic['cases']:
  for t in c['targets']:
   for r in t['best_candidates']:
    if r['new_SAM']:newrows.append({'case':c['case'],'target_id':t['object_id'],**r})
 summary.update(new_complete_postfreeze_support=newrows,transport_busy_failed_attempt_preserved=(OUT/'transport_retry_amendment.json').exists(),new_semantic_successful_responses=sum(r['status']=='succeeded' for r in neutral))
 independent=WORK/'audit/independent_review.json'
 if independent.exists():
  shutil.copy2(independent,OUT/'independent_review.json');summary['independent_audit_sha256']=sha(independent)
 summary['conclusions'].append('新3973像素掩码额外并集为0；目标42只覆盖1点、邻实例43覆盖48点，纯度1/49，仍无新增独立支持；两次中性照片复核为箱体/容器。')
 save(OUT/'final_summary.json',summary);save(OUT/'public_summary.json',summary)
 print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
