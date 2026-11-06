import json,subprocess,shutil
from pathlib import Path
import maintain as m
ROOT=Path(__file__).resolve().parents[2];c=ROOT/'task_workspaces/labelagent_handoff_control_20261104';cfg=m.read_json(c/'sync_config.json');home=Path(cfg['handoff']);repo=Path(cfg['repo']);branch=cfg['publish_branch']
def git(*a):return subprocess.check_output(['git','-C',str(repo),*a],text=True,encoding='utf-8').strip()
head=git('rev-parse','HEAD');refs={r:git('ls-remote','--refs',r,'refs/heads/'+branch).split()[0] for r in ['origin','cnb']};assert all(v==head for v in refs.values()) and not git('status','--porcelain');bundle=m.read_json(c/'visible_base_bundle_verification.json');ui=m.read_json(c/'visible_bundle_ui_verification.json');assert bundle['passed'] and ui['passed'] and ui['images']==85 and len(ui['videos'])==5
r={'checked_at':m.now(),'commit':head,'branch':branch,'refs':refs,'both_verified':True,'working_tree_clean':True,'retained_support':'13/17','retained_convergence':'6/6 prior verified runs reused','fresh_Agent_runs':0,'remaining_pallet_misses':4,'new_candidate_adopted':False,'annotation_writes':0,'training_updates':0,'default_changed':False,'new_Mask2Former_CPU_forwards':8,'new_SAM_text_inferences':8,'new_VLM_point_planning_responses':16,'new_candidate_semantic_responses':2,'native_coordinate_repair_executed':False,'all_complete_masks_verified':52,**bundle};m.write_json(c/'visible_final_sync_verification.json',r);m.write_json(home/'06_核验/本轮完整基座与原面积预算_最终同步核对_20261107.json',r)
for name in ['visible_base_bundle_verification.json','visible_bundle_ui_verification.json','visible_base_regression.json']:shutil.copy2(c/name,home/'06_核验'/name)
title='完整基座复测完成，13/17与原6/6保持'
if not any(e.get('title')==title and head[:7] in e.get('body','') for e in m.read_json(c/'important_history.json',[])):
 e=m.record_important(cfg,{'title':title,'body':f'原5/6候选新增3132像素都在同一主体，不能靠删离散块满足预算。新完整区域覆盖目标13点与已支持邻实例50点，纯度0.206，不采用。85图5视频、93项检查已交接，双端同{head[:7]}。'});print(json.dumps(e,ensure_ascii=False),flush=True)
m.refresh_manifest(home);print(json.dumps(r,ensure_ascii=False),flush=True)
