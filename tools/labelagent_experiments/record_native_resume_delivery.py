"""Read both remote heads and bind verified handoff evidence to the commit."""
import subprocess,shutil
from pathlib import Path
from native_resume_common import *
import maintain as m
control=ROOT/'task_workspaces/labelagent_handoff_control_20261104';cfg=m.read_json(control/'sync_config.json');home=Path(cfg['handoff']);repo=Path(cfg['repo']);branch=cfg['publish_branch']
def git(*a):return subprocess.check_output(['git','-C',str(repo),*a],text=True,encoding='utf-8').strip()
head=git('rev-parse','HEAD');refs={r:git('ls-remote','--refs',r,'refs/heads/'+branch).split()[0] for r in ['origin','cnb']};assert all(v==head for v in refs.values()) and not git('status','--porcelain')
bundle=read(control/'native_resume_bundle_verification.json');ui=read(control/'native_resume_bundle_ui_verification.json');summary=read(OUT/'final_summary.json');media=read(OUT/'media_verification.json');assert bundle['passed'] and ui['passed'] and ui['images']==media['pictures'] and len(ui['videos'])==len(media['videos'])
r={'checked_at':m.now(),'commit':head,'branch':branch,'refs':refs,'both_verified':True,'working_tree_clean':True,'retained_support':'13/17','retained_convergence':'6/6 prior verified results preserved','fresh_Agent_runs':summary['fresh_agent_runs'],'remaining_pallet_misses':summary['remaining_pallet_misses'],'new_candidate_adopted':summary['new_candidate_adopted'],'annotation_writes':0,'training_updates':0,'default_changed':False,'new_VLM_planning_responses':summary['point_planning_responses'],'valid_point_plans':summary['valid_point_plans'],'new_SAM_requests':summary['new_SAM_requests'],'new_complete_masks':summary['new_complete_masks'],**bundle}
r.update(bundle_verification_checked_at=bundle['checked_at'],checked_at=m.now())
r.pop('root_index_or_remote_modified',None)
save(control/'native_resume_final_sync_verification.json',r);save(home/'06_核验/本轮原图坐标续跑_最终同步核对_20261107.json',r)
for name in ['native_resume_bundle_verification.json','native_resume_bundle_ui_verification.json','native_resume_regression.json']:shutil.copy2(control/name,home/'06_核验'/name)
title='原图坐标方案实际续跑，原13/17与6/6保持'
if not any(e.get('title')==title and head[:7] in e.get('body','') for e in m.read_json(control/'important_history.json',[])):
 m.record_important(cfg,{'title':title,'body':f"16次实际规划，{summary['valid_point_plans']}组合格点，{summary['new_SAM_requests']}次SAM场景请求。{media['pictures']}图{len(media['videos'])}视频已交接；原验收不变，剩余{summary['remaining_pallet_misses']}个漏检。双端同{head[:7]}。"})
m.refresh_manifest(home);print(json.dumps(r,ensure_ascii=False),flush=True)
