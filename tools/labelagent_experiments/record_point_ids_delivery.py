"""Verify both remote refs, then bind the completed local handoff checkpoint."""
import subprocess,shutil,json
from pathlib import Path
from point_ids_common import *
import maintain as m
CONTROL=ROOT/'task_workspaces/labelagent_handoff_control_20261104';cfg=read(CONTROL/'sync_config.json');repo=Path(cfg['repo']);home=Path(cfg['handoff']);branch=cfg['publish_branch']
def git(*a):return subprocess.check_output(['git','-C',str(repo),*a],text=True,encoding='utf-8').strip()
head=git('rev-parse','HEAD');refs={remote:git('ls-remote','--refs',remote,'refs/heads/'+branch).split()[0] for remote in ['origin','cnb']};assert all(v==head for v in refs.values()) and not git('status','--porcelain')
bundle=read(CONTROL/'point_ids_bundle_verification.json');ui=read(CONTROL/'point_ids_bundle_ui_verification.json');s=read(OUT/'final_summary.json');media=read(OUT/'media_verification.json')
assert bundle['passed'] and ui['passed'] and ui['images']==media['pictures'] and len(ui['videos'])==len(media['videos'])
r={**bundle,'bundle_verification_checked_at':bundle['checked_at'],'checked_at':m.now(),'commit':head,'refs':refs,'branch':branch,'both_verified':True,'working_tree_clean':True,'retained_pallet_support':'13/17','retained_convergence':'6/6 original verified runs preserved','fresh_Agent_runs':s['fresh_agent_runs'],'remaining_pallet_misses':s['remaining_pallet_misses'],'new_candidate_adopted':s['new_candidate_adopted'],'actual_planning_responses':s['planner_total_actual_responses'],'new_SAM_requests':s['new_SAM_requests'],'temporal_RGB_pairs':16,'temporal_identity_confirmations':0,'annotation_writes':0,'training_updates':0,'default_changed':False}
r.pop('root_index_or_remote_modified',None);save(CONTROL/'point_ids_final_sync_verification.json',r);save(home/'06_核验/本轮编号点与邻帧归属_最终同步核对_20261107.json',r)
for name in ['point_ids_bundle_verification.json','point_ids_bundle_ui_verification.json','point_ids_regression.json']:shutil.copy2(CONTROL/name,home/'06_核验'/name)
title='叠层归属线索已核对，13/17与原6/6保持'
if not any(e.get('title')==title for e in m.read_json(CONTROL/'important_history.json',[])):
 m.record_important(cfg,{'title':title,'body':f"编号点20次实际回答仍无可用新实例；空白物体幻觉尚存。原42/43薄框与叠层相容，不能改货箱类别来过关。{media['pictures']}图{len(media['videos'])}视频已封存，双端{head[:7]}。"})
m.refresh_manifest(home);print(json.dumps(r,ensure_ascii=False),flush=True)
