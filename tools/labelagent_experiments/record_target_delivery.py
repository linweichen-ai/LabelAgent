"""Verified dual refs and a meaningful local notification, never a team message."""
import json,shutil,subprocess
from pathlib import Path
import maintain as m
ROOT=Path(__file__).resolve().parents[2];c=ROOT/'task_workspaces/labelagent_handoff_control_20261104';cfg=m.read_json(c/'sync_config.json');home=Path(cfg['handoff']);repo=Path(cfg['repo']);branch=cfg['publish_branch']
def git(*a):return subprocess.check_output(['git','-C',str(repo),*a],text=True,encoding='utf-8').strip()
head=git('rev-parse','HEAD');refs={r:git('ls-remote','--refs',r,'refs/heads/'+branch).split()[0] for r in ['origin','cnb']};assert all(v==head for v in refs.values()) and not git('status','--porcelain');bundle=m.read_json(c/'target_identity_bundle_verification.json');ui=m.read_json(c/'target_bundle_ui_verification.json');assert bundle['passed'] and ui['passed'] and ui['images']==96 and len(ui['videos'])==5
r={'checked_at':m.now(),'commit':head,'branch':branch,'refs':refs,'both_verified':True,'working_tree_clean':True,'retained_support':'13/17','retained_convergence':'6/6','remaining_pallet_misses':4,'new_candidate_adopted':False,'annotation_writes':0,'training_updates':0,'default_changed':False,'new_real_SAM_requests':11,'synthetic_SAM_requests':2,'new_identity_diagnostic_VLM_requests':3,'new_candidate_neutral_VLM_requests':16,'new_Mask2Former_forwards':0,'all_complete_masks_verified':43,**bundle};m.write_json(c/'target_final_sync_verification.json',r);m.write_json(home/'06_核验/本轮候选身份与纯度_最终同步核对_20261107.json',r)
for name in ['target_identity_bundle_verification.json','target_bundle_ui_verification.json','target_identity_regression.json']:shutil.copy2(c/name,home/'06_核验'/name)
title='候选身份和纯度复测完成，原13/17与6/6保留'
if not any(e.get('title')==title and head[:7] in e.get('body','') for e in m.read_json(c/'important_history.json',[])):
 e=m.record_important(cfg,{'title':title,'body':f'窄候选重居中仍无法确认，精确掩码诊断为边缘碎片；新候选5/6支持与纯度过关但面积超限，不裁剪、不采用。96图5视频、86项检查及全部原输出已交接。双端同{head[:7]}。'});print(json.dumps(e,ensure_ascii=False),flush=True)
m.refresh_manifest(home);print(json.dumps(r,ensure_ascii=False),flush=True)
