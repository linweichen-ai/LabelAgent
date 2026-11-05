"""Remote readback and one meaningful completion notice, no external messages."""
import json,shutil,csv,io,subprocess
from pathlib import Path
import maintain as m
ROOT=Path(__file__).resolve().parents[2];control=ROOT/'task_workspaces/labelagent_handoff_control_20261104';cfg=m.read_json(control/'sync_config.json');home=Path(cfg['handoff']);repo=Path(cfg['repo']);branch=cfg['publish_branch']
def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True,encoding='utf-8').strip()
head=git('rev-parse','HEAD');refs={name:git('ls-remote','--refs',name,'refs/heads/'+branch).split()[0] for name in ['origin','cnb']};assert all(v==head for v in refs.values()) and not git('status','--porcelain')
bundle=m.read_json(control/'instance_separation_bundle_verification.json');ui=m.read_json(control/'instance_bundle_ui_verification.json');assert bundle['passed'] and ui['passed'] and ui['images']==106 and len(ui['videos'])==5
receipt={'checked_at':m.now(),'commit':head,'branch':branch,'refs':refs,'both_verified':True,'working_tree_clean':True,'retained_foreground':'13/17 PAL,490/579 points,6/6 convergence','remaining_pallet_misses':4,'new_trial_adopted':False,'raw_trained_query_forwards':16,'real_no_prior_SAM_requests':15,'synthetic_blank_SAM_requests_not_scene_data':2,'complete_generated_masks_verified':37,'annotation_writes':0,'default_changed':False,**{k:bundle[k] for k in ['bundle','files','pictures','videos','fully_decoded_frames','native_files_readback_verified','tests']}}
m.write_json(control/'instance_final_sync_verification.json',receipt);m.write_json(home/'06_核验/本轮遮挡粘连实例分离_最终同步核对_20261105.json',receipt)
for name in ['instance_separation_bundle_verification.json','instance_bundle_ui_verification.json','instance_separation_regression.json']:shutil.copy2(control/name,home/'06_核验'/name)
title='遮挡粘连复测完成，保留13/17与6/6';history=m.read_json(control/'important_history.json',[])
if not any(e.get('title')==title and head[:7] in e.get('body','') for e in history):
 event=m.record_important(cfg,{'title':title,'body':f'16次模型、15次真实SAM分离后仍4漏检；未放宽或替换。整摞误判单个的复核矛盾已加拦截，106图5视频、82项检查和完整失败记录已交接。GitHub/CNB同{head[:7]}。'});print(json.dumps(event,ensure_ascii=False),flush=True)
m.refresh_manifest(home);print(json.dumps(receipt,ensure_ascii=False),flush=True)
