"""Diagnose four frozen misses against original reference/visibility protocol."""
import datetime,hashlib,json,sys
from pathlib import Path
import numpy as np
from point_separability import analyze
from projection_audit_core import ROOT,read,frame_data,cameras
OUT=ROOT/'outputs/labelagent_point_separability_20261107'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 if OUT.exists():raise FileExistsError('Immutable audit; no overwrite')
 OUT.mkdir();cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');sys.path[:0]=[cfg['source_snapshot'],str(Path(cfg['source_snapshot'])/'tools'),cfg['dependency_dir']]
 import evaluate_projected_support as ev
 scorer=Path(cfg['source_snapshot'])/'tools/evaluate_projected_support.py';assert sha(scorer)=='93c5152169d04d43b25dbf4a4139b12676daf78750a7f059837121656bb644c2'
 bindings=read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'];assert all(sha(ROOT/p)==h for p,h in bindings.items())
 plan={'at':datetime.datetime.now().astimezone().isoformat(),'purpose':'reference-only arbitrary-pixel feasibility relaxation, NOT image-based segmentation','radii':[0,3,6],'purity_integer_constraint':'33*target_hits >= 67*other_PAL_hits','support_threshold':.5,'original_eligible_points':5,'visibility_cell':8,'depth_slack_m':.15,'cases':[[0,42],[0,48],[1,47],[1,49]],'scorer_sha256':sha(scorer),'policy_sha256':sha(Path(__file__).with_name('point_separability.py')),'original_six_hashes':bindings,'generator_uses_result':False,'model_calls':0,'new_masks_or_annotations':0}
 save(OUT/'frozen_plan.json',plan)
 dataset=Path(cfg['dataset']);manifest=read(dataset/'import_manifest.json');data=frame_data(dataset,manifest,76,ev);cams=cameras(dataset,manifest,ev);rows=[];pointsets=[]
 for cam,oid in plan['cases']:
  source,pix=ev.project_visible(data['xyz'],cams[cam]);known=(data['bits'][source]==1)&(data['owner'][source]>0);px=pix[known];owners=data['owner'][source][known];owner=next(i+1 for i,b in enumerate(data['boxes']) if b['obj_id']==oid)
  points={'camera':cam,'object_id':oid,'pixels':px.tolist(),'owners':owners.tolist(),'target_owner':owner,'owner_to_object':{str(i+1):b['obj_id'] for i,b in enumerate(data['boxes'])}}
  save(OUT/f'camera{cam}_obj{oid}_frozen_reference_points.json',points);pointsets.append(points)
  row={'camera':cam,'object_id':oid,'radii':[analyze(px,owners,owner,r) for r in plan['radii']]};rows.append(row);save(OUT/'progress.json',rows);print(json.dumps(row),flush=True)
 assert all(sha(ROOT/p)==h for p,h in bindings.items())
 result={'status':'complete_reference_only_separability_audit','rows':rows,'all_original_six_exact':True,'model_calls':0,'new_masks':0,'annotation_writes':0,'references_or_thresholds_changed':False,'retained_support':'13/17','retained_original_convergence':'6/6','remaining_misses':4,'adopted':False,'interpretation':'Infeasible proves only this permissive pixel relaxation cannot pass the supplied point gate. Feasible does NOT prove a real visible physical instance or usable mask; isolated point islands are allowed only within this diagnostic.'}
 save(OUT/'completed.json',result)
if __name__=='__main__':main()
