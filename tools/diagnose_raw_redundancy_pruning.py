"""Freeze two reference-free pruning diagnostics; never change the full recipe."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
from itertools import combinations
import hashlib,json,sys
import cv2
import numpy as np

import argparse
parser=argparse.ArgumentParser(description=__doc__)
for name in ('dataset','study-root','output'):parser.add_argument('--'+name,required=True)
args=parser.parse_args()
repo=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(repo),str(repo/'tools')]
from lidar_candidate_evidence import ReferenceReadAudit,read_xyz,sensor_cameras
from spatial_candidate_evidence import projected_spatial_sensor
from source.agent.edit_agent import _rasterize
from run_local_agent_acceptance import save

root=Path(args.study_root).resolve();dataset=Path(args.dataset).resolve()
source=root/'raw_certificate_tuning_v1';out=Path(args.output).resolve()
assert not out.is_relative_to(dataset),'Keep diagnostic exports outside the original dataset'
assert not out.exists()
read=lambda p:json.loads(p.read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent_plan=read(source/'study_plan.json');manifest=read(dataset/'import_manifest.json')
baseline_path=dataset/'sam3_baseline/candidates_by_topic.json';baseline=read(baseline_path)
primary_path=source/'raw_certificate_priority_candidates.json';primary=read(primary_path)
parent_receipt=read(source/'raw_certificate_priority_receipt.json')
assert parent_receipt['prediction_json_sha256']==sha(primary_path)
full=read(root/'raw_certificate_full_validation_plan_v1.json')
assert set(parent_plan['frames'])=={15,45,76,105,135} and set(parent_plan['frames']).isdisjoint(full['full_validation_frames'])
methods={'prune_no_new_baseline_raw_points':'Remove added candidates only when a nonempty masked raw-point set is fully covered by original SAM pallets',
    'minimum_raw_union_cover':'Keep the minimum number of existing selected additions that cover the full primary raw-point union; tie by maximum original SAM score, then original order'}
paths=[baseline_path,primary_path,source/'raw_certificate_priority_receipt.json',source/'study_plan.json',dataset/'import_manifest.json']
bindings={p:sha(p) for p in paths}
out.mkdir()
plan={'created_utc':datetime.now(timezone.utc).isoformat(),'frames':parent_plan['frames'],'cameras':parent_plan['cameras'],
    'methods':methods,'planned_images':20,'code_sha256':sha(Path(__file__)),
    'input_sha256':{str(p):value for p,value in bindings.items()},'primary_recipe_unchanged':True,
    'raw_projection':{'cell_size':8,'depth_slack':.15,'near':.2,'mask_dilation_pixels':3,'kernel':'ellipse'},
    'success_requires':{'all_original_eight_gates':True,'matched_instances_not_below_existing_primary':78,
        'covered_points_not_below_existing_primary':5183,'person_cross_points_not_above_existing_primary':55,
        'pallet_candidates_strictly_below_existing_primary':197,'raw_point_union_equal_every_image':True,
        'all_original_candidates_preserved':True},
    'prediction_reads_reference':False,'reference_informed_filtering':False,'automatic_expansion_or_primary_replacement':False,
    'scope':'Tuning-only pruning diagnostics, not an adopted method or full validation result'}
save(out/'study_plan.json',plan);plan_sha=sha(out/'study_plan.json')
maps={m:{} for m in methods};image_bindings={};rows=[];audit=ReferenceReadAudit(dataset);audit.start()
try:
    cameras,_=sensor_cameras(dataset,manifest,plan['cameras'])
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7));clouds={}
    records={(r['topic'],str(r['frame'])):r for r in manifest['records'] if r['topic'].endswith('/image')}
    for frame in plan['frames']:
        clouds[frame]=read_xyz(dataset,manifest,frame)[0]
        for camera_index in plan['cameras']:
            topic=f'/capture/camera_{camera_index}/image';key=str(frame)
            seed=baseline[topic][key];current=primary[topic][key];assert current[:len(seed)]==seed
            extras=current[len(seed):];assert len(extras)<=20 and all(s['label']=='pallet' for s in extras)
            camera=cameras[camera_index];pixels=projected_spatial_sensor(clouds[frame],camera)['pixels']
            masks={s['id']:set(np.flatnonzero(cv2.dilate(_rasterize(s,camera['width'],camera['height']),kernel)[pixels[:,1],pixels[:,0]]).tolist())
                   for s in current if s['label']=='pallet'}
            base=set().union(*(masks[s['id']] for s in seed if s['label']=='pallet'))
            target=set().union(*(masks[s['id']] for s in current if s['label']=='pallet'))
            first=[s for s in extras if not(masks[s['id']] and masks[s['id']]<=base)]
            required=target-base;best=None;enumerated=0
            for size in range(len(extras)+1):
                feasible=[]
                for indices in combinations(range(len(extras)),size):
                    enumerated+=1
                    covered=set().union(*(masks[extras[i]['id']] for i in indices))
                    if required<=covered:
                        feasible.append((-sum(float(extras[i]['score']) for i in indices),indices))
                if feasible:
                    best=min(feasible)[1];break
            assert best is not None
            choices={'prune_no_new_baseline_raw_points':first,'minimum_raw_union_cover':[extras[i] for i in best]}
            for method,selected in choices.items():
                shapes=deepcopy(seed+selected)
                actual=base|set().union(*(masks[s['id']] for s in selected))
                assert actual==target and shapes[:len(seed)]==seed
                maps[method].setdefault(topic,{})[key]=shapes
                rows.append({'frame':frame,'camera':camera_index,'method':method,'original_additions':len(extras),
                    'remaining_additions':len(selected),'raw_point_union_equal':True,'baseline_preserved':True,
                    'minimum_cover_subsets_enumerated':enumerated})
            image_bindings.setdefault(topic,{})[key]=records[topic,key]['sha256']
    for method,values in maps.items():
        path=out/(method+'_candidates.json');save(path,values)
        save(out/(method+'_receipt.json'),{'archive_sha256':manifest['archive_sha256'],'prediction_json_sha256':sha(path),
            'image_sha256':image_bindings,'plan_sha256':plan_sha,'method':method,'source':'tuning_reference_free_subset_of_existing_primary',
            'new_model_requests':0,'candidates_confirmed_as_truth':False})
finally:audit.stop()
assert not audit.denied and all(sha(p)==value for p,value in bindings.items()) and sha(out/'study_plan.json')==plan_sha
result={'status':'complete','images':20,'methods':list(methods),'reference_files_opened':len(audit.denied),
    'new_model_requests':0,'plan_sha256':plan_sha,'input_files_unchanged':True,'all_raw_union_certificates_equal':True,
    'all_original_prefixes_preserved':True,'full_validation_recipe_unchanged':True,'postprocessing':rows}
save(out/'progress.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='postprocessing'}),flush=True)
