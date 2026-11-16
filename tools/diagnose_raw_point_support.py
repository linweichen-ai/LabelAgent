"""Reference-informed audit of the low-height raw point premise, not a predictor.

Only counts/bounds are exported. Reference annotations deliberately participate
after all actual predictions are frozen. A restricted-family optimistic bound
is not an achieved result; soft ranking methods can still select other inputs.
"""
import argparse
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.diagnose_proposal_feasibility import reference_matrix,matching_bound
from tools.lidar_candidate_evidence import read_xyz,fit_support_plane,sensor_cameras,projected_sensor,RawPointNovelty,POINT_NOVELTY_RECIPE
from tools.evaluate_projected_support import checked_source,class_membership,project_visible
from tools.compare_boxes3d import match
from tools.tiled_candidate_utils import _mask_record,_iou
from source.agent.edit_agent import _rasterize


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text('utf-8'))


def run(args):
 dataset=Path(args.dataset).resolve();manifest=read(dataset/'import_manifest.json')
 study=Path(args.study).resolve();comparison=read(study/'comparison.json');plan=read(study/'study_plan.json')
 assert comparison['status']=='complete' and comparison['all_predictions_frozen_before_reference_scoring']
 assert digest(study/'study_plan.json')==comparison['prediction_plan_sha256']
 assert digest(REPO/'tools/lidar_candidate_evidence.py')==plan['sensor_helper_sha256']
 def bound(path,receipt):
  record=read(receipt)
  assert digest(path)==record['prediction_json_sha256'] and record['archive_sha256']==manifest['archive_sha256']
  return read(path),record
 baseline,base_receipt=bound(args.baseline_json,args.baseline_receipt)
 pool,pool_receipt=bound(args.pool_json,args.pool_receipt)
 four,four_receipt=bound(study/'four_crop_control_candidates.json',study/'four_crop_control_receipt.json')
 output=Path(args.output).resolve()
 if output.is_relative_to(dataset):raise ValueError('Diagnostic output must stay outside source dataset')
 output.mkdir(parents=True,exist_ok=False)
 save=lambda name,data:(output/name).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
 save('plan.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'uses_reference_annotations':True,
  'diagnostic_only':True,'usable_predictions_exported':False,'new_model_requests':0,
  'point_recipe':POINT_NOVELTY_RECIPE,'fixed_parameters':plan['fixed_parameters'],
  'study_plan_sha256':digest(study/'study_plan.json'),'frozen_comparison_sha256':digest(study/'comparison.json'),
  'pool_sha256':pool_receipt['prediction_json_sha256'],'baseline_sha256':base_receipt['prediction_json_sha256'],
  'diagnostic_source_sha256':digest(__file__),
  'restricted_family':'new proposals having >=5 novel low-height raw points outside original pallet masks',
  'relaxed_constraints':['pairwise new-candidate NMS','union area cap','reference-person point cap']})
 cameras,_=sensor_cameras(dataset,manifest,plan['cameras']);clouds={};rows=[]
 sources={(r['topic'],str(r['frame'])):r for r in manifest['records'] if r['topic'].endswith('/image')}
 for topic,frames in pool.items():
  for frame,values in frames.items():
   for receipt in (base_receipt,pool_receipt,four_receipt):
    assert receipt['image_sha256'][topic][frame]==sources[topic,frame]['sha256']
   original=baseline[topic][frame];selected=four[topic][frame]
   assert values[:len(original)]==original and selected[:len(original)]==original
   proposals=values[len(original):]
   if frame not in clouds:
    points=read_xyz(dataset,manifest,int(frame))[0];clouds[frame]=(points,fit_support_plane(points))
   points,plane=clouds[frame];camera_id=int(topic.split('/')[2].split('_')[-1]);camera=cameras[camera_id]
   width,height=camera['width'],camera['height']
   assert plane==read(study/f'sensor_camera_{camera_id}_{int(frame):04d}.json')['plane']
   sensor=projected_sensor(points,camera,plane)
   context=RawPointNovelty(original,proposals,width,height,sensor)
   stamp=manifest['timestamps_ns'][int(frame)]
   annotation=json.loads(checked_source(dataset,manifest,f'labels/{stamp}.json'))
   pallets=[v for v in values if v['label']=='pallet'];identifiers={v['id']:i for i,v in enumerate(pallets)}
   assert len(identifiers)==len(pallets)
   matrix=reference_matrix(pallets,points,camera,annotation)
   fixed=[identifiers[v['id']] for v in original if v['label']=='pallet']
   old=[identifiers[v['id']] for v in selected if v['label']=='pallet']
   masks=[_mask_record(v,width,height) for v in pallets]
   persons=np.zeros((height,width),np.uint8)
   for v in original:
    if v['label']=='person':persons|=_rasterize(v,width,height)
   available=[];raw_supported=[]
   for i,v in enumerate(pallets):
    if i in fixed or v.get('score',0)<.5:continue
    if any(_iou(masks[i],masks[j])>=.8 for j in fixed):continue
    if np.any(_rasterize(v,width,height)&persons):continue
    available.append(i)
    if np.count_nonzero(context.masks[v['id']]&~context.seed_coverage)>=5:raw_supported.append(i)
   budget=max(1,math.floor(len(fixed)*.5))
   bits,owner,boxes=class_membership(points,annotation['objects'])
   indices,_=project_visible(points,camera,cell_size=8,depth_slack=.15,near=.2)
   owners=owner[indices]
   heights=points[indices]@np.asarray(plane['plane'][:3])+plane['plane'][3]
   lo,hi=POINT_NOVELTY_RECIPE['height_band'];low=(heights>=lo)&(heights<=hi)
   eligible=[np.flatnonzero(owners==i+1) for i,b in enumerate(boxes) if b['bit']==1 and np.count_nonzero(owners==i+1)>=5]
   assert len(eligible)==matrix.shape[0]
   row={'frame':int(frame),'camera':camera_id,'reference_objects':len(eligible),
    'objects_with_at_least_five_low_points':sum(np.count_nonzero(low[v])>=5 for v in eligible),
    'objects_all_visible_points_above_low_band':sum(np.all(heights[v]>hi) for v in eligible),
    'eligible_reference_points':sum(len(v) for v in eligible),
    'eligible_reference_low_points':sum(np.count_nonzero(low[v]) for v in eligible),
    'baseline_matched':len(match(matrix[:,fixed],.5)),'four_crop_matched':len(match(matrix[:,old],.5)),
    'all_pool_matched_unbounded':len(match(matrix,.5)),
    'all_guarded_proposals':len(available),'raw_supported_guarded_proposals':len(raw_supported),
    'all_guarded_upper_bound':matching_bound(matrix,fixed,available,budget),
    'raw_supported_only_upper_bound':matching_bound(matrix,fixed,raw_supported,budget)}
   row={k:int(v) for k,v in row.items()}
   rows.append(row);save('progress.json',{'status':'running','images':len(rows),'rows':rows})
   print(json.dumps({'diagnostic_images':len(rows)}),flush=True)
 keys=[k for k in rows[0] if k not in ('frame','camera')]
 totals={k:int(sum(r[k] for r in rows)) for k in keys}
 assert totals['reference_objects']==comparison['previous_best']['pallet_reference_objects']
 assert totals['baseline_matched']==comparison['baseline']['pallet_objects_matched']
 assert totals['four_crop_matched']==comparison['previous_best']['pallet_objects_matched']
 result={'status':'complete','images':len(rows),'uses_reference_annotations':True,'diagnostic_only':True,
  'usable_predictions_exported':False,'new_model_requests':0,'totals':totals,'rows':rows,
  'scope':'restricted-family optimistic bound, not achieved quality or a limit on all soft ranking methods'}
 save('summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('dataset','baseline-json','baseline-receipt','pool-json','pool-receipt','study','output'):
  parser.add_argument('--'+name,required=True)
 run(parser.parse_args())
