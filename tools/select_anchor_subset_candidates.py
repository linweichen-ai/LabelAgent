"""Reference-free anchor-subset selection with no new raw covered points.

Only frozen SAM predictions, raw point clouds and calibration are used. Outputs
are isolated/unconfirmed. No reference annotations, workbench API or model API.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json,math,os
from pathlib import Path
import sys

import cv2
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.raw_coverage_selection import solve_raw_cover,point_person_guard
from tools.lidar_candidate_evidence import ReferenceReadAudit,read_xyz,sensor_cameras
from tools.spatial_candidate_evidence import SpatialPointNovelty,projected_spatial_sensor
from tools.tiled_candidate_utils import augment_candidates,_mask_record,_iou
from tools.run_local_agent_acceptance import save
from source.agent.edit_agent import _rasterize

def read(path):return json.loads(Path(path).read_text('utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
 dataset=Path(args.dataset).resolve();manifest=read(dataset/'import_manifest.json')
 inputs={}
 def bound(name,path,receipt):
  r=read(receipt)
  if digest(path)!=r['prediction_json_sha256'] or r['archive_sha256']!=manifest['archive_sha256']:raise ValueError('Prediction binding changed')
  inputs[name]={'prediction_sha256':digest(path),'receipt_sha256':digest(receipt)}
  return read(path),r
 baseline,br=bound('baseline',args.baseline_json,args.baseline_receipt)
 pool,pr=bound('pool',args.pool_json,args.pool_receipt)
 anchors,ar=bound('anchors',args.anchor_json,args.anchor_receipt)
 for receipt,path in ((pr,args.pool_receipt),(ar,args.anchor_receipt)):
  if digest(Path(path).parent/'study_plan.json')!=receipt['plan_sha256']:raise ValueError('Parent plan binding changed')
 pairs=sorted([(topic,frame) for topic,frames in pool.items() for frame in frames],key=lambda x:(int(x[1]),x[0]))
 if not pairs:raise ValueError('No bound prediction images')
 frames=sorted({int(f) for _,f in pairs});cameras=sorted({int(t.split('/')[2].split('_')[-1]) for t,_ in pairs})
 output=Path(args.output).resolve()
 if output.is_relative_to(dataset):raise ValueError('Output must remain outside source dataset')
 output.mkdir(parents=True,exist_ok=False)
 methods={'four_crop_control':{'type':'legacy'},
  'anchor_subset_priority':{'type':'raw_cover','objective':'fixed_spatial_mass_priority_sum','maximum_new_raw_points':0}}
 before=digest(args.state_file) if args.state_file else None
 helper_names=('raw_coverage_selection.py','spatial_candidate_evidence.py','lidar_candidate_evidence.py','tiled_candidate_utils.py')
 helper_hashes={name:digest(REPO/'tools'/name) for name in helper_names}
 plan={'raw_union_contract':'Exact old raw-visible point union, no reference class information',
  'created_utc':datetime.now(timezone.utc).isoformat(),'version':1,'methods':methods,'frames':frames,'cameras':cameras,
  'planned_images':len(pairs),'new_model_requests':0,'input_bindings':inputs,'code_sha256':digest(__file__),'helper_sha256':helper_hashes,
  'fixed_parameters':{'minimum_score':.5,'addition_ratio':.5,'nms_iou':.8},'solver_seconds_per_image':args.solver_seconds,
  'raw_projection':{'cell_size':8,'depth_slack':.15,'near':.2,'mask_dilation_pixels':3,'kernel':'ellipse'},
  'coverage_certificate':'Every raw visible point covered by previous four-crop pallet masks remains covered; baseline always retained',
  'person_prediction_guard':'New additions cover no raw points outside old four-crop anchor, regardless of unknown semantic class',
  'component_objective':'0.5*sum(base_i*x_i)+0.5*sum(max_i(base_i*coherence_i/(5+component_points_i))*covered_raw_point)',
  'component_novelty':'Only foreground points outside original foreground component union; no reference class membership',
  'prediction_uses_3D_reference':False,'no_reference_oracle_inputs':True,
  'comparison_requires':{'strict_instance_gain_over_previous_best':True,'point_coverage_not_below_previous_best':True,
   'person_off_class_not_above_previous_best':True,'original_minimum_absolute_gain':.02,'original_maximum_candidate_count_ratio':1.5,
   'original_maximum_union_area_ratio':1.25,'original_person_off_class_increase_limit':.005,'all_original_candidates_preserved':True},
  'automatic_broader_run':False,'scope':'exploratory same-capture predictions, unconfirmed, not independent pixel accuracy'}
 save(output/'study_plan.json',plan);plan_sha=digest(output/'study_plan.json')
 progress={'status':'running','pid':os.getpid(),'completed_images':0,'new_model_requests':0,'postprocessing':[],
  'bound_pool_proposals':0,'workbench_modified':False}
 save(output/'progress.json',progress)
 maps={name:{} for name in methods};image_bindings={};audit=ReferenceReadAudit(dataset);audit.start()
 try:
  camera_info,_=sensor_cameras(dataset,manifest,cameras);clouds={};kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
  records={(r['topic'],str(r['frame'])):r for r in manifest['records'] if r['topic'].endswith('/image')}
  for topic,frame in pairs:
   row=records[topic,frame]
   for receipt in (br,pr,ar):
    if receipt['image_sha256'][topic][frame]!=row['sha256']:raise ValueError('Image binding changed')
   seed=baseline[topic][frame];all_values=pool[topic][frame];old=anchors[topic][frame]
   if all_values[:len(seed)]!=seed or old[:len(seed)]!=seed:raise ValueError('Original candidates changed')
   proposals=all_values[len(seed):]
   if any(s['label']!='pallet' or not str(s['id']).startswith(('tile-','dense-tile-')) for s in proposals):raise ValueError('Expected bound real-grid pallet proposals')
   if len({s['id'] for s in all_values})!=len(all_values):raise ValueError('Duplicate proposal IDs')
   values_by_id={s['id']:s for s in all_values}
   if any(values_by_id[s['id']]!=s for s in old):raise ValueError('Anchor geometry not in frozen pool')
   if frame not in clouds:clouds[frame]=read_xyz(dataset,manifest,int(frame))[0]
   camera=camera_info[int(topic.split('/')[2].split('_')[-1])];width,height=camera['width'],camera['height']
   sensor=projected_spatial_sensor(clouds[frame],camera);pixels=sensor['pixels']
   context=SpatialPointNovelty(seed,proposals,width,height,sensor)
   masks={};mask_records={};person_mask=np.zeros((height,width),np.uint8)
   for shape in all_values:
    mask=_rasterize(shape,width,height)
    if shape['label']=='person':person_mask|=mask
    elif shape['label']=='pallet':
     d=cv2.dilate(mask,kernel);masks[shape['id']]=set(np.flatnonzero(d[pixels[:,1],pixels[:,0]]).tolist())
     mask_records[shape['id']]=_mask_record(shape,width,height)
   original_pallets=[s for s in seed if s['label']=='pallet']
   base=set().union(*(masks[s['id']] for s in original_pallets))
   anchor=set().union(*(masks[s['id']] for s in old if s['label']=='pallet'));required=anchor-base
   pd=cv2.dilate(person_mask,kernel);person_points=set(np.flatnonzero(pd[pixels[:,1],pixels[:,0]]).tolist())
   eligible=[]
   for shape in proposals:
    if shape.get('score',0)<.5:continue
    if any(_iou(mask_records[shape['id']],mask_records[s['id']])>=.8 for s in original_pallets):continue
    if masks[shape['id']]<=anchor:eligible.append(shape)
   allowed_ids={s['id'] for s in eligible}
   if not all(s['id'] in allowed_ids for s in old[len(seed):]):raise ValueError('Old anchor witness unexpectedly excluded')
   budget=max(1,math.floor(len(original_pallets)*.5))
   conflicts=[(i,j) for i,a in enumerate(eligible) for j,b in enumerate(eligible) if i<j and _iou(mask_records[a['id']],mask_records[b['id']])>=.8]
   four,_=augment_candidates(seed,[s for s in proposals if str(s['id']).startswith('tile-')],width,height,**plan['fixed_parameters'])
   if four!=old:raise ValueError('Four-crop control changed')
   outputs={'four_crop_control':four}
   static=[context.priority(s,tuple(seed),'component_mass') for s in eligible]
   features=[set(np.flatnonzero(context.components[s['id']]&~context.seeds['component']).tolist()) for s in eligible]
   weights={}
   for shape,points in zip(eligible,features):
    facts=context.facts[shape['id']];value=.5*context.base[shape['id']]*facts['coherence']/(5+facts['component_points'])
    for point in points:weights[point]=max(weights.get(point,0.),value)
   solver_summaries={}
   for method in ('anchor_subset_priority',):
    options={}
    scores=static
    solved=solve_raw_cover([masks[s['id']] for s in eligible],required,scores,conflicts,budget,time_limit=args.solver_seconds,**options)
    solver_summaries[method]=solved['summary']
    if not solved['summary']['certificate_satisfied']:raise ValueError('No verified raw coverage certificate for '+method)
    selected=[eligible[i] for i in solved['selected_indices']]
    covered=base|set().union(*(masks[s['id']] for s in selected))
    if covered!=anchor:raise ValueError('Actual raw-point union differs from old anchor')
    if any(not masks[s['id']]<=anchor for s in selected):raise ValueError('Added raw points outside old anchor')
    outputs[method]=deepcopy(seed+selected)
   for method,shapes in outputs.items():
    if shapes[:len(seed)]!=seed:raise ValueError('Original prefix changed')
    maps[method].setdefault(topic,{})[frame]=shapes
    info={'baseline_candidates_preserved':True,'added_pallets':len(shapes)-len(seed),'original_pallets':len(original_pallets)}
    if method in solver_summaries:info.update(solver=solver_summaries[method],raw_certificate_verified=True)
    progress['postprocessing'].append({'frame':int(frame),'topic':topic,'method':method,**info})
   image_bindings.setdefault(topic,{})[frame]=row['sha256'];progress['completed_images']+=1;progress['bound_pool_proposals']+=len(proposals)
   save(output/'progress.json',progress);print(json.dumps({'prediction_images':progress['completed_images']}),flush=True)
  for method,predictions in maps.items():
   path=output/(method+'_candidates.json');save(path,predictions)
   save(output/(method+'_receipt.json'),{'created_utc':datetime.now(timezone.utc).isoformat(),'archive_sha256':manifest['archive_sha256'],
    'prediction_json_sha256':digest(path),'image_sha256':image_bindings,'plan_sha256':plan_sha,'method':method,
    'new_model_requests':0,'candidates_confirmed_as_truth':False,'source':'reference_free_frozen_SAM_proposals_and_raw_point_coverage'})
  progress.update(status='complete',plan_sha256=plan_sha,plan_unchanged=digest(output/'study_plan.json')==plan_sha,
   finished_utc=datetime.now(timezone.utc).isoformat())
 except Exception as error:
  progress.update(status='failed',error_type=type(error).__name__);(output/'local_error.txt').write_text(str(error),encoding='utf-8')
 finally:
  audit.stop();progress.update(reference_files_opened=len(audit.denied),raw_source_files_read=sorted(audit.sources),
   state_sha256_unchanged=digest(args.state_file)==before if args.state_file else None,
   baseline_cache_unchanged=digest(args.baseline_json)==inputs['baseline']['prediction_sha256'],
   frozen_pool_and_anchors_unchanged=digest(args.pool_json)==inputs['pool']['prediction_sha256'] and digest(args.anchor_json)==inputs['anchors']['prediction_sha256'],
   code_unchanged=digest(__file__)==plan['code_sha256'] and all(digest(REPO/'tools'/name)==h for name,h in helper_hashes.items()))
  if audit.denied:progress['status']='failed'
  save(output/'progress.json',progress)
 print(json.dumps({k:v for k,v in progress.items() if k not in ('postprocessing','raw_source_files_read')}),flush=True)
 return 0 if progress['status']=='complete' else 1


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('dataset','baseline-json','baseline-receipt','pool-json','pool-receipt','anchor-json','anchor-receipt','output'):parser.add_argument('--'+name,required=True)
 parser.add_argument('--state-file');parser.add_argument('--solver-seconds',type=float,default=10.)
 raise SystemExit(run(parser.parse_args()))
