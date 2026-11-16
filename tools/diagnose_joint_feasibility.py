"""Reference-informed joint feasibility audit; not a label-free predictor.

Reads reference labels deliberately after actual predictions are frozen. Exports
counts/status/bounds only, never oracle-selected candidate IDs or annotations.
The solver relaxes pixel-union area; its internal witness area is checked later.
"""
import argparse
from datetime import datetime,timezone
import hashlib,json,math,os
from pathlib import Path
import sys

import cv2
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.joint_selection_bounds import solve_joint_selection,selected_statistics
from tools.diagnose_proposal_feasibility import reference_matrix
from tools.lidar_candidate_evidence import read_xyz,sensor_cameras
from tools.evaluate_projected_support import checked_source,class_membership,project_visible
from tools.tiled_candidate_utils import _mask_record,_iou
from source.agent.edit_agent import _rasterize

def read(path):return json.loads(Path(path).read_text('utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
 dataset=Path(args.dataset).resolve();manifest=read(dataset/'import_manifest.json')
 study=Path(args.study).resolve();comparison=read(study/'comparison.json')
 assert comparison['status']=='complete' and comparison['all_predictions_frozen_before_reference_scoring']
 assert digest(REPO/'tools/evaluate_projected_support.py')==comparison['metric_sha256']
 def bound(path,receipt):
  r=read(receipt);assert digest(path)==r['prediction_json_sha256'] and r['archive_sha256']==manifest['archive_sha256']
  return read(path),r
 baseline,baseline_receipt=bound(args.baseline_json,args.baseline_receipt)
 pool,pool_receipt=bound(args.pool_json,args.pool_receipt)
 four,four_receipt=bound(study/'four_crop_control_candidates.json',study/'four_crop_control_receipt.json')
 spatial,spatial_receipt=bound(study/'spatial_component_mass_guard_candidates.json',study/'spatial_component_mass_guard_receipt.json')
 output=Path(args.output).resolve()
 if output.is_relative_to(dataset):raise ValueError('Keep diagnostic output outside source dataset')
 output.mkdir(parents=True,exist_ok=False)
 save=lambda name,value:(output/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
 before=digest(REPO/'work/state.json')
 minimum_points=comparison['previous_best']['pallet_points_covered']
 maximum_person=comparison['previous_best']['person_points_covered_by_pallet']
 minimum_matches=comparison['previous_best']['pallet_objects_matched']+1
 limits={'minimum_score':.5,'addition_ratio':.5,'nms_iou':.8,'maximum_predicted_person_overlap':0.,
  'minimum_points':minimum_points,'maximum_person_points':maximum_person,'minimum_matches':minimum_matches,
  'maximum_count_ratio':1.5,'maximum_pixel_union_ratio':1.25,'minimum_baseline_absolute_gain':.02}
 plan={'created_utc':datetime.now(timezone.utc).isoformat(),'uses_reference_annotations':True,
  'diagnostic_only':True,'usable_predictions_exported':False,'new_model_requests':0,'constraints':limits,
  'relaxed_solver_constraint':'global raw pixel-union area only; internal incumbent area checked separately',
  'solver_seconds':args.solver_seconds,'baseline_sha256':baseline_receipt['prediction_json_sha256'],
  'pool_sha256':pool_receipt['prediction_json_sha256'],'comparison_sha256':digest(study/'comparison.json'),
  'metric_sha256':comparison['metric_sha256'],'diagnostic_sha256':digest(__file__),
  'solver_helper_sha256':digest(REPO/'tools/joint_selection_bounds.py')}
 save('plan.json',plan)
 progress={'status':'running','pid':os.getpid(),'phase':'build_reference_constraints','images':0,'new_model_requests':0}
 save('progress.json',progress)
 try:
  cameras,_=sensor_cameras(dataset,manifest,range(4));clouds={};annotations={};images=[];private_shapes=[]
  controls={name:[] for name in ('baseline','previous_four','spatial_mass','all_pool')}
  control_totals={name:{'matched':0,'points':0,'person_points':0,'candidates':0,'union_pixels':0} for name in controls}
  records={(r['topic'],str(r['frame'])):r for r in manifest['records'] if r['topic'].endswith('/image')}
  for topic,frames in pool.items():
   for frame,values in frames.items():
    for receipt in (baseline_receipt,pool_receipt,four_receipt,spatial_receipt):
     assert receipt['image_sha256'][topic][frame]==records[topic,frame]['sha256']
    original=baseline[topic][frame];assert values[:len(original)]==original
    for frozen in (four[topic][frame],spatial[topic][frame]):assert frozen[:len(original)]==original
    if frame not in clouds:clouds[frame]=read_xyz(dataset,manifest,int(frame))[0]
    points=clouds[frame];camera=cameras[int(topic.split('/')[2].split('_')[-1])];width,height=camera['width'],camera['height']
    if frame not in annotations:
     stamp=manifest['timestamps_ns'][int(frame)];annotations[frame]=json.loads(checked_source(dataset,manifest,f'labels/{stamp}.json'))
    annotation=annotations[frame];pallets=[s for s in values if s['label']=='pallet']
    identifiers={s['id']:i for i,s in enumerate(pallets)};assert len(identifiers)==len(pallets)
    selections={name:[identifiers[s['id']] for s in data if s['label']=='pallet'] for name,data in
     (('baseline',original),('previous_four',four[topic][frame]),('spatial_mass',spatial[topic][frame]),('all_pool',values))}
    fixed=selections['baseline'];fixed_set=set(fixed)
    matrix=reference_matrix(pallets,points,camera,annotation)>=.5
    bits,_,_=class_membership(points,annotation['objects']);source,pixels=project_visible(points,camera,cell_size=8,depth_slack=.15,near=.2)
    point_rows=np.flatnonzero(bits[source]==1);person_rows=np.flatnonzero(bits[source]==2)
    persons=np.zeros((height,width),np.uint8)
    for shape in original:
     if shape['label']=='person':persons|=_rasterize(shape,width,height)
    records_masks=[_mask_record(s,width,height) for s in pallets]
    point_sets=[];person_sets=[];allowed=[]
    unions={name:np.zeros((height,width),np.uint8) for name in controls}
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    for column,shape in enumerate(pallets):
     mask=_rasterize(shape,width,height);dilated=cv2.dilate(mask,kernel)
     for rows,target in ((point_rows,point_sets),(person_rows,person_sets)):
      positions=pixels[rows];hits=dilated[positions[:,1],positions[:,0]]!=0
      target.append(set(map(int,source[rows[hits]])))
     for name,columns in selections.items():
      if column in columns:unions[name]|=mask
     if column in fixed_set or shape.get('score',0)<.5:continue
     if any(_iou(records_masks[column],records_masks[j])>=.8 for j in fixed):continue
     if np.any(mask&persons):continue
     allowed.append(column)
    raw={'matrix':matrix,'point_sets':point_sets,'person_sets':person_sets}
    for name,columns in selections.items():
     counts=selected_statistics([raw],[columns]);counts['union_pixels']=int(np.count_nonzero(unions[name]))
     for key,value in counts.items():control_totals[name][key]+=value
    retained=fixed+allowed;local={column:i for i,column in enumerate(retained)}
    conflicts=[(local[a],local[b]) for index,a in enumerate(allowed) for b in allowed[index+1:]
     if _iou(records_masks[a],records_masks[b])>=.8]
    images.append({'matrix':matrix[:,retained],'point_sets':[point_sets[i] for i in retained],
     'person_sets':[person_sets[i] for i in retained],'fixed_columns':list(range(len(fixed))),
     'budget':max(1,math.floor(len(fixed)*.5)),'conflicts':conflicts})
    private_shapes.append({'shapes':[pallets[i] for i in retained],'width':width,'height':height})
    progress.update(images=len(images),allowed_new_candidates=sum(len(i['point_sets'])-len(i['fixed_columns']) for i in images))
    save('progress.json',progress);print(json.dumps({'constraint_images':len(images)}),flush=True)
  names={'matched':'pallet_objects_matched','points':'pallet_points_covered','person_points':'person_points_covered_by_pallet',
   'candidates':'pallet_candidates','union_pixels':'pallet_union_pixels'}
  expected={'baseline':comparison['baseline'],'previous_four':comparison['previous_best'],
   'spatial_mass':next(r for r in comparison['rows'] if r['method']=='spatial_component_mass_guard')}
  for name,counts in expected.items():
   for key,field in names.items():assert control_totals[name][key]==counts[field],name+' '+key+' did not reproduce'
  save('control_counts.json',control_totals)
  progress.update(phase='solve_joint_numeric_constraints');save('progress.json',progress)
  solved=solve_joint_selection(images,minimum_points,maximum_person,time_limit=args.solver_seconds)
  summary=solved['summary'];witness_area=None;all_gates=None
  if solved['_selection'] is not None:
   witness_area=0
   for image,selected in zip(private_shapes,solved['_selection']):
    mask=np.zeros((image['height'],image['width']),np.uint8)
    for column in selected:mask|=_rasterize(image['shapes'][column],image['width'],image['height'])
    witness_area+=int(np.count_nonzero(mask))
   c=summary['incumbent'];b=control_totals['baseline'];total_objects=comparison['baseline']['pallet_reference_objects']
   all_gates=bool(c['matched']>=minimum_matches and c['points']>=minimum_points and c['person_points']<=maximum_person
    and c['candidates']<=b['candidates']*1.5 and witness_area<=b['union_pixels']*1.25
    and c['matched']/total_objects>=b['matched']/total_objects+.02)
  result={'status':'complete','diagnostic_only':True,'uses_reference_annotations':True,'usable_predictions_exported':False,
   'images':len(images),'new_model_requests':0,'control_counts':control_totals,'solver':summary,
   'internal_reference_chosen_witness_union_pixels':witness_area,
   'internal_reference_chosen_witness_passes_all_numeric_gates':all_gates,
   'actual_prediction_or_quality_improvement':False,'plan_sha256':digest(output/'plan.json'),
   'source_state_unchanged':digest(REPO/'work/state.json')==before,
   'frozen_inputs_unchanged':digest(args.baseline_json)==baseline_receipt['prediction_json_sha256'] and digest(args.pool_json)==pool_receipt['prediction_json_sha256'] and digest(study/'comparison.json')==plan['comparison_sha256'],
   'code_unchanged':digest(__file__)==plan['diagnostic_sha256'] and digest(REPO/'tools/joint_selection_bounds.py')==plan['solver_helper_sha256'],
   'scope':'reference-informed feasibility only; no selected IDs or shapes are written; not an achieved model score'}
  save('summary.json',result);progress.update(status='complete',phase='complete');save('progress.json',progress)
  print(json.dumps(result),flush=True)
 except Exception as error:
  progress.update(status='failed',error_type=type(error).__name__);save('progress.json',progress)
  (output/'local_error.txt').write_text(str(error),encoding='utf-8');raise


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('dataset','baseline-json','baseline-receipt','pool-json','pool-receipt','study','output'):parser.add_argument('--'+name,required=True)
 parser.add_argument('--solver-seconds',type=float,default=30.)
 run(parser.parse_args())
