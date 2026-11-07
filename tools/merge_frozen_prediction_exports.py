"""Merge disjoint, complete exports of one frozen per-image raw selection recipe.

No reference annotations, model APIs, shape edits or source receipt rewrites.
The merged export gets its own plan and receipt with all parent hashes retained.
"""
import argparse
from datetime import datetime,timezone
import hashlib,json,re
from pathlib import Path


SIGNATURE_KEYS=('version','methods','fixed_parameters','solver_seconds_per_image','raw_projection',
 'coverage_certificate','person_prediction_guard','component_objective','component_novelty',
 'prediction_uses_3D_reference','no_reference_oracle_inputs','comparison_requires','helper_sha256','code_sha256')


def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merge_studies(dataset,frozen_plan,baseline_json,baseline_receipt,studies,output,methods=None):
 dataset=Path(dataset).resolve();studies=[Path(p).resolve() for p in studies];output=Path(output).resolve()
 if not studies or len(studies)!=len(set(studies)):raise ValueError('Distinct source studies required')
 if output.is_relative_to(dataset) or any(output.is_relative_to(p) for p in studies):raise ValueError('Keep merged output outside input dataset/studies')
 if output.exists():raise ValueError('Merged output must be a fresh directory')
 source_hashes={}
 def unique_keys(pairs):
  result={}
  for key,value in pairs:
   if key in result:raise ValueError('Duplicate JSON object key')
   result[key]=value
  return result
 def read(path):
  path=Path(path).resolve()
  if any(path.is_relative_to(dataset/'source'/name) for name in ('labels','ali_label','wilddet3d-labels')):
   raise ValueError('Reference annotation input is forbidden')
  raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
  if path in source_hashes and source_hashes[path]!=sha:raise ValueError('Input changed during read')
  source_hashes[path]=sha
  return json.loads(raw,object_pairs_hook=unique_keys,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON value')))
 def dimensions(values):
  if not isinstance(values,list) or not values or any(type(v) is not int or v<0 for v in values) or len(values)!=len(set(values)):
   raise ValueError('Distinct nonnegative integer dimensions required')
  return values
 frozen=read(frozen_plan);manifest=read(dataset/'import_manifest.json')
 baseline=read(baseline_json);base_receipt=read(baseline_receipt)
 baseline_sha=digest(baseline_json)
 if baseline_sha!=base_receipt['prediction_json_sha256'] or baseline_sha!=frozen['baseline_prediction_sha256']:
  raise ValueError('Frozen baseline binding changed')
 if base_receipt['archive_sha256']!=manifest['archive_sha256']:raise ValueError('Baseline archive changed')
 frames=dimensions(frozen['full_validation_frames']);cameras=dimensions(frozen['cameras']);primary=frozen['primary_method']
 expected={(f'/capture/camera_{camera}/image',str(frame)) for camera in cameras for frame in frames}
 records=[r for r in manifest['records'] if r['topic'].endswith('/image')]
 image_records={(r['topic'],str(r['frame'])):r for r in records}
 if len(image_records)!=len(records) or not expected<=image_records.keys():raise ValueError('Invalid manifest target coverage')
 if frozen.get('full_validation_images',len(expected))!=len(expected):raise ValueError('Frozen image count differs')
 maps=None;image_hashes={};seen=set();signature=None;parent_bindings=[]
 for directory in studies:
  plan_path=directory/'study_plan.json';progress_path=directory/'progress.json'
  plan=read(plan_path);progress=read(progress_path);plan_sha=source_hashes[plan_path.resolve()]
  if progress.get('status')!='complete' or progress.get('plan_sha256')!=plan_sha or not progress.get('plan_unchanged'):
   raise ValueError('Incomplete or changed parent study')
  if progress.get('reference_files_opened')!=0 or plan.get('prediction_uses_3D_reference') is not False or plan.get('no_reference_oracle_inputs') is not True:
   raise ValueError('Source must be reference-free raw predictions')
  if any(progress.get(key) is not True for key in ('baseline_cache_unchanged','frozen_pool_and_anchors_unchanged','code_unchanged')):
   raise ValueError('Parent source bindings or implementation changed')
  if progress.get('workbench_modified') is not False:raise ValueError('Unexpected workbench write marker')
  if plan['code_sha256']!=frozen['predictor_sha256'] or plan['helper_sha256']!=frozen['helper_sha256']:
   raise ValueError('Parent implementation differs from frozen primary')
  if (plan['input_bindings']['baseline']['prediction_sha256']!=baseline_sha or
      plan['input_bindings']['baseline']['receipt_sha256']!=digest(baseline_receipt) or
      plan['comparison_requires']!=frozen['comparison_requires']):
   raise ValueError('Parent baseline or adoption requirements differ')
  if any(not re.fullmatch(r'[a-z][a-z0-9_]*',m) for m in plan['methods']):raise ValueError('Invalid method name')
  current_signature=canonical({key:plan[key] for key in SIGNATURE_KEYS})
  if signature is None:
   signature=current_signature
   methods=list(plan['methods']) if methods is None else list(methods)
   if not methods or len(methods)!=len(set(methods)) or primary not in methods or any(m not in plan['methods'] for m in methods):
    raise ValueError('Frozen primary and valid distinct methods required')
   maps={method:{} for method in methods}
  elif current_signature!=signature:raise ValueError('Source recipe parameters differ')
  own={(f'/capture/camera_{camera}/image',str(frame)) for camera in dimensions(plan['cameras']) for frame in dimensions(plan['frames'])}
  if len(own)!=plan['planned_images'] or progress['completed_images']!=len(own):raise ValueError('Parent image coverage count differs')
  if not own<=expected:raise ValueError('Source contains images outside frozen target cohort')
  if seen&own:raise ValueError('Source image cohorts overlap')
  log={}
  for row in progress['postprocessing']:
   key=(row['topic'],str(row['frame']),row['method'])
   if key in log:raise ValueError('Duplicate parent processing row')
   log[key]=row
  if set(log)!={(t,f,m) for t,f in own for m in plan['methods']}:raise ValueError('Parent processing coverage differs')
  binding={'study':directory.name,'plan_sha256':plan_sha,'progress_sha256':source_hashes[progress_path.resolve()],'images':len(own),'methods':{}}
  for method in methods:
   prediction_path=directory/(method+'_candidates.json');receipt_path=directory/(method+'_receipt.json')
   predictions=read(prediction_path);receipt=read(receipt_path);prediction_sha=source_hashes[prediction_path.resolve()]
   if receipt.get('prediction_json_sha256')!=prediction_sha or receipt.get('plan_sha256')!=plan_sha:
    raise ValueError('Parent prediction or receipt binding changed')
   if receipt.get('archive_sha256')!=manifest['archive_sha256'] or receipt.get('method')!=method:
    raise ValueError('Parent archive or method differs')
   if receipt.get('candidates_confirmed_as_truth') is not False or receipt.get('source')!='reference_free_frozen_SAM_proposals_and_raw_point_coverage':
    raise ValueError('Only unconfirmed outputs of the frozen recipe can merge')
   actual={(topic,frame) for topic,rows in predictions.items() for frame in rows}
   receipt_pairs={(topic,frame) for topic,rows in receipt['image_sha256'].items() for frame in rows}
   if actual!=own or receipt_pairs!=own:raise ValueError('Parent export or image receipt coverage differs')
   for topic,frame in own:
    if frame!=str(int(frame)):raise ValueError('Noncanonical frame key')
    expected_sha=image_records[topic,frame]['sha256']
    if receipt['image_sha256'][topic][frame]!=expected_sha or base_receipt['image_sha256'][topic][frame]!=expected_sha:
     raise ValueError('Image SHA binding differs from manifest or baseline')
    values=predictions[topic][frame];original=baseline[topic][frame]
    if not isinstance(values,list) or not isinstance(original,list):raise ValueError('Candidate arrays required')
    if canonical(values[:len(original)])!=canonical(original):raise ValueError('Original candidate prefix changed')
    identifiers=[value['id'] for value in values]
    if any(type(i) not in (str,int) for i in identifiers) or len(set(identifiers))!=len(identifiers):raise ValueError('Invalid candidate IDs')
    row=log.get((topic,frame,method))
    if not row or row.get('baseline_candidates_preserved') is not True:raise ValueError('Missing original-preservation readback')
    if method.startswith('raw_certificate_') and (row.get('raw_certificate_verified') is not True or row.get('solver',{}).get('certificate_satisfied') is not True):
     raise ValueError('Missing raw point coverage certificate')
    maps[method].setdefault(topic,{})[frame]=values
    image_hashes.setdefault(topic,{})[frame]=expected_sha
   binding['methods'][method]={'prediction_sha256':prediction_sha,'receipt_sha256':source_hashes[receipt_path.resolve()]}
  seen|=own;parent_bindings.append(binding)
 if seen!=expected:raise ValueError(f'Incomplete frozen image cohort: {len(expected-seen)} missing')
 if any(digest(path)!=value for path,value in source_hashes.items()):raise ValueError('Source changed during merge')
 output.mkdir(parents=True,exist_ok=False)
 def save(name,value):
  path=output/name;path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'));return path
 plan={'created_utc':datetime.now(timezone.utc).isoformat(),'version':1,'operation':'merge_disjoint_complete_frozen_per_image_predictions',
  'primary_method':primary,'methods':methods,'frames':frames,'cameras':cameras,'planned_images':len(expected),
  'archive_sha256':manifest['archive_sha256'],'baseline_prediction_sha256':baseline_sha,'frozen_validation_plan_sha256':digest(frozen_plan),
  'predictor_sha256':frozen['predictor_sha256'],'helper_sha256':frozen['helper_sha256'],'recipe_signature_sha256':hashlib.sha256(signature.encode()).hexdigest(),
  'parent_bindings':parent_bindings,'new_model_requests':0,'prediction_uses_reference_labels':False,'original_candidate_prefix_preserved':True,
  'merge_code_sha256':digest(__file__),'model_call_counter_scope':'merge operation only; prior crop collection is separate',
  'scope':'same-capture frozen predictions; not independent pixel accuracy or confirmed truth'}
 plan_path=save('study_plan.json',plan);plan_sha=digest(plan_path)
 ordered_hashes={t:{f:v for f,v in sorted(rows.items(),key=lambda item:int(item[0]))} for t,rows in sorted(image_hashes.items())}
 for method,rows in maps.items():
  ordered={t:{f:v for f,v in sorted(values.items(),key=lambda item:int(item[0]))} for t,values in sorted(rows.items())}
  path=save(method+'_candidates.json',ordered)
  receipt_path=save(method+'_receipt.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'archive_sha256':manifest['archive_sha256'],
   'prediction_json_sha256':digest(path),'image_sha256':ordered_hashes,'plan_sha256':plan_sha,'method':method,
   'parent_prediction_receipt_sha256':[p['methods'][method]['receipt_sha256'] for p in parent_bindings],
   'source':'merged_bound_disjoint_reference_free_predictions','new_model_requests':0,'candidates_confirmed_as_truth':False})
  if canonical(json.loads(path.read_bytes()))!=canonical(ordered) or json.loads(receipt_path.read_bytes())['prediction_json_sha256']!=digest(path):
   raise ValueError('Merged export readback differs')
 if any(digest(path)!=value for path,value in source_hashes.items()):raise ValueError('Source changed before merge readback completed')
 summary={'status':'complete','images':len(expected),'source_studies':len(studies),'methods':methods,'primary_method':primary,
  'source_bindings_unchanged':True,'original_prefix_preserved':True,'all_frozen_target_images_present':True,'new_model_requests':0,
  'reference_files_opened':0,'plan_sha256':plan_sha,'candidates_confirmed_as_truth':False,'export_readback_verified':True}
 save('progress.json',summary);return summary


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('dataset','frozen-plan','baseline-json','baseline-receipt','output'):parser.add_argument('--'+name,required=True)
 parser.add_argument('--study',action='append',required=True);parser.add_argument('--methods',help='Comma-separated method names; frozen primary must be included')
 args=parser.parse_args()
 print(json.dumps(merge_studies(args.dataset,args.frozen_plan,args.baseline_json,args.baseline_receipt,args.study,args.output,
  args.methods.split(',') if args.methods else None)),flush=True)
