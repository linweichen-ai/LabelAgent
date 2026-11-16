"""Create isolated point-verified overlap merges from real cached candidates.

The predictor reads images and prior candidates only. It never reads reference
3D labels. A separate scorer must assess quality after this output completes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys

from PIL import Image
import psutil

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from connect_local_vlm import loopback_url
from merge_candidate_overlaps import merge_added_overlaps
from point_identity_verifier import PointIdentityVerifier,PointVerificationStopped
from run_local_agent_acceptance import load_image,save


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checked_predictions(path,receipt_path,manifest):
    receipt=read(receipt_path)
    if file_sha(path)!=receipt.get('prediction_json_sha256') or receipt.get('archive_sha256')!=manifest['archive_sha256']:
        raise ValueError('Prediction receipt does not bind these candidates and dataset')
    return read(path),receipt


def evaluate(args):
    if not .3<=args.merge_iou<=1 or not .5<=args.point_iou<=1:
        raise ValueError('Invalid overlap or point-agreement threshold')
    repo=Path(__file__).resolve().parents[1]
    dataset=Path(args.dataset).resolve()
    manifest=read(dataset/'import_manifest.json')
    source,binding=checked_predictions(args.candidates_json,args.prediction_receipt,manifest)
    baseline,base_binding=checked_predictions(args.baseline_json,args.baseline_receipt,manifest)
    records={(row['topic'],str(row['frame'])):row for row in manifest['records'] if row['topic'].endswith('/image')}
    selected=sorted([(topic,frame) for topic,frames in source.items() for frame in frames],key=lambda item:(int(item[1]),item[0]))
    if not selected:raise ValueError('No image predictions selected')
    for topic,frame in selected:
        if ((topic,frame) not in records or frame not in baseline.get(topic,{})):
            raise ValueError('Selected predictions are absent from source or baseline')
        digest=records[topic,frame]['sha256']
        if (binding['image_sha256'].get(topic,{}).get(frame)!=digest or
                base_binding['image_sha256'].get(topic,{}).get(frame)!=digest):
            raise ValueError('Candidates are not bound to the selected source image')
        seed=baseline[topic][frame]
        if source[topic][frame][:len(seed)]!=seed:raise ValueError('Expected the unchanged original baseline prefix')
    sam=loopback_url(args.sam)
    selection=read(args.selection_json) if args.selection_json else None
    if selection is not None and (selection.get('point_iou')!=args.point_iou
            or selection.get('selected_using_tuning_only') is not True or selection.get('usefulness_checks_passed') is not True):
        raise ValueError('Parameters differ from the frozen useful tuning selection')
    weights=sorted((repo/'source/sam3/weights').glob('*.pt'))
    if not weights:raise ValueError('Local SAM3 weights are required')
    fingerprint_data={'weights_sha256':file_sha(weights[0]),
        'point_service_sha256':file_sha(repo/'source/sam3_service/point_segment.py')}
    fingerprint=hashlib.sha256(json.dumps(fingerprint_data,sort_keys=True).encode()).hexdigest()
    requested={'scope':args.scope,'images':len(selected),'merge_iou':args.merge_iou,'point_iou':args.point_iou,
        'source_prediction_sha256':file_sha(args.candidates_json),'baseline_sha256':file_sha(args.baseline_json),
        'archive_sha256':manifest['archive_sha256'],'model_fingerprint':fingerprint,
        'model_fingerprint_data':fingerprint_data,'source_code_sha256':file_sha(__file__),
        'merger_sha256':file_sha(repo/'tools/merge_candidate_overlaps.py'),
        'verifier_sha256':file_sha(repo/'tools/point_identity_verifier.py'),
        'selection_sha256':file_sha(args.selection_json) if args.selection_json else None,
        'semantic_class_verified_by_point_agreement':False,'reference_3D_labels_read_for_prediction':False,
        'workbench_modified':False,'candidates_confirmed_as_truth':False,'independent_scene_validation':False}
    output=Path(args.output).resolve()
    if output.is_relative_to((dataset/'source').resolve()):raise ValueError('Keep experiment output outside original source data')
    if args.resume:
        plan=read(output/'study_plan.json')
        if any(plan.get(key)!=value for key,value in requested.items()):raise ValueError('Resume code, parameters or input bytes changed')
        old=read(output/'progress.json')
        if old['status']=='complete':raise ValueError('This output is complete; do not restart it')
        try:
            command=psutil.Process(old['pid']).cmdline()
            if any('evaluate_point_merges.py' in part for part in command) and any(str(output).lower() in part.lower() for part in command):
                raise ValueError('The original matching process is still live')
        except psutil.NoSuchProcess:pass
    else:
        output.mkdir(parents=True,exist_ok=False)
        plan={**requested,'created_utc':datetime.now(timezone.utc).isoformat()}
        save(output/'study_plan.json',plan)
    plan_hash=file_sha(output/'study_plan.json')
    stats={}
    progress={'status':'running','pid':os.getpid(),'planned_images':len(selected),'completed_images':0,
        'plan_sha256':plan_hash,'stats':stats,'merged_additions':0,'image_summaries':[],'new_requests_this_process':0}
    save(output/'progress.json',progress)
    predictions={}
    try:
        for topic,frame in selected:
            if (output/'STOP').exists():raise PointVerificationStopped('Study stop requested')
            raw=load_image(dataset,records[topic,frame])
            width,height=Image.open(io.BytesIO(raw)).size
            verifier=PointIdentityVerifier(raw,width,height,sam,output/'responses',args.point_iou,fingerprint,
                stats=stats,stop_file=output/'STOP')
            seed=baseline[topic][frame]
            additions=source[topic][frame][len(seed):]
            candidates,audit=merge_added_overlaps(seed,additions,width,height,args.merge_iou,verifier=verifier)
            predictions.setdefault(topic,{})[frame]=candidates
            folder=output/'image_audits'
            folder.mkdir(exist_ok=True)
            save(folder/f"{topic.split('/')[2]}_{int(frame):04d}.json",audit)
            progress['completed_images']+=1
            progress['merged_additions']+=audit['merged_additions']
            progress['new_requests_this_process']=stats['fresh_requests']
            progress['image_summaries'].append({'topic':topic,'frame':int(frame),
                'before_candidates':len(source[topic][frame]),'after_candidates':len(candidates),
                'merged_additions':audit['merged_additions'],'exact_class_union_preserved':audit['exact_class_union_preserved']})
            save(output/'progress.json',progress)
            if progress['completed_images']%16==0:
                print(json.dumps({'images':progress['completed_images'],'merged':progress['merged_additions'],'stats':stats}),flush=True)
        file=output/'merged_candidates.json'
        save(file,predictions)
        save(output/'merged_receipt.json',{'archive_sha256':manifest['archive_sha256'],
            'prediction_json_sha256':file_sha(file),'image_sha256':binding['image_sha256'],
            'source':'experimental_real_SAM_point_verified_overlap_merge','plan_sha256':plan_hash,
            'model_fingerprint':fingerprint,'candidates_confirmed_as_truth':False})
        progress['status']='complete'
    except PointVerificationStopped:
        progress['status']='stopped'
    except Exception as error:
        progress.update(status='failed',error_type=type(error).__name__)
        (output/'local_error.txt').write_text(str(error),encoding='utf-8')
    finally:
        progress.update(finished_utc=datetime.now(timezone.utc).isoformat(),
            source_prediction_unchanged=file_sha(args.candidates_json)==plan['source_prediction_sha256'],
            baseline_unchanged=file_sha(args.baseline_json)==plan['baseline_sha256'],
            plan_unchanged=file_sha(output/'study_plan.json')==plan_hash,
            unique_cached_responses=len(list((output/'responses').glob('*.json'))) if (output/'responses').exists() else 0)
        save(output/'progress.json',progress)
    print(json.dumps({key:progress[key] for key in ('status','completed_images','merged_additions','stats','unique_cached_responses')}),flush=True)
    return 0 if progress['status']=='complete' else 2 if progress['status']=='stopped' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','candidates-json','prediction-receipt','baseline-json','baseline-receipt','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--selection-json')
    parser.add_argument('--scope',default='isolated_exploratory_repeat')
    parser.add_argument('--merge-iou',type=float,default=.5)
    parser.add_argument('--point-iou',type=float,default=.85)
    parser.add_argument('--sam',default='http://127.0.0.1:8766')
    parser.add_argument('--resume',action='store_true')
    raise SystemExit(evaluate(parser.parse_args()))
