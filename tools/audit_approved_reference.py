"""Audit fixed predictions against approved model-derived masks, without edits.

Matching and overlap are reference consistency, not independent accuracy.
Unmatched predictions and pixels outside approved masks remain unjudged.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sys
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'tools')]
from compare_boxes3d import match
from tiled_candidate_utils import _mask_record,_iou
from run_local_agent_acceptance import load_image,save
from source.agent.edit_agent import valid_shape

LABELS=('pallet','person')
THRESHOLDS=(.5,.75,.9)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def consistency(reference,predictions,width,height):
    for shape in [*reference,*predictions]:
        if shape.get('label') not in LABELS or not valid_shape({**shape,'id':1},width,height):
            raise ValueError('Supported valid mask geometry required')
    def union(records):
        value=np.zeros((height,width),np.uint8)
        for record in records:
            left,top,right,bottom=record['bbox'];value[top:bottom,left:right]|=record['mask']
        return value.astype(bool)
    def key(record):
        return record['bbox'],hashlib.sha256(np.packbits(record['mask']).tobytes()).hexdigest()
    result={}
    for label in LABELS:
        refs=[_mask_record(s,width,height) for s in reference if s['label']==label]
        preds=[_mask_record(s,width,height) for s in predictions if s['label']==label]
        matrix=np.array([[_iou(r,p) for p in preds] for r in refs],float).reshape(len(refs),len(preds))
        ref_mask,pred_mask=union(refs),union(preds)
        ref_area,pred_area=int(ref_mask.sum()),int(pred_mask.sum())
        intersection=int((ref_mask&pred_mask).sum());combined=int((ref_mask|pred_mask).sum())
        values={'approved_objects':len(refs),'prediction_objects':len(preds),
            'approved_mask_exactly_reused_objects':sum(any(key(r)==key(p) for p in preds) for r in refs),
            'mean_best_iou_per_approved_object':float(matrix.max(axis=1).mean()) if refs and preds else 0. if refs else None,
            'approved_objects_with_multiple_candidate_hits_at_0_5':int(((matrix>=.5).sum(axis=1)>1).sum()),
            'approved_union_pixels':ref_area,'prediction_union_pixels':pred_area,'intersection_pixels':intersection,
            'approved_union_coverage':intersection/ref_area if ref_area else None,
            'union_iou_against_approved':intersection/combined if combined else None,
            'prediction_pixels_outside_approved_union':int((pred_mask&~ref_mask).sum()),
            'outside_pixels_are_verified_errors':False,'matches':{}}
        for threshold in THRESHOLDS:
            pairs=match(matrix,threshold)
            values['matches'][str(threshold)]={'matched_approved_objects':len(pairs),
                'unmatched_approved_objects':len(refs)-len(pairs),'unmatched_prediction_objects':len(preds)-len(pairs),
                'matched_iou_mean':float(np.mean([p[2] for p in pairs])) if pairs else None,
                'unmatched_predictions_are_verified_errors':False}
        result[label]=values
    return result


def audit(fixture_json,output):
    fixture_path=Path(fixture_json).resolve();fixture=json.loads(fixture_path.read_bytes())
    output=Path(output).resolve();dataset=Path(fixture['dataset']).resolve()
    if output.exists() or output.is_relative_to(dataset):raise ValueError('Fresh isolated audit output required')
    bindings={}
    def read(path):
        path=Path(path).resolve();raw=path.read_bytes();value=hashlib.sha256(raw).hexdigest()
        if path in bindings and bindings[path]!=value:raise ValueError('Input changed')
        bindings[path]=value;return json.loads(raw)
    approval=read(fixture['approval']);draft=read(fixture['draft_receipt']);reference=read(fixture['reference'])
    seed=read(fixture['source_seed']);manifest=read(dataset/'import_manifest.json')
    if (approval.get('human_approved') is not True or approval['approved_draft_sha256']!=sha(fixture['reference'])
        or draft['draft_sha256']!=sha(fixture['reference']) or len(reference)!=approval['reference_shapes']):
        raise ValueError('Reference approval or version binding differs')
    if sha(fixture['source_seed'])!=draft['source_seed_sha256']:raise ValueError('Original model seed changed')
    source={s['id']:s for s in seed};references={s['id']:s for s in reference}
    if len(references)!=len(reference) or len(draft['selected_mapping'])!=len(reference):raise ValueError('Ambiguous reference mapping')
    for row in draft['selected_mapping']:
        expected=deepcopy(source[row['source_id']]);expected['id']=row['draft_id']
        if canonical(expected)!=canonical(references[row['draft_id']]):raise ValueError('Reference differs from recorded seed mapping')
    topic,frame=draft['topic'],str(draft['frame'])
    if approval['topic']!=topic or approval['reference_frame']!=int(frame):raise ValueError('Approved image differs')
    record=next(r for r in manifest['records'] if r['topic']==topic and str(r['frame'])==frame)
    if record['sha256']!=draft['source_image_sha256']:raise ValueError('Reference image binding differs')
    from PIL import Image
    import io
    raw_image=load_image(dataset,record)
    width,height=Image.open(io.BytesIO(raw_image)).size
    predictions=[]
    names=set()
    for entry in fixture['predictions']:
        name=entry['name']
        if name in names:raise ValueError('Duplicate prediction name')
        names.add(name);receipt=read(entry['receipt']);mapping=read(entry['path'])
        if receipt['prediction_json_sha256']!=sha(entry['path']) or receipt['archive_sha256']!=manifest['archive_sha256']:
            raise ValueError('Prediction receipt changed')
        if receipt['image_sha256'][topic][frame]!=record['sha256']:raise ValueError('Prediction image differs')
        predictions.append((name,mapping[topic][frame]))
    bindings[fixture_path]=sha(fixture_path)
    helpers=[REPO/'tools'/name for name in ('compare_boxes3d.py','tiled_candidate_utils.py','run_local_agent_acceptance.py')]
    helpers.append(REPO/'source/agent/edit_agent.py')
    helper_bindings={str(p):sha(p) for p in helpers}
    output.mkdir(parents=True,exist_ok=False)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'code_sha256':sha(__file__),
        'fixture_sha256':sha(fixture_path),'input_bindings':{str(p):h for p,h in bindings.items()},'helper_bindings':helper_bindings,
        'topic':topic,'frame':int(frame),'image_sha256':record['sha256'],'approved_reference_sha256':sha(fixture['reference']),
        'thresholds':list(THRESHOLDS),'scope':'approved model-derived mask consistency on one image',
        'independent_reannotation':False,'unannotated_region_is_verified_negative':False,
        'human_approved':True,'reference_geometry_copied_from_model_seed':True,
        'model_calls':0,'prediction_exports_created':False,'workbench_writes':False}
    save(output/'study_plan.json',plan)
    reports={name:consistency(reference,values,width,height) for name,values in predictions}
    if any(sha(p)!=h for p,h in bindings.items()) or any(sha(p)!=h for p,h in helper_bindings.items()) or sha(__file__)!=plan['code_sha256']:
        raise ValueError('Input or audit code changed')
    result={'status':'complete','scope':plan['scope'],'images':1,'approved_objects':len(reference),
        'human_approved':True,'reference_geometry_copied_from_model_seed':True,'independent_reannotation':False,
        'unannotated_region_is_verified_negative':False,'independent_accuracy_measured':False,
        'methods':reports,'all_source_bindings_unchanged':True,'model_calls':0,'prediction_exports_created':False,
        'workbench_writes':False,'image_sha256':record['sha256'],'reference_sha256':sha(fixture['reference'])}
    save(output/'summary.json',result)
    print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-json',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();audit(args.fixture_json,args.output)
