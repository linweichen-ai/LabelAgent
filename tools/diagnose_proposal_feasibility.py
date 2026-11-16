"""Reference-informed diagnostic bounds, NOT a predictor or usable annotation.

Reads labels intentionally after predictions have been frozen. It writes only
counts and bounds, never selected candidate sets. Pairwise new-candidate NMS,
union area and the reference-person point cap are relaxed, so bounds can only
be optimistic for the actual selection problem.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.evaluate_projected_support import checked_source,project_visible,class_membership
from tools.compare_boxes3d import match
from tools.lidar_candidate_evidence import read_xyz,sensor_cameras
from tools.tiled_candidate_utils import _mask_record,_iou
from source.agent.edit_agent import _rasterize


def matching_bound(matrix,fixed,available,budget):
    if type(budget) is not int or budget<0:raise ValueError('Nonnegative integer budget required')
    fixed=list(dict.fromkeys(fixed));available=[i for i in dict.fromkeys(available) if i not in fixed]
    rank_fixed=len(match(matrix[:,fixed],.5))
    rank_union=len(match(matrix[:,fixed+available],.5))
    # Matching rank is a transversal matroid: at most one new rank per new
    # column; an augmenting sequence reaches the union rank if budget allows.
    return min(rank_fixed+budget,rank_union)


def reference_matrix(shapes,points,camera,annotation):
    bits,owner,boxes=class_membership(points,annotation['objects'])
    source,pixels=project_visible(points,camera,cell_size=8,depth_slack=.15,near=.2)
    owners=owner[source];semantic=bits[source]
    eligible=[np.flatnonzero(owners==i+1) for i,b in enumerate(boxes)
              if b['bit']==1 and np.count_nonzero(owners==i+1)>=5]
    known=pixels[(semantic==1)&(owners>0)]
    matrix=np.zeros((len(eligible),len(shapes)))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    for column,shape in enumerate(shapes):
        mask=cv2.dilate(_rasterize(shape,camera['width'],camera['height']),kernel)
        denominator=int(mask[known[:,1],known[:,0]].sum()) if len(known) else 0
        for row,indices in enumerate(eligible):
            samples=pixels[indices];hits=int(mask[samples[:,1],samples[:,0]].sum())
            if denominator and hits/denominator>=.67:matrix[row,column]=hits/len(indices)
    return matrix


def run(args):
    dataset=Path(args.dataset).resolve();manifest=json.loads((dataset/'import_manifest.json').read_text(encoding='utf-8'))
    def bound(path,receipt):
        raw=Path(path).read_bytes();r=json.loads(Path(receipt).read_text(encoding='utf-8'))
        if hashlib.sha256(raw).hexdigest()!=r['prediction_json_sha256'] or r['archive_sha256']!=manifest['archive_sha256']:
            raise ValueError('Prediction receipt changed')
        return json.loads(raw),r
    baseline,b=bound(args.baseline_json,args.baseline_receipt)
    pool,p=bound(args.pool_json,args.pool_receipt)
    selected,s=bound(args.selected_json,args.selected_receipt)
    output=Path(args.output).resolve()
    if output.is_relative_to(dataset):raise ValueError('Keep diagnostic output outside the source dataset')
    output.mkdir(parents=True,exist_ok=False)
    save=lambda name,value:(output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    save('plan.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'uses_reference_annotations':True,
        'diagnostic_only':True,'usable_predictions_exported':False,'new_model_requests':0,
        'minimum_score':.5,'nms_iou':.8,'addition_ratio':.5,'radii_pixels':3,'minimum_points':5,'minimum_purity':.67,
        'relaxed_constraints':['pairwise new-candidate NMS','union area cap','reference-person point cap'],
        'baseline_sha256':b['prediction_json_sha256'],'pool_sha256':p['prediction_json_sha256'],
        'selected_sha256':s['prediction_json_sha256'],'metric_sha256':hashlib.sha256((REPO/'tools/evaluate_projected_support.py').read_bytes()).hexdigest(),
        'diagnostic_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    cameras,_=sensor_cameras(dataset,manifest,range(4));clouds={};rows=[]
    records={(r['topic'],str(r['frame'])):r for r in manifest['records'] if r['topic'].endswith('/image')}
    for topic,frames in pool.items():
        for frame,values in frames.items():
            for receipt in (b,p,s):
                if receipt['image_sha256'].get(topic,{}).get(frame)!=records[topic,frame]['sha256']:
                    raise ValueError('Source image binding differs')
            original=baseline[topic][frame];four=selected[topic][frame]
            if values[:len(original)]!=original or four[:len(original)]!=original:
                raise ValueError('Original candidates were not preserved')
            camera=cameras[int(topic.split('/')[2].split('_')[-1])]
            width,height=camera['width'],camera['height']
            if frame not in clouds:clouds[frame]=read_xyz(dataset,manifest,int(frame))[0]
            stamp=manifest['timestamps_ns'][int(frame)]
            annotation=json.loads(checked_source(dataset,manifest,f'labels/{stamp}.json'))
            pallets=[v for v in values if v['label']=='pallet']
            identifiers={v['id']:i for i,v in enumerate(pallets)}
            if len(identifiers)!=len(pallets):raise ValueError('Duplicate candidate IDs')
            matrix=reference_matrix(pallets,clouds[frame],camera,annotation)
            fixed=[identifiers[v['id']] for v in original if v['label']=='pallet']
            keep_four=[identifiers[v['id']] for v in four if v['label']=='pallet']
            records_masks=[_mask_record(v,width,height) for v in pallets]
            persons=np.zeros((height,width),dtype=np.uint8)
            for v in original:
                if v['label']=='person':persons|=_rasterize(v,width,height)
            guarded=[];allowed=[];preserved=[];preserved_unrestricted=[]
            for index,v in enumerate(pallets):
                if index in fixed or v.get('score',0)<.5:continue
                if any(_iou(records_masks[index],records_masks[j])>=.8 for j in fixed):continue
                allowed.append(index)
                r=records_masks[index];x0,y0,x1,y1=r['bbox']
                if not np.any(r['mask']&persons[y0:y1,x0:x1]):guarded.append(index)
                if index not in keep_four and not any(_iou(r,records_masks[j])>=.8 for j in keep_four):
                    preserved_unrestricted.append(index)
                if index not in keep_four and not np.any(r['mask']&persons[y0:y1,x0:x1]) and not any(
                        _iou(r,records_masks[j])>=.8 for j in keep_four):preserved.append(index)
            budget=max(1,math.floor(len(fixed)*.5));remaining=budget-(len(keep_four)-len(fixed))
            if remaining<0:raise ValueError('Selected set exceeds original budget')
            row={'frame':int(frame),'topic':topic,'reference_objects':matrix.shape[0],
                'baseline_matched':len(match(matrix[:,fixed],.5)),
                'four_crop_matched':len(match(matrix[:,keep_four],.5)),
                'all_pool_matched_unbounded':len(match(matrix,.5)),
                'score_baseline_NMS_budget_upper_bound':matching_bound(matrix,fixed,allowed,budget),
                'with_zero_person_overlap_upper_bound':matching_bound(matrix,fixed,guarded,budget),
                'preserved_four_unrestricted_upper_bound':matching_bound(matrix,keep_four,preserved_unrestricted,remaining),
                'preserved_four_guarded_upper_bound':matching_bound(matrix,keep_four,preserved,remaining),
                'original_addition_budget':budget,'remaining_after_four':remaining}
            rows.append(row);save('progress.json',{'status':'running','images':len(rows),'rows':rows})
    keys=['reference_objects','baseline_matched','four_crop_matched','all_pool_matched_unbounded',
        'score_baseline_NMS_budget_upper_bound','with_zero_person_overlap_upper_bound',
        'preserved_four_unrestricted_upper_bound','preserved_four_guarded_upper_bound']
    summary={'status':'complete','diagnostic_only':True,'uses_reference_annotations':True,
        'usable_predictions_exported':False,'new_model_requests':0,'images':len(rows),
        'totals':{k:sum(r[k] for r in rows) for k in keys},'rows':rows,
        'interpretation':'Optimistic matching bounds, not achieved model results or validated segmentation accuracy'}
    save('summary.json',summary)
    save('progress.json',{'status':'complete','images':len(rows),'rows':rows})
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','baseline-json','baseline-receipt','pool-json','pool-receipt','selected-json','selected-receipt','output'):
        parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
