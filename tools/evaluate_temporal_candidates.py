"""Frozen offline temporal proposal pilot, without reference labels or model calls."""
import argparse
from collections import Counter
from datetime import datetime,timezone
from functools import lru_cache
import io
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from evaluate_point_merges import checked_predictions,file_sha,read
from run_local_agent_acceptance import load_image,save
from temporal_candidate_utils import (estimate_bidirectional_flow,prepare_flow_warp,warp_candidate,
    exclude_predicted_person,restrict_to_semantic_support,augment_preserving_tiled)
from tiled_candidate_utils import augment_candidates
from source.agent.edit_agent import _rasterize


def evaluate(args):
    repo=Path(__file__).resolve().parents[1]
    dataset=Path(args.dataset).resolve()
    plan=read(args.plan_json)
    manifest=read(dataset/'import_manifest.json')
    baseline,baseline_binding=checked_predictions(args.baseline_json,args.baseline_receipt,manifest)
    tiled,tiled_binding=checked_predictions(args.tiled_json,args.tiled_receipt,manifest)
    if (file_sha(args.baseline_json)!=plan['baseline_sha256'] or file_sha(args.tiled_json)!=plan['tiled_comparator_sha256']
            or file_sha(repo/'tools/evaluate_projected_support.py')!=plan['metric_sha256']):
        raise ValueError('Source candidates or metric differ from the frozen pilot plan')
    records={(row['topic'],str(row['frame'])):row for row in manifest['records'] if row['topic'].endswith('/image')}
    selected=[(f'/capture/camera_{camera}/image',str(frame)) for frame in plan['frames'] for camera in plan['cameras']]
    if len(set(selected))!=len(selected) or not selected:raise ValueError('Invalid selected images')
    for topic,frame in selected:
        if tiled.get(topic,{}).get(frame,[])[:len(baseline[topic][frame])]!=baseline[topic][frame]:
            raise ValueError('Tiled candidates must preserve the exact original baseline prefix')
        if tiled_binding['image_sha256'][topic][frame]!=records[topic,frame]['sha256']:
            raise ValueError('Tiled candidates do not bind the selected source image')
        for offset in [0,*plan['neighbors']]:
            neighbor=str(int(frame)+offset)
            if (topic,neighbor) not in records or neighbor not in baseline.get(topic,{}):
                raise ValueError('Required neighbor image or baseline missing')
            if baseline_binding['image_sha256'][topic][neighbor]!=records[topic,neighbor]['sha256']:
                raise ValueError('Neighbor candidates do not bind the actual source image')
    output=Path(args.output).resolve()
    if output.is_relative_to((dataset/'source').resolve()):raise ValueError('Keep output outside original source data')
    output.mkdir(parents=True,exist_ok=False)
    run_plan={'created_utc':datetime.now(timezone.utc).isoformat(),'recipe':plan,
        'frozen_recipe_sha256':file_sha(args.plan_json),'source_code_sha256':file_sha(__file__),
        'operator_sha256':file_sha(repo/'tools/temporal_candidate_utils.py'),
        'augmentation_sha256':file_sha(repo/'tools/tiled_candidate_utils.py'),
        'archive_sha256':manifest['archive_sha256'],'opencv_version':cv2.__version__,
        'images':len(selected),'new_model_requests':0,'reference_3D_labels_read_for_prediction':False,
        'candidates_confirmed_as_truth':False,'workbench_modified':False,'independent_validation':False}
    save(output/'study_plan.json',run_plan)
    plan_sha=file_sha(output/'study_plan.json')
    state_before=file_sha(repo/'work/state.json')
    progress={'status':'running','pid':os.getpid(),'planned_images':len(selected),'completed_images':0,
        'plan_sha256':plan_sha,'flow_pairs':0,'source_pallet_hypotheses':0,'accepted_warps':0,
        'rejection_reasons':{},'image_summaries':[],'new_model_requests':0}
    save(output/'progress.json',progress)
    outputs={name:{} for name in plan['variants']}
    image_bindings={}
    reasons=Counter()
    cv2.setNumThreads(2)

    @lru_cache(maxsize=3)
    def image(topic,frame):
        return np.asarray(Image.open(io.BytesIO(load_image(dataset,records[topic,frame]))).convert('RGB'))

    try:
        for topic,frame in selected:
            if (output/'STOP').exists():raise InterruptedError('Study stop requested')
            target=image(topic,frame)
            height,width=target.shape[:2]
            guard=plan.get('predicted_person_occlusion_guard',{})
            semantic=plan.get('target_semantic_support',{})
            person_union=np.zeros((height,width),dtype=np.uint8)
            semantic_union=np.zeros((height,width),dtype=np.uint8)
            if guard.get('enabled'):
                for value in baseline[topic][frame]:
                    if value['label']=='person':person_union|=_rasterize(value,width,height)
            if semantic.get('enabled'):
                for value in tiled[topic][frame]:
                    if value['label']=='pallet':semantic_union|=_rasterize(value,width,height)
            proposals=[]
            audit=[]
            for offset in plan['neighbors']:
                if (output/'STOP').exists():raise InterruptedError('Study stop requested')
                neighbor=str(int(frame)+offset)
                seeds=[value for value in baseline[topic][neighbor] if value['label']=='pallet']
                if not seeds:continue
                forward,backward=estimate_bidirectional_flow(image(topic,neighbor),target,plan['image_flow'])
                warp=prepare_flow_warp(forward,backward,width,height,plan['candidate_checks']['forward_backward_error_pixels'])
                progress['flow_pairs']+=1
                for index,source in enumerate(seeds):
                    identifier=f"flow-{topic.split('/')[2]}-{frame}-from-{neighbor}-{index}"
                    candidate,row=warp_candidate(source,warp,identifier,plan['candidate_checks'])
                    if candidate is not None and guard.get('enabled'):
                        candidate,protection=exclude_predicted_person(candidate,person_union,guard)
                        row.update(flow_checks_passed=True,person_guard=protection,accepted=candidate is not None)
                        if candidate is None:row['reason']=protection['reason']
                    if candidate is not None and semantic.get('enabled'):
                        candidate,support_check=restrict_to_semantic_support(candidate,semantic_union,semantic)
                        row.update(semantic_support=support_check,accepted=candidate is not None)
                        if candidate is None:row['reason']=support_check['reason']
                    audit.append({'source_frame':int(neighbor),'source_candidate_id':source['id'],'proposal_id':identifier,**row})
                    progress['source_pallet_hypotheses']+=1
                    if candidate is not None:proposals.append(candidate)
                    else:reasons[row['reason']]+=1
                del warp,forward,backward
            seed=baseline[topic][frame]
            existing=tiled[topic][frame][len(seed):]
            image_info={'topic':topic,'frame':int(frame),'accepted_warps':len(proposals),'variants':{}}
            progress['accepted_warps']+=len(proposals)
            for name in plan['variants']:
                if name=='baseline_plus_temporal':combined=proposals
                elif name=='baseline_plus_existing_tiles_and_temporal':combined=[*existing,*proposals]
                elif name=='tiled_plus_semantically_constrained_temporal':combined=proposals
                else:raise ValueError('Unknown frozen temporal variant')
                if name=='tiled_plus_semantically_constrained_temporal':
                    result,info=augment_preserving_tiled(seed,tiled[topic][frame],combined,width,height,
                        {'minimum_score':plan['candidate_checks']['minimum_inherited_source_score'],
                            'addition_ratio':plan['selection']['addition_ratio'],'nms_iou':plan['selection']['nms_iou']})
                else:
                    result,info=augment_candidates(seed,combined,width,height,
                        minimum_score=plan['candidate_checks']['minimum_inherited_source_score'],
                        addition_ratio=plan['selection']['addition_ratio'],nms_iou=plan['selection']['nms_iou'])
                if result[:len(seed)]!=seed or [value for value in result if value['label']=='person']!=[value for value in seed if value['label']=='person']:
                    raise AssertionError('Original baseline or person shapes changed')
                outputs[name].setdefault(topic,{})[frame]=result
                image_info['variants'][name]=info
            folder=output/'image_audits'
            folder.mkdir(exist_ok=True)
            save(folder/f"{topic.split('/')[2]}_{int(frame):04d}.json",{'warp_checks':audit,'selection':image_info})
            image_bindings.setdefault(topic,{})[frame]=records[topic,frame]['sha256']
            progress['completed_images']+=1
            progress['rejection_reasons']=dict(reasons)
            progress['image_summaries'].append(image_info)
            save(output/'progress.json',progress)
            print(json.dumps({'images':progress['completed_images'],'flow_pairs':progress['flow_pairs'],
                'accepted_warps':progress['accepted_warps']}),flush=True)
        for name,predictions in outputs.items():
            file=output/(name+'_candidates.json')
            save(file,predictions)
            save(output/(name+'_receipt.json'),{'archive_sha256':manifest['archive_sha256'],
                'prediction_json_sha256':file_sha(file),'image_sha256':image_bindings,
                'source':'cached_real_SAM3_masks_transported_by_image_only_optical_flow',
                'plan_sha256':plan_sha,'new_model_requests':0,'inherited_score_is_target_confidence':False,
                'candidates_confirmed_as_truth':False})
        progress['status']='complete'
    except InterruptedError:
        progress['status']='stopped'
    except Exception as error:
        progress.update(status='failed',error_type=type(error).__name__)
        (output/'local_error.txt').write_text(str(error),encoding='utf-8')
    finally:
        progress.update(finished_utc=datetime.now(timezone.utc).isoformat(),
            source_baseline_unchanged=file_sha(args.baseline_json)==plan['baseline_sha256'],
            source_tiles_unchanged=file_sha(args.tiled_json)==plan['tiled_comparator_sha256'],
            recipe_unchanged=file_sha(args.plan_json)==run_plan['frozen_recipe_sha256'],
            plan_unchanged=file_sha(output/'study_plan.json')==plan_sha,
            saved_workbench_state_unchanged=file_sha(repo/'work/state.json')==state_before)
        save(output/'progress.json',progress)
    print(json.dumps({key:progress[key] for key in ('status','completed_images','flow_pairs','accepted_warps','rejection_reasons')}),flush=True)
    return 0 if progress['status']=='complete' else 2 if progress['status']=='stopped' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ('dataset','baseline-json','baseline-receipt','tiled-json','tiled-receipt','plan-json','output'):
        parser.add_argument('--'+option,required=True)
    raise SystemExit(evaluate(parser.parse_args()))
