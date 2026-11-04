"""Reuse bound real SAM crop responses for image-only multiscale proposals.

Reads no 3D labels, point clouds, calibration, workbench API, or model API.
All proposals remain unconfirmed; default workbench behavior is unchanged.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys

from PIL import Image

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from run_local_agent_acceptance import load_image,save
from tiled_candidate_utils import augment_candidates,augment_preserving_selected,proposal_priorities,remap_tile_result,tile_grid

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))

def replay(args):
    if sum((args.preserve_four,args.diagnostic_all_proposals,args.ranking_study,args.diversity_study,args.preserved_ranking_study))>1:
        raise ValueError('Preserved-budget, unbounded diagnostic and ranking study modes are mutually exclusive')
    dataset=Path(args.dataset).resolve()
    manifest=read(dataset/'import_manifest.json')
    baseline_path=Path(args.candidates_json).resolve()
    baseline=read(baseline_path)
    binding=read(Path(args.prediction_receipt))
    if digest(baseline_path)!=binding['prediction_json_sha256'] or binding['archive_sha256']!=manifest['archive_sha256']:
        raise ValueError('Baseline export is not bound to the imported dataset')
    studies=[Path(value).resolve() for value in args.study]
    if len(studies)!=2 or len(set(studies))!=2:
        raise ValueError('Exactly two distinct complete grid studies are required')
    plans=[read(path/'study_plan.json') for path in studies]
    progress=[read(path/'progress.json') for path in studies]
    for directory,plan,run in zip(studies,plans,progress):
        if (run['status']!='complete' or run['completed_requests']!=plan['planned_SAM3_requests']
                or digest(directory/'study_plan.json')!=run['plan_sha256']
                or plan['baseline_prediction_sha256']!=digest(baseline_path)
                or plan['archive_sha256']!=manifest['archive_sha256'] or plan['query']!='pallet'):
            raise ValueError('Source study is incomplete or its binding changed')
    if plans[0]['frames']!=plans[1]['frames'] or plans[0]['cameras']!=plans[1]['cameras']:
        raise ValueError('Source studies must use identical frames and cameras')
    grids=[plan.get('grid_spec',{'divisions':2,'crop_fraction':plan['crop_fraction']}) for plan in plans]
    if grids!=[{'divisions':2,'crop_fraction':.6},{'divisions':3,'crop_fraction':.45}]:
        raise ValueError('This comparison requires the original four-crop and new nine-crop studies in order')
    methods={'four_crop_control':{'pool':'four','maximum_person_overlap':None},
        'multiscale':{'pool':'both','maximum_person_overlap':None},
        'multiscale_person_guard':{'pool':'both','maximum_person_overlap':0.},
        'dense_person_guard':{'pool':'dense','maximum_person_overlap':0.}}
    if args.preserve_four:
        methods={'four_crop_control':methods['four_crop_control'],
            'preserve_four_add_dense':{'pool':'dense','maximum_person_overlap':None,'preserve_four':True},
            'preserve_four_add_dense_guard':{'pool':'dense','maximum_person_overlap':0.,'preserve_four':True}}
    if args.diagnostic_all_proposals:
        methods={'four_crop_control':methods['four_crop_control'],
            'all_four_proposals':{'pool':'four','maximum_person_overlap':None,'export_all':True},
            'all_dense_proposals':{'pool':'dense','maximum_person_overlap':None,'export_all':True},
            'all_multiscale_proposals':{'pool':'both','maximum_person_overlap':None,'export_all':True}}
    if args.ranking_study:
        methods={'four_crop_control':methods['four_crop_control'],
            'multiscale_novelty_control':{'pool':'both','maximum_person_overlap':None},
            **{f'{mode}_rank'+('_person_guard' if guard else ''):
                {'pool':'both','maximum_person_overlap':0. if guard else None,'ranking':mode}
                for mode in ('confidence','consensus') for guard in (False,True)}}
    if args.diversity_study:
        methods={'four_crop_control':methods['four_crop_control'],
            'multiscale_novelty_control':{'pool':'both','maximum_person_overlap':None},
            **{'novel_consensus_rank'+('_person_guard' if guard else ''):
                {'pool':'both','maximum_person_overlap':0. if guard else None,'ranking':'novel_consensus'} for guard in (False,True)},
            **{'dynamic_diversity_rank'+('_person_guard' if guard else ''):
                {'pool':'both','maximum_person_overlap':0. if guard else None,'dynamic_diversity':True} for guard in (False,True)}}
    if args.preserved_ranking_study:
        methods={'four_crop_control':methods['four_crop_control'],
            'preserved_novelty_control':{'pool':'dense','maximum_person_overlap':0.,'preserve_four':True},
            'preserve_four_novel_consensus_guard':{'pool':'dense','maximum_person_overlap':0.,'preserve_four':True,'ranking':'novel_consensus'},
            'preserve_four_dynamic_guard':{'pool':'dense','maximum_person_overlap':0.,'preserve_four':True,'dynamic_diversity':True}}
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'version':1,
        'frames':plans[0]['frames'],'cameras':plans[0]['cameras'],
        'planned_images':len(plans[0]['frames'])*len(plans[0]['cameras']),
        'new_model_requests':0,'historical_real_SAM3_responses_reused':sum(p['planned_SAM3_requests'] for p in plans),
        'parent_plan_sha256':[p['plan_sha256'] for p in progress],
        'methods':methods,'diagnostic_only':args.diagnostic_all_proposals,
        'priority_seed':'previous_four_crop_selection' if args.preserved_ranking_study else 'original_candidates',
        'ranking_specifications':{'confidence':'original SAM3 confidence',
            'consensus':'confidence*(0.5+0.5*max_mask_IoU_to_the_other_crop_scale)',
            'novel_consensus':'confidence*(1-max_mask_IoU_to_original_pallets)*(0.5+0.5*cross_scale_IoU)',
            'dynamic_diversity':'recompute confidence*(1-max_mask_IoU_to_original_and_accepted_pallets) after each addition',
            'tie_breaking':'original model confidence then earlier proposal input index'},
        'fixed_parameters':{'minimum_score':.5,'addition_ratio':.5,'nms_iou':.8},
        'prediction_uses_3D_reference':False,'person_guard_source':'original SAM3 person predictions, not reference truth',
        'archive_sha256':manifest['archive_sha256'],'baseline_prediction_sha256':digest(baseline_path),
        'code_sha256':digest(Path(__file__)),
        'postprocessor_sha256':digest(Path(__file__).with_name('tiled_candidate_utils.py')),
        'comparison_requires':{'strict_instance_gain_over_previous_best':True,
            'point_coverage_not_below_previous_best':True,'person_off_class_not_above_previous_best':True,
            'original_minimum_absolute_gain':.02,'original_maximum_candidate_count_ratio':1.5,
            'original_maximum_union_area_ratio':1.25,'original_person_off_class_increase_limit':.005,
            'all_original_candidates_preserved':True},
        'scope':'exploratory_same_capture_tuning_not_independent_accuracy','automatic_broader_run':False}
    save(output/'study_plan.json',plan)
    plan_hash=digest(output/'study_plan.json')
    before=digest(Path(args.state_file)) if args.state_file else None
    run={'status':'running','pid':os.getpid(),'completed_images':0,'new_model_requests':0,
        'verified_historical_responses':0,'postprocessing':[],'source_response_bindings':[],'workbench_modified':False}
    save(output/'progress.json',run)
    maps={method:{} for method in methods}
    inputs={}
    try:
        topics={f'/capture/camera_{camera}/image' for camera in plan['cameras']}
        records=sorted((row for row in manifest['records'] if row['topic'] in topics and row['frame'] in plan['frames']),key=lambda r:(r['frame'],r['topic']))
        if len(records)!=plan['planned_images']:raise ValueError('Incomplete source selection')
        for row in records:
            topic,frame=row['topic'],str(row['frame'])
            if binding['image_sha256'].get(topic,{}).get(frame)!=row['sha256']:
                raise ValueError('Baseline image binding differs')
            image=Image.open(io.BytesIO(load_image(dataset,row))).convert('RGB')
            width,height=image.size
            pools=[]
            for scale,(directory,source_plan,source_progress,grid) in enumerate(zip(studies,plans,progress,grids)):
                proposals=[]
                for tile,bounds in enumerate(tile_grid(width,height,grid['crop_fraction'],grid['divisions'])):
                    path=directory/'responses'/f"{topic.split('/')[2]}_{int(frame):04d}_tile_{tile}.json"
                    receipt=read(path)
                    buffer=io.BytesIO();image.crop(bounds).save(buffer,'PNG')
                    if (receipt['plan_sha256']!=source_progress['plan_sha256'] or receipt['source_image_sha256']!=row['sha256']
                            or receipt['bounds']!=list(bounds) or receipt['query']!='pallet'
                            or receipt['crop_png_sha256']!=hashlib.sha256(buffer.getvalue()).hexdigest()):
                        raise ValueError('Historical crop response does not match image and geometry')
                    run['source_response_bindings'].append({'file':path.name,'scale':scale,'sha256':digest(path)})
                    run['verified_historical_responses']+=1
                    for index,value in enumerate(receipt['response']['results']):
                        prefix='' if scale==0 else 'dense-'
                        identifier=f"{prefix}tile-{topic.split('/')[2]}-{frame}-{tile}-{index}"
                        mapped=remap_tile_result(value,bounds,width,height,identifier,source_plan['internal_edge_margin_pixels'])
                        if mapped is not None:proposals.append(mapped)
                pools.append(proposals)
            seed=baseline[topic][frame]
            selected_four,_=augment_candidates(seed,pools[0],width,height,**plan['fixed_parameters'])
            priorities={mode:proposal_priorities(pools[0],pools[1],width,height,mode,
                        baseline=selected_four if args.preserved_ranking_study else seed)
                        for mode in {settings['ranking'] for settings in methods.values() if 'ranking' in settings}}
            inputs.setdefault(topic,{})[frame]=row['sha256']
            for method,settings in methods.items():
                proposals=pools[0] if settings['pool']=='four' else pools[1] if settings['pool']=='dense' else pools[0]+pools[1]
                priority=priorities.get(settings.get('ranking'))
                if priority is not None:priority={shape['id']:priority[shape['id']] for shape in proposals}
                if settings.get('export_all'):
                    candidates=deepcopy(seed+proposals)
                    if len({shape['id'] for shape in candidates})!=len(candidates):
                        raise ValueError('Proposal identifiers collide in diagnostic export')
                    info={'baseline_candidates_preserved':True,'added_pallets':len(proposals),
                          'diagnostic_only':True,'addition_budget_enforced':False}
                elif settings.get('preserve_four'):
                    candidates,info=augment_preserving_selected(seed,selected_four,proposals,width,height,
                        **plan['fixed_parameters'],maximum_person_overlap=settings['maximum_person_overlap'],
                        ranking_scores=priority,dynamic_diversity=settings.get('dynamic_diversity',False))
                else:
                    candidates,info=augment_candidates(seed,proposals,width,height,
                        **plan['fixed_parameters'],maximum_person_overlap=settings['maximum_person_overlap'],
                        ranking_scores=priority,
                        dynamic_diversity=settings.get('dynamic_diversity',False))
                if candidates[:len(seed)]!=seed:raise ValueError('An original candidate changed')
                maps[method].setdefault(topic,{})[frame]=candidates
                run['postprocessing'].append({'frame':row['frame'],'topic':topic,'method':method,**info})
            run['completed_images']+=1
            save(output/'progress.json',run)
            print(json.dumps({'images':run['completed_images'],'reused_requests':run['verified_historical_responses']}),flush=True)
        if run['verified_historical_responses']!=plan['historical_real_SAM3_responses_reused']:
            raise ValueError('Incomplete historical request coverage')
        for method,predictions in maps.items():
            path=output/(method+'_candidates.json')
            save(path,predictions)
            save(output/(method+'_receipt.json'),{'created_utc':datetime.now(timezone.utc).isoformat(),
                'archive_sha256':manifest['archive_sha256'],'prediction_json_sha256':digest(path),
                'image_sha256':inputs,'plan_sha256':plan_hash,'method':method,'new_model_requests':0,
                'source':'replayed_bound_real_SAM3_crop_responses','candidates_confirmed_as_truth':False})
        run['status']='complete'
    except Exception as error:
        run.update(status='failed',error_type=type(error).__name__)
        (output/'local_error.txt').write_text(str(error),encoding='utf-8')
    finally:
        run.update(plan_sha256=plan_hash,plan_unchanged=digest(output/'study_plan.json')==plan_hash,
            source_plans_unchanged=all(digest(p/'study_plan.json')==r['plan_sha256'] for p,r in zip(studies,progress)),
            baseline_cache_unchanged=digest(baseline_path)==binding['prediction_json_sha256'],
            state_sha256_unchanged=digest(Path(args.state_file))==before if args.state_file else None,
            finished_utc=datetime.now(timezone.utc).isoformat())
        save(output/'progress.json',run)
    print(json.dumps({k:v for k,v in run.items() if k not in ('postprocessing','source_response_bindings')}),flush=True)
    return 0 if run['status']=='complete' else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','candidates-json','prediction-receipt','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--study',action='append',required=True)
    parser.add_argument('--state-file')
    parser.add_argument('--preserve-four',action='store_true')
    parser.add_argument('--diagnostic-all-proposals',action='store_true')
    parser.add_argument('--ranking-study',action='store_true')
    parser.add_argument('--diversity-study',action='store_true')
    parser.add_argument('--preserved-ranking-study',action='store_true')
    raise SystemExit(replay(parser.parse_args()))
