"""Reuse bound real SAM crop responses for image-only multiscale proposals.

Reads no 3D reference labels, workbench API, or model API.
Explicit sensor study options add unlabelled raw data; --lidar-spatial-study uses local 3D components.
all existing image-only modes retain their original behavior.
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

def select_replay_frames(plans,selection=None):
    if len(plans)!=2 or plans[0]['cameras']!=plans[1]['cameras']:
        raise ValueError('Two source plans with identical cameras required')
    if selection is None:
        if plans[0]['frames']!=plans[1]['frames']:
            raise ValueError('Different source frame cohorts require an explicit shared subset')
        return list(plans[0]['frames'])
    chosen=[int(value) for value in selection.split(',') if value.strip()]
    if not chosen or len(chosen)!=len(set(chosen)):
        raise ValueError('Nonempty distinct explicit frames required')
    if any(not set(chosen)<=set(plan['frames']) for plan in plans):
        raise ValueError('Every explicit frame must occur in both complete source studies')
    return sorted(chosen)


def replay(args):
    novelty_study=bool(getattr(args,'lidar_novelty_study',False))
    point_only_study=bool(getattr(args,'lidar_point_only_study',False))
    spatial_study=bool(getattr(args,'lidar_spatial_study',False))
    dual_study=bool(getattr(args,'lidar_dual_study',False))
    uses_spatial=spatial_study or dual_study
    point_study=novelty_study or point_only_study or uses_spatial
    uses_lidar=args.lidar_study or point_study
    if sum((args.preserve_four,args.diagnostic_all_proposals,args.ranking_study,args.diversity_study,args.preserved_ranking_study,args.containment_study,args.lidar_study,novelty_study,point_only_study,spatial_study,dual_study))>1:
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
    selected_frames=select_replay_frames(plans,getattr(args,'frames',None))
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
    if args.containment_study:
        methods={'four_crop_control':methods['four_crop_control'],
            'multiscale_novelty_control':{'pool':'both','maximum_person_overlap':None},
            **{('dynamic' if dynamic else 'static')+'_containment_rank'+('_person_guard' if guard else ''):
                {'pool':'both','maximum_person_overlap':0. if guard else None,
                 'soft_containment':True,'dynamic_diversity':dynamic}
                for dynamic in (False,True) for guard in (False,True)}}
    if uses_lidar:
        from lidar_candidate_evidence import (PLANE_RECIPE,POINT_RECIPE,ReferenceReadAudit,
            read_xyz,fit_support_plane,sensor_cameras,projected_sensor,candidate_evidence,height_priorities,RawPointNovelty,POINT_NOVELTY_RECIPE)
        methods={'four_crop_control':methods['four_crop_control'],
            'multiscale_novelty_control':{'pool':'both','maximum_person_overlap':None},
            'lidar_positive_guard':{'pool':'both','maximum_person_overlap':0.,'height_priority':'positive'},
            'lidar_penalty_guard':{'pool':'both','maximum_person_overlap':0.,'height_priority':'penalty'},
            'preserve_four_lidar_penalty_guard':{'pool':'both','maximum_person_overlap':0.,'height_priority':'penalty','preserve_four':True}}
    if novelty_study:
        methods={'four_crop_control':methods['four_crop_control'],
            'multiscale_novelty_control':{'pool':'both','maximum_person_overlap':None},
            'height_positive_guard_control':{'pool':'both','maximum_person_overlap':0.,'height_priority':'positive'},
            **{'raw_point_'+mode+'_guard':{'pool':'both','maximum_person_overlap':0.,'point_policy':mode}
                for mode in ('mass','fraction','dynamic')}}
    if point_only_study:
        methods={'four_crop_control':methods['four_crop_control'],
            'multiscale_novelty_control':{'pool':'both','maximum_person_overlap':None},
            'hybrid_fraction_control':{'pool':'both','maximum_person_overlap':0.,'point_policy':'fraction'},
            'hybrid_dynamic_control':{'pool':'both','maximum_person_overlap':0.,'point_policy':'dynamic'},
            **{'point_only_'+mode+'_guard':{'pool':'both','maximum_person_overlap':0.,'point_policy':mode,'point_base':'point_only'}
                for mode in ('fraction','dynamic')}}
    if uses_spatial:
        from spatial_candidate_evidence import SpatialPointNovelty,projected_spatial_sensor,SPATIAL_RECIPE
        methods={'four_crop_control':methods['four_crop_control'],
            'multiscale_novelty_control':{'pool':'both','maximum_person_overlap':None},
            'hybrid_dynamic_control':{'pool':'both','maximum_person_overlap':0.,'point_policy':'dynamic'},
            **{'spatial_'+policy+'_guard':{'pool':'both','maximum_person_overlap':0.,'spatial_policy':policy}
                for policy in ('all_dynamic','component_mass','component_fraction')}}
    if dual_study:
        methods={'four_crop_control':methods['four_crop_control'],
            'multiscale_novelty_control':{'pool':'both','maximum_person_overlap':None},
            **{'spatial_component_'+mode+'_control':{'pool':'both','maximum_person_overlap':0.,'spatial_policy':'component_'+mode}
                for mode in ('mass','fraction')},
            **{'spatial_dual_'+mode+'_guard':{'pool':'both','maximum_person_overlap':0.,'spatial_policy':'dual_'+mode}
                for mode in ('mass','fraction')}}
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'version':1,
        'frames':selected_frames,'cameras':plans[0]['cameras'],
        'planned_images':len(selected_frames)*len(plans[0]['cameras']),
        'new_model_requests':0,'historical_real_SAM3_responses_reused':len(selected_frames)*len(plans[0]['cameras'])*sum(g['divisions']**2 for g in grids),
        'parent_plan_sha256':[p['plan_sha256'] for p in progress],
        'methods':methods,'diagnostic_only':args.diagnostic_all_proposals,
        'priority_seed':'previous_four_crop_selection' if args.preserved_ranking_study else 'original_candidates',
        'ranking_specifications':{'confidence':'original SAM3 confidence',
            'consensus':'confidence*(0.5+0.5*max_mask_IoU_to_the_other_crop_scale)',
            'novel_consensus':'confidence*(1-max_mask_IoU_to_original_pallets)*(0.5+0.5*cross_scale_IoU)',
            'dynamic_diversity':'recompute confidence*(1-max_mask_IoU_to_original_and_accepted_pallets) after each addition',
            'soft_containment':'multiply original novelty priority by 0.5+0.5*fraction_outside_pallet_union; original model score retained',
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
    if uses_lidar:
        sensor_paths=['calib/lidar2camera.yaml',*[f'calib/camera{c}_calib.yaml' for c in plan['cameras']],
            *[f"lidar/{manifest['timestamps_ns'][f]}.pcd" for f in plan['frames']]]
        plan.update(plane_recipe=PLANE_RECIPE,point_recipe=POINT_RECIPE,
            sensor_helper_sha256=digest(Path(__file__).with_name('lidar_candidate_evidence.py')),
            raw_sensor_source_sha256={p:manifest['extracted_sha256'][p] for p in sensor_paths},
            source_units='nominal metres from supplied calibration; not independently certified',
            height_rank={'positive':'novelty*confidence*(1+low_object_fraction)/2; uniform scale preserves ordering',
                'penalty':'novelty*confidence*(1-0.75*floor_fraction)*(1-0.75*tall_fraction)',
                'unsupported':'neutral evidence factor 1 before uniform positive-mode scaling; never reject because points are missing',
                'preserved_seed':'previous four-crop candidates for the residual-budget variant'},
            reference_file_read_boundary_enforced=True)
    if novelty_study:
        plan.update(point_novelty_recipe=POINT_NOVELTY_RECIPE,
            point_priority={'base':'original confidence*(1-max_mask_IoU_to_original_pallets)',
                'mass':'base*(0.5+0.5*novel_low_points/(5+novel_low_points))',
                'fraction':'base*(0.5+0.5*novel_low_points/(5+all_low_points))',
                'dynamic':'mass rule, updating covered point indices after each actually selected candidate',
                'unsupported':'factor0.5; preserve neutral ordering and all original candidates'},
            fixed_prefix='original SAM baseline only, not previously selected four-crop additions')
    if point_only_study:
        plan.update(point_novelty_recipe=POINT_NOVELTY_RECIPE,
            point_priority={'hybrid_control_base':'original confidence*(1-max_mask_IoU_to_original_pallets)',
                'point_only_base':'original SAM3 confidence, without pixel-overlap priority multiplier',
                'fraction':'base*(0.5+0.5*novel_low_points/(5+all_low_points))',
                'dynamic':'base*(0.5+0.5*novel_low_points/(5+novel_low_points)), update coverage after actual acceptance',
                'unsupported':'factor0.5 on chosen base; never hard-reject missing points'},
            fixed_prefix='original SAM baseline only; original masks/scores and all adoption gates retained',
            ablation='only remove pixel-IoU priority multiplier; NMS at0.8, budget0.5 and predicted-person protection unchanged')
    if uses_spatial:
        plan.update(spatial_recipe=SPATIAL_RECIPE,
            spatial_helper_sha256=digest(Path(__file__).with_name('spatial_candidate_evidence.py')),
            spatial_priority={'all_dynamic':'base*(0.5+0.5*novel_all_points/(5+novel_all_points))',
                'component_mass':'base*(0.5+0.5*coherence*novel_foreground_points/(5+novel_foreground_points))',
                'component_fraction':'base*(0.5+0.5*coherence*novel_foreground_points/(5+foreground_points))',
                'all_methods':'dynamic coverage after actual acceptance; no height/floor filtering; missing support factor0.5'},
            fixed_prefix='original SAM baseline only; scores/geometry/all adoption gates unchanged')
    if dual_study:
        plan['spatial_priority'].update(dual_mass='0.5*component_mass_priority+0.5*all_dynamic_priority',
            dual_fraction='0.5*component_fraction_priority+0.5*all_dynamic_priority',
            fixed_equal_weight=.5,selection='both coverage states update after actual acceptance; scores/gates unchanged')
    save(output/'study_plan.json',plan)
    plan_hash=digest(output/'study_plan.json')
    before=digest(Path(args.state_file)) if args.state_file else None
    run={'status':'running','pid':os.getpid(),'completed_images':0,'new_model_requests':0,
        'verified_historical_responses':0,'postprocessing':[],'source_response_bindings':[],'workbench_modified':False}
    save(output/'progress.json',run)
    maps={method:{} for method in methods}
    inputs={}
    sensor_audit=ReferenceReadAudit(dataset) if uses_lidar else None
    if sensor_audit:sensor_audit.start()
    try:
        if uses_lidar:
            sensor_info,_=sensor_cameras(dataset,manifest,plan['cameras'])
            sensor_cache={}
            run['lidar_evidence']=[]
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
            if uses_lidar:
                if row['frame'] not in sensor_cache:
                    xyz,_=read_xyz(dataset,manifest,row['frame'])
                    sensor_cache[row['frame']]=(xyz,fit_support_plane(xyz))
                xyz,plane=sensor_cache[row['frame']]
                sensor=projected_sensor(xyz,sensor_info[int(topic.split('/')[2].split('_')[-1])],plane)
                evidence={s['id']:candidate_evidence(s,width,height,sensor) for s in pools[0]+pools[1]}
                save(output/f"sensor_{topic.split('/')[2]}_{int(frame):04d}.json",{'plane':plane,'candidates':evidence})
                run['lidar_evidence'].append({'frame':row['frame'],'topic':topic,'plane':plane,
                    'projected_raw_points':len(sensor['pixels']),'proposals':len(evidence),
                    'proposals_with_height_evidence':sum(v['supported'] for v in evidence.values())})
            if point_study:
                point_contexts={mode:RawPointNovelty(seed,pools[0]+pools[1],width,height,sensor,base_mode=mode)
                    for mode in ({'hybrid','point_only'} if point_only_study else {'hybrid'})}
                run['lidar_evidence'][-1]['point_novelty']={mode:context.summary() for mode,context in point_contexts.items()}
            if uses_spatial:
                spatial_sensor=projected_spatial_sensor(xyz,sensor_info[int(topic.split('/')[2].split('_')[-1])])
                spatial_context=SpatialPointNovelty(seed,pools[0]+pools[1],width,height,spatial_sensor)
                run['lidar_evidence'][-1]['spatial']=spatial_context.summary()
            inputs.setdefault(topic,{})[frame]=row['sha256']
            for method,settings in methods.items():
                proposals=pools[0] if settings['pool']=='four' else pools[1] if settings['pool']=='dense' else pools[0]+pools[1]
                priority=priorities.get(settings.get('ranking'))
                if priority is not None:priority={shape['id']:priority[shape['id']] for shape in proposals}
                if 'height_priority' in settings:
                    priority=height_priorities(selected_four if settings.get('preserve_four') else seed,
                        proposals,width,height,evidence,settings['height_priority'])
                callback=None
                if 'point_policy' in settings:
                    point_context=point_contexts[settings.get('point_base','hybrid')]
                    if settings['point_policy']=='dynamic':
                        priority=None;callback=point_context.dynamic_priority
                    else:priority=point_context.priorities(settings['point_policy'])
                if 'spatial_policy' in settings:
                    priority=None;callback=spatial_context.callback(settings['spatial_policy'])
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
                        dynamic_diversity=settings.get('dynamic_diversity',False),
                        soft_containment=settings.get('soft_containment',False),priority_callback=callback)
                if 'height_priority' in settings:
                    info['ranking_source']='raw_LiDAR_height_and_original_predictions; model scores retained'
                if 'point_policy' in settings:
                    info['ranking_source']='raw_LiDAR_novel_low_point_support; model scores retained'
                if 'spatial_policy' in settings:
                    info['ranking_source']='unlabelled_full_height_spatial_point_support; original scores retained'
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
        if uses_lidar:
            from tools.evaluate_projected_support import checked_source
            try:
                for p in plan['raw_sensor_source_sha256']:checked_source(dataset,manifest,p)
                run['raw_sensor_sources_unchanged']=True
            except Exception:run['raw_sensor_sources_unchanged']=False
            run['sensor_helper_unchanged']=digest(Path(__file__).with_name('lidar_candidate_evidence.py'))==plan['sensor_helper_sha256']
            if uses_spatial:run['spatial_helper_unchanged']=digest(Path(__file__).with_name('spatial_candidate_evidence.py'))==plan['spatial_helper_sha256']
            sensor_audit.stop()
            run['reference_files_opened']=len(sensor_audit.denied)
            run['raw_source_files_read']=sorted(sensor_audit.sources)
            if sensor_audit.denied:run['status']='failed'
        save(output/'progress.json',run)
    print(json.dumps({k:v for k,v in run.items() if k not in ('postprocessing','source_response_bindings','lidar_evidence','raw_source_files_read')}),flush=True)
    return 0 if run['status']=='complete' else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','candidates-json','prediction-receipt','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--study',action='append',required=True)
    parser.add_argument('--frames',help='Explicit shared frame subset; source plans and response receipts remain unchanged')
    parser.add_argument('--state-file')
    parser.add_argument('--preserve-four',action='store_true')
    parser.add_argument('--diagnostic-all-proposals',action='store_true')
    parser.add_argument('--ranking-study',action='store_true')
    parser.add_argument('--diversity-study',action='store_true')
    parser.add_argument('--preserved-ranking-study',action='store_true')
    parser.add_argument('--containment-study',action='store_true')
    parser.add_argument('--lidar-study',action='store_true')
    parser.add_argument('--lidar-novelty-study',action='store_true')
    parser.add_argument('--lidar-point-only-study',action='store_true')
    parser.add_argument('--lidar-spatial-study',action='store_true')
    parser.add_argument('--lidar-dual-study',action='store_true')
    raise SystemExit(replay(parser.parse_args()))
