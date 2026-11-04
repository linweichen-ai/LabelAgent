"""Compare reference-free person envelopes while preserving raw anchor coverage.

All original SAM candidates remain unchanged. New isolated exports are scored
separately; the tool never reads reference labels or calls model/workbench APIs.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json,math,os,re,sys
from pathlib import Path
import cv2
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.person_guard_envelope import person_envelope_mask
from tools.raw_coverage_selection import solve_raw_cover,point_person_guard
from tools.lidar_candidate_evidence import ReferenceReadAudit,read_xyz,sensor_cameras
from tools.spatial_candidate_evidence import SpatialPointNovelty,projected_spatial_sensor
from tools.tiled_candidate_utils import _mask_record,_iou
from tools.run_local_agent_acceptance import save
from source.agent.edit_agent import _rasterize

def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    dataset=Path(args.dataset).resolve();manifest=read(dataset/'import_manifest.json')
    paths={'manifest':dataset/'import_manifest.json'};receipts={};maps={}
    for name in ('baseline','pool','anchor','current'):
        path=Path(getattr(args,name+'_json')).resolve();receipt=Path(getattr(args,name+'_receipt')).resolve()
        value=read(receipt)
        if value['prediction_json_sha256']!=sha(path) or value['archive_sha256']!=manifest['archive_sha256']:
            raise ValueError('Candidate and dataset binding changed')
        if name!='baseline':
            parent=receipt.parent/'study_plan.json'
            if sha(parent)!=value['plan_sha256']:raise ValueError('Parent plan changed')
            paths[name+'_parent_plan']=parent
        paths[name]=path;paths[name+'_receipt']=receipt;receipts[name]=value;maps[name]=read(path)
    pairs=sorted([(topic,key) for topic,frames in maps['pool'].items() for key in frames],key=lambda p:(int(p[1]),p[0]))
    if not pairs or any(not re.fullmatch(r'/capture/camera_[0-3]/image',t) for t,_ in pairs):
        raise ValueError('Bound camera candidates required')
    frames=sorted({int(f) for _,f in pairs});cameras=sorted({int(t.split('/')[2].split('_')[-1]) for t,_ in pairs})
    if len(pairs)!=len(frames)*len(cameras) or any(f not in range(len(manifest['timestamps_ns'])) for f in frames):
        raise ValueError('Complete valid frame and camera cohort required')
    methods={'original_point_guard_control':'mask','convex_hull_guard':'convex_hull','bounding_box_guard':'bounding_box'}
    output=Path(args.output).resolve()
    if output.is_relative_to(dataset):raise ValueError('Keep outputs outside the original dataset')
    bindings={str(p):sha(p) for p in paths.values()}
    helper_names=('person_guard_envelope.py','raw_coverage_selection.py','spatial_candidate_evidence.py',
                  'lidar_candidate_evidence.py','tiled_candidate_utils.py','run_local_agent_acceptance.py')
    helpers={name:sha(REPO/'tools'/name) for name in helper_names}
    output.mkdir(parents=True,exist_ok=False)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'version':1,'methods':methods,
        'frames':frames,'cameras':cameras,'planned_images':len(pairs),'input_bindings':bindings,
        'code_sha256':sha(__file__),'helper_sha256':helpers,'metric_sha256':sha(REPO/'tools/evaluate_projected_support.py'),
        'minimum_score':.5,'addition_ratio':.5,'nms_iou':.8,'solver_seconds':args.solver_seconds,
        'raw_projection':{'cell_size':8,'depth_slack':.15,'near':.2,'dilation_pixels':3,'kernel':'ellipse'},
        'person_envelopes':'Per original person instance before union; no padding or confidence changes',
        'guard':'No newly covered raw points in person envelope beyond old four-crop anchor',
        'raw_coverage_certificate':'All old four-crop raw coverage retained; original candidate prefix immutable',
        'prediction_uses_reference_labels':False,'model_calls':0,'workbench_writes':False,
        'tuning_success_requires':{'all_original_eight_gates':True,'instances_at_least':78,
            'points_at_least':5183,'person_cross_at_most':55,'strict_change_required':True},
        'default_enabled':False,'automatic_full_expansion':False,'scope':'same-capture tuning, not pixel accuracy'}
    save(output/'study_plan.json',plan);plan_sha=sha(output/'study_plan.json')
    state={'status':'running','pid':os.getpid(),'completed_images':0,'rows':[],
           'new_model_requests':0,'workbench_writes':False}
    save(output/'progress.json',state)
    outputs={name:{} for name in methods};image_hashes={};audit=ReferenceReadAudit(dataset);audit.start()
    try:
        camera_info,_=sensor_cameras(dataset,manifest,cameras);clouds={}
        records={(r['topic'],str(r['frame'])):r for r in manifest['records'] if r['topic'].endswith('/image')}
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        for topic,key in pairs:
            if (output/'STOP').exists():raise InterruptedError('Selection stop requested')
            record=records[topic,key]
            if any(r['image_sha256'][topic][key]!=record['sha256'] for r in receipts.values()):
                raise ValueError('Source image binding changed')
            seed=maps['baseline'][topic][key];pool=maps['pool'][topic][key];anchor_shapes=maps['anchor'][topic][key]
            current=maps['current'][topic][key]
            if any(value[:len(seed)]!=seed for value in (pool,anchor_shapes,current)):
                raise ValueError('Original candidate prefix changed')
            if len({s['id'] for s in pool})!=len(pool):raise ValueError('Duplicate candidate IDs')
            by_id={s['id']:s for s in pool};proposals=pool[len(seed):]
            if any(s['label']!='pallet' for s in proposals):raise ValueError('Pallet-only additions required')
            if any(by_id[s['id']]!=s for s in anchor_shapes+current):raise ValueError('Anchor/current not in bound pool')
            if key not in clouds:clouds[key]=read_xyz(dataset,manifest,int(key))[0]
            camera=camera_info[int(topic.split('/')[2].split('_')[-1])];width,height=camera['width'],camera['height']
            sensor=projected_spatial_sensor(clouds[key],camera);pixels=sensor['pixels']
            context=SpatialPointNovelty(seed,proposals,width,height,sensor)
            masks={};raster_records={}
            for shape in pool:
                if shape['label']=='pallet':
                    mask=_rasterize(shape,width,height);d=cv2.dilate(mask,kernel)
                    masks[shape['id']]=set(np.flatnonzero(d[pixels[:,1],pixels[:,0]]).tolist())
                    raster_records[shape['id']]=_mask_record(shape,width,height)
            original=[s for s in seed if s['label']=='pallet']
            base=set().union(*(masks[s['id']] for s in original))
            anchor=set().union(*(masks[s['id']] for s in anchor_shapes if s['label']=='pallet'))
            budget=max(1,math.floor(len(original)*.5));previous_guard=None
            for name,mode in methods.items():
                envelope=person_envelope_mask(seed,width,height,mode)
                dilated=cv2.dilate(envelope,kernel)
                person_points=set(np.flatnonzero(dilated[pixels[:,1],pixels[:,0]]).tolist())
                if previous_guard is not None and not previous_guard<=person_points:
                    raise ValueError('Person envelope does not contain the preceding region')
                previous_guard=person_points
                eligible=[s for s in proposals if s.get('score',0)>=.5 and
                    not any(_iou(raster_records[s['id']],raster_records[o['id']])>=.8 for o in original) and
                    point_person_guard(masks[s['id']],person_points,anchor)]
                allowed={s['id'] for s in eligible}
                if any(s['id'] not in allowed for s in anchor_shapes[len(seed):]):raise ValueError('Old anchor witness excluded')
                conflicts=[(i,j) for i,a in enumerate(eligible) for j,b in enumerate(eligible)
                           if i<j and _iou(raster_records[a['id']],raster_records[b['id']])>=.8]
                scores=[context.priority(s,tuple(seed),'component_mass') for s in eligible]
                solved=solve_raw_cover([masks[s['id']] for s in eligible],anchor-base,scores,conflicts,budget,args.solver_seconds)
                if not solved['summary']['certificate_satisfied']:raise ValueError('No verified raw coverage certificate')
                selected=[eligible[i] for i in solved['selected_indices']]
                covered=base|set().union(*(masks[s['id']] for s in selected))
                if not anchor<=covered or any(not point_person_guard(masks[s['id']],person_points,anchor) for s in selected):
                    raise ValueError('Actual coverage or person guard failed')
                result=deepcopy(seed+selected)
                if name=='original_point_guard_control' and result!=current:raise ValueError('Original guard control drifted')
                outputs[name].setdefault(topic,{})[key]=result
                state['rows'].append({'frame':int(key),'topic':topic,'method':name,'solver':solved['summary'],
                    'original_prefix_preserved':result[:len(seed)]==seed,'raw_certificate_verified':True,
                    'person_guard_verified':True,'guard_raw_points':len(person_points),'eligible_additions':len(eligible),
                    'selected_additions':len(selected),'candidate_membership_changed':result!=current})
            image_hashes.setdefault(topic,{})[key]=record['sha256'];state['completed_images']+=1
            save(output/'progress.json',state)
            print(json.dumps({'images':state['completed_images'],'planned':len(pairs)}),flush=True)
        if any(sha(p)!=value for p,value in bindings.items()):raise ValueError('Input files changed')
        for name,mapping in outputs.items():
            path=output/(name+'_candidates.json');save(path,mapping)
            save(output/(name+'_receipt.json'),{'created_utc':datetime.now(timezone.utc).isoformat(),
                'archive_sha256':manifest['archive_sha256'],'prediction_json_sha256':sha(path),
                'image_sha256':image_hashes,'plan_sha256':plan_sha,'method':name,
                'new_model_requests':0,'candidates_confirmed_as_truth':False,'source':'original SAM predictions and label-free raw coverage'})
        state.update(status='complete',all_predictions_frozen_before_scoring=True)
    except InterruptedError:
        state['status']='stopped'
    except Exception as error:
        state.update(status='failed',error_type=type(error).__name__)
        (output/'local_error.txt').write_text(str(error),encoding='utf-8')
    finally:
        audit.stop();state.update(reference_files_opened=len(audit.denied),
            input_bindings_unchanged=all(sha(p)==value for p,value in bindings.items()),
            plan_unchanged=sha(output/'study_plan.json')==plan_sha,
            code_unchanged=sha(__file__)==plan['code_sha256'] and all(sha(REPO/'tools'/n)==h for n,h in helpers.items()))
        if audit.denied:state['status']='failed'
        save(output/'progress.json',state)
    print(json.dumps({k:v for k,v in state.items() if k!='rows'}),flush=True)
    return 0 if state['status']=='complete' else 2 if state['status']=='stopped' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','baseline-json','baseline-receipt','pool-json','pool-receipt','anchor-json','anchor-receipt',
                 'current-json','current-receipt','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--solver-seconds',type=float,default=10.)
    raise SystemExit(run(parser.parse_args()))
