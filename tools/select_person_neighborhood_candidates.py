"""Keep old instances and protect one 15cm hop around predicted person points.

Predicted-person protection is not a guarantee about reference labels. Actual
person cross coverage must still pass the unchanged metric and all radii.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json,os,re,sys
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.global_anchor_additions import point_hits,allocate,union_pixels
from tools.multiview_person_guard import merge_point_views,rejects_novel_points,ranking_inputs
from tools.person_seed_neighborhood import expand_person_seeds
from tools.evaluate_projected_support import project_visible
from tools.lidar_candidate_evidence import ReferenceReadAudit,read_xyz,sensor_cameras
from tools.spatial_candidate_evidence import SpatialPointNovelty,projected_spatial_sensor
from tools.tiled_candidate_utils import _mask_record,_iou
from tools.run_local_agent_acceptance import save

def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    ranking_anchor=getattr(args,'ranking_anchor','baseline')
    if ranking_anchor not in ('baseline','four'):raise ValueError('Unknown ranking context')
    minimum_extra=getattr(args,'minimum_extra_raw3_points',1)
    if type(minimum_extra) is not int or minimum_extra not in (1,3):
        raise ValueError('Minimum total raw3px points must be 1 (original) or 3 (bounded study)')
    dataset=Path(args.dataset).resolve();manifest=read(dataset/'import_manifest.json');paths=[dataset/'import_manifest.json']
    def bound(path,receipt_path,parent=False):
        path,receipt_path=Path(path).resolve(),Path(receipt_path).resolve();receipt=read(receipt_path)
        if sha(path)!=receipt['prediction_json_sha256'] or receipt['archive_sha256']!=manifest['archive_sha256']:
            raise ValueError('Prediction binding changed')
        paths.extend([path,receipt_path])
        if parent:
            plan=receipt_path.parent/'study_plan.json'
            if sha(plan)!=receipt['plan_sha256']:raise ValueError('Parent plan changed')
            paths.append(plan)
        return read(path),receipt
    baseline,br=bound(args.baseline_json,args.baseline_receipt)
    anchors,ar=bound(args.anchor_json,args.anchor_receipt,True)
    if len(args.pool_json)!=len(args.pool_receipt):raise ValueError('Each pool needs its receipt')
    pool={};pool_receipts={}
    for file,receipt in zip(args.pool_json,args.pool_receipt):
        values,meta=bound(file,receipt,True)
        for topic,frames in values.items():
            for key,shapes in frames.items():
                if key in pool.get(topic,{}):raise ValueError('Overlapping source pools')
                pool.setdefault(topic,{})[key]=shapes;pool_receipts[topic,key]=meta
    pairs=sorted([(t,f) for t,frames in pool.items() for f in frames],key=lambda p:(int(p[1]),p[0]))
    if not pairs or any(not re.fullmatch(r'/capture/camera_[0-3]/image',t) for t,_ in pairs):raise ValueError('Bound image cohort required')
    frames=sorted({int(f) for _,f in pairs});cameras=sorted({int(t.split('/')[2].split('_')[-1]) for t,_ in pairs})
    if len(pairs)!=len(frames)*len(cameras):raise ValueError('Complete rectangular cohort required')
    if cameras!=[0,1,2,3]:raise ValueError('All four camera views required')
    output=Path(args.output).resolve()
    if output.is_relative_to(dataset):raise ValueError('Isolate predictions from source dataset')
    output.mkdir(parents=True,exist_ok=False);bindings={str(p):sha(p) for p in paths}
    helpers=('person_seed_neighborhood.py','global_anchor_additions.py','multiview_person_guard.py','evaluate_projected_support.py','raw_coverage_selection.py','spatial_candidate_evidence.py','lidar_candidate_evidence.py','tiled_candidate_utils.py','run_local_agent_acceptance.py')
    helper_hashes={name:sha(REPO/'tools'/name) for name in helpers}
    methods={'four_crop_control':'unchanged','raw_subset_control':'subset','cross_view_person_control':'cross','person_neighborhood_veto':'neighborhood'}
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'primary_method':'person_neighborhood_veto',
        'secondary_can_replace_primary':False,'methods':methods,'frames':frames,'cameras':cameras,'planned_images':len(pairs),
        'input_bindings':bindings,'code_sha256':sha(__file__),'helper_sha256':helper_hashes,
        'metric_sha256':sha(REPO/'tools/evaluate_projected_support.py'),
        'fixed_parameters':{'minimum_score':.5,'nms_iou':.8,'global_candidate_count_ratio':1.5,'global_union_area_ratio':1.25,
            'minimum_extra_raw3_points':minimum_extra,'predicted_person_dilation_pixels':6,'person_neighborhood_meters':.15,'neighborhood_growth':'one nonrecursive Euclidean hop from original seeds','ranking_anchor':ranking_anchor},
        'local_candidate_budget':'none; distribute unused slots under unchanged original global caps',
        'ranking':f'{ranking_anchor}-relative fixed spatial component-mass priority; deterministic global order',
        'anchor_contract':'Every old four-crop shape stays as the literal prefix, including zero-point masks',
        'primary_raw_contract':'Old raw coverage retained; new raw points at0/3/6 avoid a 15cm one-hop neighbourhood of any-view predicted-person points at6px',
        'person_seed_source':'original baseline model person masks in all four views; no reference labels',
        'extra_eligibility':f'at least {minimum_extra} total raw covered points at3px; no novel-point minimum; anchors exempt',
        'prediction_reads_reference':False,'approved_reference_inputs':False,'model_calls':0,'workbench_writes':False,
        'scope':'new bounded exploratory study, not independent accuracy or default adoption'}
    save(output/'study_plan.json',plan);plan_sha=sha(output/'study_plan.json')
    progress={'status':'preparing','pid':os.getpid(),'prepared_images':0,'model_calls':0,'workbench_writes':False}
    save(output/'progress.json',progress);audit=ReferenceReadAudit(dataset);audit.start()
    active={'value':True};denied=[];forbidden=[Path(p).resolve() for p in args.deny_reference_root]
    def guard(event,values):
        if event=='open' and active['value'] and values and isinstance(values[0],(str,bytes)):
            path=Path(values[0].decode() if isinstance(values[0],bytes) else values[0]).resolve()
            if any(path.is_relative_to(p) for p in forbidden):denied.append(str(path));raise PermissionError('Approved-reference input forbidden')
    sys.addaudithook(guard)
    try:
        info,_=sensor_cameras(dataset,manifest,cameras)
        records={(r['topic'],str(r['frame'])):r for r in manifest['records'] if r['topic'].endswith('/image')}
        old_records={};sensors={};anchor_hits={};candidates=[];base_count=0;base_area=0;images={};last_frame=None;xyz=None
        frame_sensors={};evidence_by_frame={};raw_indices={}
        for topic,key in pairs:
            if (output/'STOP').exists():raise InterruptedError('Study stop requested')
            source=records[topic,key];pr=pool_receipts[topic,key]
            if any(r['image_sha256'][topic][key]!=source['sha256'] for r in (br,ar,pr)):raise ValueError('Image binding changed')
            seed=baseline[topic][key];old=anchors[topic][key];values=pool[topic][key]
            if old[:len(seed)]!=seed or values[:len(seed)]!=seed:raise ValueError('Original prefix changed')
            by_id={s['id']:s for s in values}
            if len(by_id)!=len(values) or any(by_id[s['id']]!=s for s in old):raise ValueError('Bound anchor geometry differs')
            camera=info[int(topic.split('/')[2].split('_')[-1])];w,h=camera['width'],camera['height']
            if key!=last_frame:
                xyz,_=read_xyz(dataset,manifest,int(key));last_frame=key;views={};frame_sensors={}
                for other in cameras:
                    other_topic=f'/capture/camera_{other}/image';other_info=info[other];ow,oh=other_info['width'],other_info['height']
                    source_indices,p=project_visible(xyz,other_info);person=np.zeros(len(source_indices),bool)
                    for shape in baseline[other_topic][key]:
                        if shape['label']=='person':person|=point_hits(_mask_record(shape,ow,oh),p,6,ow,oh)
                    views[other]={'indices':source_indices,'person_hits':person};frame_sensors[other]=(source_indices,p)
                evidence_by_frame[key]=merge_point_views(len(xyz),views)
                evidence_by_frame[key]['person_neighborhood']=expand_person_seeds(xyz,evidence_by_frame[key]['person_any'],.15)
            indices,pixels=frame_sensors[int(topic.split('/')[2].split('_')[-1])]
            visible_xyz=xyz[indices]
            sensor={'pixels':pixels,'xyz':visible_xyz,'depths':visible_xyz@camera['transform'][2,:3]+camera['transform'][2,3]}
            image_key=(topic,key);raw_indices[image_key]=indices;evidence=evidence_by_frame[key]
            sensors[image_key]=(pixels,w,h)
            props=values[len(seed):]
            if any(s['label']!='pallet' for s in props):raise ValueError('Pallet-only additions required')
            rank_seed,rank_props=ranking_inputs(seed,old,props,ranking_anchor)
            context=SpatialPointNovelty(rank_seed,rank_props,w,h,sensor)
            masks={s['id']:_mask_record(s,w,h) for s in values if s['label']=='pallet'}
            old_records[image_key]=[masks[s['id']] for s in old if s['label']=='pallet']
            original=[masks[s['id']] for s in seed if s['label']=='pallet'];base_count+=len(original);base_area+=union_pixels(original)
            anchor_hits[image_key]={radius:np.zeros(len(pixels),bool) for radius in (0,3,6)}
            for radius in (0,3,6):
                for record in old_records[image_key]:anchor_hits[image_key][radius]|=point_hits(record,pixels,radius,w,h)
            old_ids={s['id'] for s in old}
            for shape in props:
                if shape['id'] in old_ids or shape.get('score',0)<.5:continue
                record=masks[shape['id']]
                if any(_iou(record,r)>=.8 for r in old_records[image_key]):continue
                hits={r:point_hits(record,pixels,r,w,h) for r in (0,3,6)}
                total_raw3=int(np.count_nonzero(hits[3]))
                if total_raw3<minimum_extra:continue
                own=evidence['per_view'][int(topic.split('/')[2].split('_')[-1])]
                allowed={'subset':all(not np.any(hits[r]&~anchor_hits[image_key][r]) for r in (0,3,6)),
                    'own':not any(rejects_novel_points(hits[r],anchor_hits[image_key][r],indices,own) for r in (0,3,6)),
                    'cross':not any(rejects_novel_points(hits[r],anchor_hits[image_key][r],indices,evidence['person_any']) for r in (0,3,6)),
                    'neighborhood':not any(rejects_novel_points(hits[r],anchor_hits[image_key][r],indices,evidence['person_neighborhood']) for r in (0,3,6))}
                candidates.append({'key':image_key,'shape':shape,'record':record,
                    'priority':float(context.priority(shape,tuple(rank_seed),'component_mass')),
                    'total_raw3_points':total_raw3,
                    'allowed':allowed})
            images.setdefault(topic,{})[key]=source['sha256'];progress['prepared_images']+=1
            save(output/'progress.json',progress)
        cap_count=base_count*3//2;cap_area=base_area*5//4;outputs={'four_crop_control':{t:{f:deepcopy(anchors[t][f]) for tt,f in pairs if tt==t} for t in cameras_as_topics(pairs)}}
        summaries={};readbacks=[]
        for name,mode in methods.items():
            if name=='four_crop_control':continue
            allowed_candidates=[{**c,'allowed':{'all':c['allowed'][mode]}} for c in candidates]
            selected,summary=allocate(old_records,allowed_candidates,cap_count,cap_area,'all')
            mapping={t:{f:deepcopy(anchors[t][f]) for tt,f in pairs if tt==t} for t in cameras_as_topics(pairs)}
            for item in selected:mapping[item['key'][0]][item['key'][1]].append(deepcopy(item['shape']))
            for topic,key in pairs:
                old=anchors[topic][key];shapes=mapping[topic][key];pixels,w,h=sensors[topic,key]
                if shapes[:len(old)]!=old:raise ValueError('Old instance prefix changed')
                exact={};nonreg={};novel_person={}
                for radius in (0,3,6):
                    covered=anchor_hits[topic,key][radius].copy()
                    for shape in shapes[len(old):]:covered|=point_hits(_mask_record(shape,w,h),pixels,radius,w,h)
                    exact[str(radius)]=bool(np.array_equal(covered,anchor_hits[topic,key][radius]))
                    nonreg[str(radius)]=not bool(np.any(anchor_hits[topic,key][radius]&~covered))
                    person=evidence_by_frame[key]['person_neighborhood'] if mode=='neighborhood' else evidence_by_frame[key]['person_any']
                    novel_person[str(radius)]=int(np.count_nonzero((covered&~anchor_hits[topic,key][radius])&person[raw_indices[topic,key]]))
                if not all(nonreg.values()) or mode=='subset' and not all(exact.values()) or mode!='subset' and any(novel_person.values()):
                    raise ValueError('Actual raw point protection differs')
                readbacks.append({'topic':topic,'frame':int(key),'method':name,'old_four_prefix_exact':True,'raw_union_equal':exact,
                    'raw_coverage_nonregressing':nonreg,'new_predicted_person_points':novel_person})
            outputs[name]=mapping;summaries[name]=summary
            summary['selected_extra_raw3_below3']=sum(c['total_raw3_points']<3 for c in selected)
        if any(sha(p)!=value for p,value in bindings.items()):raise ValueError('Input changed')
        for name,mapping in outputs.items():
            file=output/(name+'_candidates.json');save(file,mapping)
            save(output/(name+'_receipt.json'),{'created_utc':datetime.now(timezone.utc).isoformat(),
                'archive_sha256':manifest['archive_sha256'],'prediction_json_sha256':sha(file),'image_sha256':images,
                'plan_sha256':plan_sha,'method':name,'candidates_confirmed_as_truth':False,'new_model_requests':0,
                'source':'reference_free_3d_person_neighborhood_study'})
        progress.update(status='complete',completed_images=len(pairs),allocation=summaries,readbacks=readbacks,
            globally_eligible_candidates=len(candidates),baseline_pallet_candidates=base_count,baseline_union_pixels=base_area,
            eligible_extra_raw3_below3=sum(c['total_raw3_points']<3 for c in candidates),
            minimum_extra_raw3_points=minimum_extra,point_gate_applies_to_anchors=False,
            all_predictions_frozen_before_scoring=True,plan_sha256=plan_sha)
    except InterruptedError:progress['status']='stopped'
    except Exception as error:
        progress.update(status='failed',error_type=type(error).__name__);(output/'local_error.txt').write_text(str(error),encoding='utf-8')
    finally:
        audit.stop();active['value']=False
        progress.update(reference_files_opened=len(audit.denied),approved_reference_files_opened=len(denied),
            input_bindings_unchanged=all(sha(p)==value for p,value in bindings.items()),
            code_unchanged=sha(__file__)==plan['code_sha256'] and all(sha(REPO/'tools'/n)==h for n,h in helper_hashes.items()),
            plan_unchanged=sha(output/'study_plan.json')==plan_sha)
        save(output/'progress.json',progress)
    print(json.dumps({k:v for k,v in progress.items() if k not in ('readbacks',)}),flush=True)
    return 0 if progress['status']=='complete' else 1


def cameras_as_topics(pairs):return sorted({t for t,_ in pairs})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','baseline-json','baseline-receipt','anchor-json','anchor-receipt','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--pool-json',action='append',required=True);p.add_argument('--pool-receipt',action='append',required=True)
    p.add_argument('--deny-reference-root',action='append',default=[])
    p.add_argument('--minimum-extra-raw3-points',type=int,choices=(1,3),default=1)
    p.add_argument('--ranking-anchor',choices=('baseline','four'),default='baseline')
    raise SystemExit(run(p.parse_args()))
