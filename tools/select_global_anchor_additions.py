"""Keep all four-crop instances and allocate extra masks under original global caps.

The primary adds no new raw covered points at 0/3/6px. A 3px-only control is
reported separately. No reference labels, model APIs or workbench writes.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json,os,re,sys
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.global_anchor_additions import point_hits,allocate,union_pixels
from tools.lidar_candidate_evidence import ReferenceReadAudit,read_xyz,sensor_cameras
from tools.spatial_candidate_evidence import SpatialPointNovelty,projected_spatial_sensor
from tools.tiled_candidate_utils import _mask_record,_iou
from tools.run_local_agent_acceptance import save

def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
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
    output=Path(args.output).resolve()
    if output.is_relative_to(dataset):raise ValueError('Isolate predictions from source dataset')
    output.mkdir(parents=True,exist_ok=False);bindings={str(p):sha(p) for p in paths}
    helpers=('global_anchor_additions.py','raw_coverage_selection.py','spatial_candidate_evidence.py','lidar_candidate_evidence.py','tiled_candidate_utils.py','run_local_agent_acceptance.py')
    helper_hashes={name:sha(REPO/'tools'/name) for name in helpers}
    methods={'four_crop_control':'unchanged','global_3px_addition_control':'r3','global_all_radii_additions':'all'}
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'primary_method':'global_all_radii_additions',
        'secondary_can_replace_primary':False,'methods':methods,'frames':frames,'cameras':cameras,'planned_images':len(pairs),
        'input_bindings':bindings,'code_sha256':sha(__file__),'helper_sha256':helper_hashes,
        'metric_sha256':sha(REPO/'tools/evaluate_projected_support.py'),
        'fixed_parameters':{'minimum_score':.5,'nms_iou':.8,'global_candidate_count_ratio':1.5,'global_union_area_ratio':1.25,
            'minimum_extra_raw3_points':minimum_extra},
        'local_candidate_budget':'none; distribute unused slots under unchanged original global caps',
        'ranking':'original baseline-relative fixed spatial component-mass priority; deterministic global order',
        'anchor_contract':'Every old four-crop shape stays as the literal prefix, including zero-point masks',
        'primary_raw_contract':'Exact old raw-visible-point union at ellipse radii 0/3/6 per image',
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
        for topic,key in pairs:
            if (output/'STOP').exists():raise InterruptedError('Study stop requested')
            source=records[topic,key];pr=pool_receipts[topic,key]
            if any(r['image_sha256'][topic][key]!=source['sha256'] for r in (br,ar,pr)):raise ValueError('Image binding changed')
            seed=baseline[topic][key];old=anchors[topic][key];values=pool[topic][key]
            if old[:len(seed)]!=seed or values[:len(seed)]!=seed:raise ValueError('Original prefix changed')
            by_id={s['id']:s for s in values}
            if len(by_id)!=len(values) or any(by_id[s['id']]!=s for s in old):raise ValueError('Bound anchor geometry differs')
            camera=info[int(topic.split('/')[2].split('_')[-1])];w,h=camera['width'],camera['height']
            if key!=last_frame:xyz,_=read_xyz(dataset,manifest,int(key));last_frame=key
            sensor=projected_spatial_sensor(xyz,camera);pixels=sensor['pixels'];image_key=(topic,key)
            sensors[image_key]=(pixels,w,h)
            props=values[len(seed):]
            if any(s['label']!='pallet' for s in props):raise ValueError('Pallet-only additions required')
            context=SpatialPointNovelty(seed,props,w,h,sensor)
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
                allowed={r:not np.any(hits[r]&~anchor_hits[image_key][r]) for r in (0,3,6)}
                if not allowed[3]:continue
                candidates.append({'key':image_key,'shape':shape,'record':record,
                    'priority':float(context.priority(shape,tuple(seed),'component_mass')),
                    'total_raw3_points':total_raw3,
                    'allowed':{'r3':True,'all':all(allowed.values())}})
            images.setdefault(topic,{})[key]=source['sha256'];progress['prepared_images']+=1
            save(output/'progress.json',progress)
        cap_count=base_count*3//2;cap_area=base_area*5//4;outputs={'four_crop_control':{t:{f:deepcopy(anchors[t][f]) for tt,f in pairs if tt==t} for t in cameras_as_topics(pairs)}}
        summaries={};readbacks=[]
        for name,mode in methods.items():
            if name=='four_crop_control':continue
            selected,summary=allocate(old_records,candidates,cap_count,cap_area,mode)
            mapping={t:{f:deepcopy(anchors[t][f]) for tt,f in pairs if tt==t} for t in cameras_as_topics(pairs)}
            for item in selected:mapping[item['key'][0]][item['key'][1]].append(deepcopy(item['shape']))
            for topic,key in pairs:
                old=anchors[topic][key];shapes=mapping[topic][key];pixels,w,h=sensors[topic,key]
                if shapes[:len(old)]!=old:raise ValueError('Old instance prefix changed')
                exact={}
                for radius in (0,3,6):
                    covered=anchor_hits[topic,key][radius].copy()
                    for shape in shapes[len(old):]:covered|=point_hits(_mask_record(shape,w,h),pixels,radius,w,h)
                    exact[str(radius)]=bool(np.array_equal(covered,anchor_hits[topic,key][radius]))
                if not exact['3'] or mode=='all' and not all(exact.values()):raise ValueError('Actual raw-union certificate differs')
                readbacks.append({'topic':topic,'frame':int(key),'method':name,'old_four_prefix_exact':True,'raw_union_equal':exact})
            outputs[name]=mapping;summaries[name]=summary
            summary['selected_extra_raw3_below3']=sum(c['total_raw3_points']<3 for c in selected)
        if any(sha(p)!=value for p,value in bindings.items()):raise ValueError('Input changed')
        for name,mapping in outputs.items():
            file=output/(name+'_candidates.json');save(file,mapping)
            save(output/(name+'_receipt.json'),{'created_utc':datetime.now(timezone.utc).isoformat(),
                'archive_sha256':manifest['archive_sha256'],'prediction_json_sha256':sha(file),'image_sha256':images,
                'plan_sha256':plan_sha,'method':name,'candidates_confirmed_as_truth':False,'new_model_requests':0,
                'source':'reference_free_literal_four_crop_anchors_and_global_raw_subset_additions'})
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
    raise SystemExit(run(p.parse_args()))
