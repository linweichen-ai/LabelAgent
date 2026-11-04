"""Recycle exact three-radius raw-hit duplicate additions of a frozen safe incumbent."""
import argparse,hashlib,json,os,sys
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.recycle_duplicate_slots import certified_duplicate_additions
from tools.global_anchor_additions import point_hits,allocate,union_pixels
from tools.multiview_person_guard import ranking_inputs
from tools.evaluate_projected_support import project_visible
from tools.lidar_candidate_evidence import ReferenceReadAudit,read_xyz,sensor_cameras
from tools.spatial_candidate_evidence import SpatialPointNovelty
from tools.tiled_candidate_utils import _mask_record,_iou
from tools.run_local_agent_acceptance import save

read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
NAME='duplicate_slot_recycling'


def run(args):
    dataset=Path(args.dataset).resolve();manifest=read(dataset/'import_manifest.json');paths=[dataset/'import_manifest.json']
    def bound(path,receipt_path):
        path,receipt_path=Path(path).resolve(),Path(receipt_path).resolve();receipt=read(receipt_path)
        if sha(path)!=receipt['prediction_json_sha256'] or receipt['archive_sha256']!=manifest['archive_sha256']:raise ValueError('Changed prediction binding')
        paths.extend([path,receipt_path])
        if 'plan_sha256' in receipt:
            parent_plan=receipt_path.parent/'study_plan.json'
            if sha(parent_plan)!=receipt['plan_sha256']:raise ValueError('Parent plan binding differs')
            paths.append(parent_plan)
        return read(path),receipt
    baseline,br=bound(args.baseline_json,args.baseline_receipt)
    four,fr=bound(args.four_json,args.four_receipt)
    incumbent,ir=bound(args.incumbent_json,args.incumbent_receipt)
    pool={};pool_receipts={}
    if len(args.pool_json)!=len(args.pool_receipt):raise ValueError('Paired pools required')
    for path,receipt in zip(args.pool_json,args.pool_receipt):
        values,meta=bound(path,receipt)
        for t,fs in values.items():
            for f,shapes in fs.items():
                if f in pool.get(t,{}):raise ValueError('Overlapping pools')
                pool.setdefault(t,{})[f]=shapes;pool_receipts[t,f]=meta
    pairs=sorted([(t,f) for t,fs in pool.items() for f in fs],key=lambda p:(int(p[1]),p[0]))
    frames=sorted({int(f) for _,f in pairs});cameras=[0,1,2,3]
    expected={(f'/capture/camera_{c}/image',str(f)) for f in frames for c in cameras}
    if not pairs or set(pairs)!=expected or {(t,f) for t,fs in incumbent.items() for f in fs}!=expected:raise ValueError('Complete same incumbent cohort required')
    output=Path(args.output).resolve()
    if output.is_relative_to(dataset):raise ValueError('Isolate from source data')
    output.mkdir(parents=True,exist_ok=False);bindings={str(p):sha(p) for p in paths}
    helpers=('recycle_duplicate_slots.py','global_anchor_additions.py','multiview_person_guard.py','evaluate_projected_support.py','raw_coverage_selection.py','spatial_candidate_evidence.py','lidar_candidate_evidence.py','tiled_candidate_utils.py','run_local_agent_acceptance.py')
    helper_hashes={n:sha(REPO/'tools'/n) for n in helpers}
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),primary_method=NAME,secondary_can_replace_primary=False,frames=frames,cameras=cameras,planned_images=len(pairs),
        input_bindings=bindings,code_sha256=sha(__file__),helper_sha256=helper_hashes,
        fixed_parameters=dict(minimum_score=.5,nms_iou=.8,global_count_ratio=1.5,global_area_ratio=1.25,minimum_new_total_raw3=3,duplicate_all_radii=True,original_minimum_purity=.67,radii=[0,3,6],ranking_anchor='four'),
        removal='Only additions after immutable four-crop prefix with an identical0/3/6 raw-hit retained witness; deterministic representative; fixed metric purity.67 gives at most one eligible object per column',
        addition='Unchanged component-mass ranking; every raw hit must lie inside frozen incumbent union at all radii',
        certificate='Every image raw union exactly equals incumbent at all0/3/6; an identical raw-hit witness stays for every removed addition',
        prediction_reads_reference=False,approved_reference_inputs=False,model_calls=0,workbench_writes=False,default_enabled=False)
    save(output/'study_plan.json',plan);plan_sha=sha(output/'study_plan.json')
    progress=dict(status='preparing',pid=os.getpid(),prepared_images=0,model_calls=0,workbench_writes=False)
    save(output/'progress.json',progress);audit=ReferenceReadAudit(dataset);audit.start();active=True;denied=[]
    forbidden=[Path(p).resolve() for p in args.deny_reference_root]
    def guard(event,values):
        if event=='open' and active and values and isinstance(values[0],(str,bytes)):
            path=Path(values[0].decode() if isinstance(values[0],bytes) else values[0]).resolve()
            if any(path.is_relative_to(p) for p in forbidden):denied.append(str(path));raise PermissionError('Reference/diagnostic read forbidden')
    sys.addaudithook(guard)
    try:
        cameras_info,_=sensor_cameras(dataset,manifest,cameras)
        records={(r['topic'],str(r['frame'])):r for r in manifest['records'] if r['topic'].endswith('/image')}
        kept_map=deepcopy(incumbent);anchors={};sensors={};unions={};candidates=[];removed=[];images={};last=None;base_count=0;base_area=0
        for t,f in pairs:
            if (output/'STOP').exists():raise InterruptedError('Stop requested')
            source=records[t,f]
            if any(r['image_sha256'][t][f]!=source['sha256'] for r in (br,fr,ir,pool_receipts[t,f])):raise ValueError('Changed source image')
            seed,old,control,values=baseline[t][f],four[t][f],incumbent[t][f],pool[t][f]
            if old[:len(seed)]!=seed or control[:len(old)]!=old or values[:len(seed)]!=seed:raise ValueError('Original prefix differs')
            by_id={s['id']:s for s in values}
            if len(by_id)!=len(values) or any(by_id[s['id']]!=s for s in control):raise ValueError('Incumbent geometry not in bound pool')
            ci=cameras_info[int(t.split('/')[2].split('_')[-1])];w,h=ci['width'],ci['height']
            if f!=last:xyz,_=read_xyz(dataset,manifest,int(f));last=f
            indices,pixels=project_visible(xyz,ci);sensors[t,f]=(pixels,w,h)
            masks={s['id']:_mask_record(s,w,h) for s in values if s['label']=='pallet'}
            pals=[s for s in control if s['label']=='pallet']
            hits=[{r:point_hits(masks[s['id']],pixels,r,w,h) for r in (0,3,6)} for s in pals]
            unions[t,f]={r:np.logical_or.reduce([v[r] for v in hits]) if hits else np.zeros(len(pixels),bool) for r in (0,3,6)}
            keep,drop=certified_duplicate_additions(pals,hits,{s['id'] for s in old})
            drop_ids={pals[i]['id'] for i in drop};kept_map[t][f]=[deepcopy(s) for s in control if s['id'] not in drop_ids]
            for i in drop:
                witness=next(j for j in keep if all(np.array_equal(hits[i][r],hits[j][r]) for r in (0,3,6)))
                removed.append(dict(topic=t,frame=int(f),id=pals[i]['id'],raw3=int(hits[i][3].sum()),witness_id=pals[witness]['id']))
            anchors[t,f]=[masks[s['id']] for s in kept_map[t][f] if s['label']=='pallet']
            original=[masks[s['id']] for s in seed if s['label']=='pallet'];base_count+=len(original);base_area+=union_pixels(original)
            props=values[len(seed):]
            if any(s['label']!='pallet' for s in props):raise ValueError('Pallet-only source additions required')
            rank_seed,rank_props=ranking_inputs(seed,old,props,'four')
            sensor=dict(pixels=pixels,xyz=xyz[indices],depths=xyz[indices]@ci['transform'][2,:3]+ci['transform'][2,3])
            context=SpatialPointNovelty(rank_seed,rank_props,w,h,sensor);old_ids={s['id'] for s in control}
            for s in props:
                if s['id'] in old_ids or s.get('score',0)<.5:continue
                record=masks[s['id']]
                if any(_iou(record,r)>=.8 for r in anchors[t,f]):continue
                sh={r:point_hits(record,pixels,r,w,h) for r in (0,3,6)}
                if sh[3].sum()<3 or any(np.any(sh[r]&~unions[t,f][r]) for r in (0,3,6)):continue
                candidates.append(dict(key=(t,f),shape=s,record=record,priority=float(context.priority(s,tuple(rank_seed),'component_mass')),allowed={'all':True}))
            images.setdefault(t,{})[f]=source['sha256'];progress['prepared_images']+=1;save(output/'progress.json',progress)
        selected,allocation=allocate(anchors,candidates,base_count*3//2,base_area*5//4,'all')
        for item in selected:kept_map[item['key'][0]][item['key'][1]].append(deepcopy(item['shape']))
        readbacks=[]
        for t,f in pairs:
            pixels,w,h=sensors[t,f];old=four[t][f]
            if kept_map[t][f][:len(old)]!=old:raise ValueError('Protected four-crop prefix changed')
            for r in (0,3,6):
                actual=np.zeros(len(pixels),bool)
                for s in kept_map[t][f]:
                    if s['label']=='pallet':actual|=point_hits(_mask_record(s,w,h),pixels,r,w,h)
                if not np.array_equal(actual,unions[t,f][r]):raise ValueError('Actual all-radius union certificate failed')
            readbacks.append(dict(topic=t,frame=int(f),old_four_prefix_exact=True,all_radius_raw_union_exact=True))
        if any(sha(p)!=h for p,h in bindings.items()):raise ValueError('Input changed')
        for name,mapping in [('incumbent_2342_control',incumbent),(NAME,kept_map)]:
            file=output/(name+'_candidates.json');save(file,mapping)
            save(output/(name+'_receipt.json'),dict(created_utc=datetime.now(timezone.utc).isoformat(),archive_sha256=manifest['archive_sha256'],prediction_json_sha256=sha(file),image_sha256=images,plan_sha256=plan_sha,method=name,candidates_confirmed_as_truth=False,new_model_requests=0))
        progress.update(status='complete',completed_images=len(pairs),removed_redundant_additions=removed,removed_count=len(removed),added_count=len(selected),eligible_replacements=len(candidates),allocation=allocation,readbacks=readbacks,all_predictions_frozen_before_scoring=True,plan_sha256=plan_sha)
    except InterruptedError:progress['status']='stopped'
    except Exception as error:
        progress.update(status='failed',error_type=type(error).__name__);(output/'local_error.txt').write_text(repr(error),encoding='utf-8')
    finally:
        audit.stop();active=False
        progress.update(reference_files_opened=len(audit.denied),approved_reference_files_opened=len(denied),input_bindings_unchanged=all(sha(p)==h for p,h in bindings.items()),code_unchanged=sha(__file__)==plan['code_sha256'] and all(sha(REPO/'tools'/n)==v for n,v in helper_hashes.items()),plan_unchanged=sha(output/'study_plan.json')==plan_sha)
        save(output/'progress.json',progress)
    print(json.dumps({k:v for k,v in progress.items() if k not in ('readbacks','removed_redundant_additions')}),flush=True)
    return 0 if progress['status']=='complete' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('dataset','baseline-json','baseline-receipt','four-json','four-receipt','incumbent-json','incumbent-receipt','output'):p.add_argument('--'+n,required=True)
    for n in ('pool-json','pool-receipt'):p.add_argument('--'+n,action='append',required=True)
    p.add_argument('--deny-reference-root',action='append',default=[])
    raise SystemExit(run(p.parse_args()))
