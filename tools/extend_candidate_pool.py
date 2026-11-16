"""Fuse bound historical proposal sources; never read reference labels or APIs."""
import argparse,hashlib,json,os,re,sys
from pathlib import Path
from datetime import datetime,timezone
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.candidate_pool_fusion import fuse_image
from tools.lidar_candidate_evidence import ReferenceReadAudit,sensor_cameras
from tools.run_local_agent_acceptance import save

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(args):
    dataset=Path(args.dataset).resolve();manifest=read(dataset/'import_manifest.json')
    names=args.source_name
    if not names or len(names)!=len(args.source_json) or len(names)!=len(args.source_receipt):raise ValueError('Bind every named source')
    if len(set(names))!=len(names) or any(not re.fullmatch(r'[a-z][a-z0-9_]{0,31}',name) for name in names):raise ValueError('Unique source names required')
    output=Path(args.output).resolve()
    if output.is_relative_to(dataset):raise ValueError('Output must be outside source dataset')
    audit=ReferenceReadAudit(dataset);audit.start()
    active={'value':True};denied=[];forbidden=[Path(p).resolve() for p in args.deny_reference_root]
    def guard(event,values):
        if event=='open' and active['value'] and values and isinstance(values[0],(str,bytes)):
            p=Path(values[0].decode() if isinstance(values[0],bytes) else values[0]).resolve()
            if any(p.is_relative_to(root) for root in forbidden):denied.append(str(p));raise PermissionError('Approved reference forbidden')
    sys.addaudithook(guard)
    bindings={str(dataset/'import_manifest.json'):sha(dataset/'import_manifest.json')}
    def bound(file,receipt_file):
        file,receipt_file=Path(file).resolve(),Path(receipt_file).resolve();meta=read(receipt_file);plan=receipt_file.parent/'study_plan.json'
        if sha(file)!=meta['prediction_json_sha256'] or meta['archive_sha256']!=manifest['archive_sha256']:raise ValueError('Source binding changed')
        if sha(plan)!=meta['plan_sha256']:raise ValueError('Source plan changed')
        for p in (file,receipt_file,plan):bindings[str(p)]=sha(p)
        return read(file),meta
    original,original_receipt=bound(args.pool_json,args.pool_receipt)
    sources=[bound(file,receipt) for file,receipt in zip(args.source_json,args.source_receipt)]
    pairs={(t,k) for t,frames in original.items() for k in frames}
    if not pairs or any(not re.fullmatch(r'/capture/camera_[0-3]/image',t) for t,_ in pairs):raise ValueError('Image cohort required')
    for values,meta in sources:
        if {(t,k) for t,frames in values.items() for k in frames}!=pairs:raise ValueError('Source cohort differs')
    helper_files=[REPO/'tools'/n for n in ('candidate_pool_fusion.py','tiled_candidate_utils.py','lidar_candidate_evidence.py','run_local_agent_acceptance.py','evaluate_projected_support.py','compare_boxes3d.py')]
    helper_files.append(REPO/'source/agent/edit_agent.py');helpers={str(p):sha(p) for p in helper_files}
    output.mkdir(parents=True,exist_ok=False)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'code_sha256':sha(__file__),'helper_sha256':helpers,
        'input_bindings':bindings,'source_names_in_order':names,'minimum_new_score':.5,
        'original_prefix':'all original shapes, IDs, scores, parts and holes unchanged',
        'new_shapes':'copy source mask and score; replace only ID with stable source namespace',
        'duplicates':'same positioned binary raster; preserve original or first source in fixed order',
        'reference_inputs':False,'new_model_requests':0,'workbench_writes':False,'planned_images':len(pairs)}
    save(output/'study_plan.json',plan);plan_hash=sha(output/'study_plan.json')
    progress={'status':'running','pid':os.getpid(),'completed_images':0,'added_shapes':0,'new_model_requests':0,'workbench_writes':False}
    save(output/'progress.json',progress)
    try:
        cameras,_=sensor_cameras(dataset,manifest,sorted({int(t.split('/')[2].split('_')[-1]) for t,_ in pairs}))
        records={(r['topic'],str(r['frame'])):r['sha256'] for r in manifest['records'] if r['topic'].endswith('/image')}
        result={};image_hashes={};provenance={};totals={n:{k:0 for k in ('added','duplicate','low_score','non_pallet')} for n in names}
        for topic,key in sorted(pairs,key=lambda p:(int(p[1]),p[0])):
            if (output/'STOP').exists():raise InterruptedError('Stop requested')
            if any(meta['image_sha256'][topic][key]!=records[topic,key] for meta in [original_receipt,*[s[1] for s in sources]]):raise ValueError('Source image binding changed')
            info=cameras[int(topic.split('/')[2].split('_')[-1])]
            values,origin,counts=fuse_image(original[topic][key],[(name,s[0][topic][key]) for name,s in zip(names,sources)],info['width'],info['height'])
            result.setdefault(topic,{})[key]=values;provenance.setdefault(topic,{})[key]=origin
            image_hashes.setdefault(topic,{})[key]=records[topic,key]
            for name,stats in counts.items():
                for field,value in stats.items():totals[name][field]+=value
            progress['completed_images']+=1;progress['added_shapes']+=len(origin);save(output/'progress.json',progress)
        if any(sha(p)!=h for p,h in bindings.items()) or any(sha(p)!=h for p,h in helpers.items()):raise ValueError('Input or helpers changed')
        file=output/'all_fused_proposals_candidates.json';save(file,result);save(output/'provenance.json',provenance)
        save(output/'all_fused_proposals_receipt.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
            'archive_sha256':manifest['archive_sha256'],'prediction_json_sha256':sha(file),'image_sha256':image_hashes,
            'plan_sha256':plan_hash,'source':'bound_historical_prompt_pool_fusion','new_model_requests':0,'candidates_confirmed_as_truth':False})
        progress.update(status='complete',source_counts=totals,all_original_prefixes_exact=True,plan_sha256=plan_hash)
    except InterruptedError:progress['status']='stopped'
    except Exception as error:
        progress.update(status='failed',error_type=type(error).__name__);(output/'local_error.txt').write_text(str(error),encoding='utf-8')
    finally:
        audit.stop();active['value']=False
        progress.update(reference_files_opened=len(audit.denied),approved_reference_files_opened=len(denied),
            input_bindings_unchanged=all(sha(p)==h for p,h in bindings.items()),
            code_unchanged=sha(__file__)==plan['code_sha256'] and all(sha(p)==h for p,h in helpers.items()),
            plan_unchanged=sha(output/'study_plan.json')==plan_hash)
        save(output/'progress.json',progress)
    print(json.dumps(progress),flush=True);return 0 if progress['status']=='complete' else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','pool-json','pool-receipt','output'):parser.add_argument('--'+name,required=True)
    for name in ('source-name','source-json','source-receipt'):parser.add_argument('--'+name,action='append',required=True)
    parser.add_argument('--deny-reference-root',action='append',default=[])
    raise SystemExit(run(parser.parse_args()))
