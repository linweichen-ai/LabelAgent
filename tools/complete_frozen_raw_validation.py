"""Finish the frozen per-image raw-coverage study after its existing collection.

Default mode checks readiness only. --execute --wait watches the existing
collector, then replays cached SAM responses, selects, merges and scores.
It never starts collection, calls a model, or confirms workbench annotations.
"""
import argparse
from datetime import datetime,timezone
import hashlib,json,os,re
from pathlib import Path
import subprocess,sys,time
import psutil

from compare_frozen_validation import build_comparison,canonical
from run_local_agent_acceptance import save as atomic_save


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    json.dumps(value,allow_nan=False)
    atomic_save(Path(path),value)


def pipeline_paths(root,run_id):
    if not re.fullmatch(r'v[1-9][0-9]{0,8}',run_id):raise ValueError('Run ID must be a version such as v1 or v2')
    root=Path(root)
    return root/('raw_certificate_full_pipeline_'+run_id+'.json'),root/('raw_certificate_full_pipeline_logs_'+run_id)


def collection_readiness(root,frozen):
    root=Path(root)
    controller=read(root/'raw_certificate_remaining_collection_v1.json')
    if (root/'raw_certificate_remaining_CANCEL').exists():return {'status':'cancelled'}
    if controller.get('plan_sha256')!=sha(root/'raw_certificate_full_validation_plan_v1.json'):
        raise ValueError('Collection belongs to a different frozen full plan')
    if controller['status'] in ('failed','cancelled'):raise ValueError('Existing collection is terminal without completion')
    expected=frozen['new_remaining_SAM_requests']
    if controller['status']=='complete':
        leaf=read(root/'raw_certificate_validation_remaining_dense_v1/progress.json')
        if (leaf['status']!='complete' or leaf['completed_requests']!=expected or
            leaf['completed_images']!=frozen['remaining_images'] or leaf.get('plan_unchanged') is not True or
            leaf.get('baseline_cache_unchanged') is not True):raise ValueError('Collection completion receipt differs')
        plan=root/'raw_certificate_validation_remaining_dense_v1/study_plan.json'
        if leaf['plan_sha256']!=sha(plan):raise ValueError('Completed collection plan changed')
        return {'status':'ready','completed_requests':expected,'completed_images':leaf['completed_images']}
    proc=psutil.Process(controller['controller_pid'])
    created=datetime.fromisoformat(controller['created_utc']).timestamp()
    if (not proc.is_running() or abs(proc.create_time()-created)>10 or
        'collect_raw_certificate_remaining.py' not in ' '.join(proc.cmdline())):
        raise ValueError('Cannot verify the existing collection controller')
    result={'status':'waiting_for_existing_collection','controller_pid':proc.pid,'controller_created':proc.create_time()}
    try:
        leaf=read(root/'raw_certificate_validation_remaining_dense_v1/progress.json')
        result.update(completed_requests=max(controller.get('already_cached_responses',0),leaf['completed_requests']),
                      completed_images=max(controller.get('already_cached_images',0),leaf['completed_images']),
                      verified_requests_this_process=leaf['completed_requests'],
                      new_requests_this_process=leaf.get('new_requests_this_process',0))
    except (json.JSONDecodeError,OSError) as error:
        result['progress_observation_error']=type(error).__name__
    return result


def frozen_code(repo,frozen):
    bindings={'select_raw_coverage_candidates.py':frozen['predictor_sha256'],
        'collect_tiled_candidates.py':frozen['collector_sha256'],'evaluate_projected_support.py':frozen['metric_sha256'],
        **frozen['helper_sha256']}
    for name,value in bindings.items():
        if sha(Path(repo)/'tools'/name)!=value:raise ValueError('Frozen implementation changed: '+name)


def run(args):
    repo=Path(__file__).resolve().parents[1];root=Path(args.artifact_root).resolve();dataset=Path(args.dataset).resolve()
    full_path=root/'raw_certificate_full_validation_plan_v1.json';frozen=read(full_path)
    frozen_code(repo,frozen)
    if sha(dataset/'sam3_baseline/candidates_by_topic.json')!=frozen['baseline_prediction_sha256']:
        raise ValueError('Original baseline differs from frozen full plan')
    if (frozen['secondary_can_replace_primary'] is not False or frozen['primary_method']!='raw_certificate_priority' or
        len(frozen['full_validation_frames'])*len(frozen['cameras'])!=frozen['full_validation_images']):
        raise ValueError('Frozen primary or full cohort differs')
    status=collection_readiness(root,frozen)
    if not args.execute:return status
    state_path,logs=pipeline_paths(root,args.run_id)
    pool=root/'raw_certificate_validation_remaining_pool_v1'
    predictions=root/'raw_certificate_validation_remaining_predictions_v1'
    merged=root/'raw_certificate_full_predictions_v1'
    if any(p.exists() for p in (state_path,logs,pool,predictions,merged)):
        raise ValueError('Pipeline outputs already exist; inspect the current process before any continuation')
    if status['status']=='cancelled':return status
    if status['status']!='ready' and not args.wait:return status
    logs.mkdir()
    state={'status':'starting','pid':os.getpid(),'process_created':psutil.Process().create_time(),
        'created_utc':datetime.now(timezone.utc).isoformat(),'frozen_plan_sha256':sha(full_path),
        'full_images':frozen['full_validation_images'],'primary_method':frozen['primary_method'],
        'new_model_requests':0,'workbench_annotation_writes':False,'child_pid':None}
    inputs=[full_path,dataset/'import_manifest.json',dataset/'sam3_baseline/candidates_by_topic.json',root/'baseline_prediction_receipt.json']
    previous=root/'tiles_validation_v1';first=root/'raw_certificate_validation_predictions_v1'
    inputs.extend(p for p in first.glob('*.json'))
    inputs.extend([previous/'tiles_s50_b50_candidates.json',previous/'evaluation_frozen/summary.json',previous/'evaluation_frozen/per_image.json',
                   root/'raw_certificate_validation_remaining_dense_v1/study_plan.json'])
    bindings={p:sha(p) for p in inputs}
    tools=('replay_multiscale_candidates.py','select_raw_coverage_candidates.py','merge_frozen_prediction_exports.py',
           'evaluate_projected_support.py','compare_frozen_validation.py','complete_frozen_raw_validation.py')
    code={repo/'tools'/name:sha(repo/'tools'/name) for name in tools}
    for relative in ('tools/compare_boxes3d.py','tools/import_capture.py','tools/run_local_agent_acceptance.py',
                     'source/agent/edit_agent.py','source/agent/media.py','source/agent/mcap_video.py','source/agent/sensors.py'):
        code[repo/relative]=sha(repo/relative)
    def cancelled():return (root/'raw_certificate_remaining_CANCEL').exists() or (root/'raw_certificate_full_pipeline_CANCEL').exists()
    def validate():
        frozen_code(repo,frozen)
        if any(sha(p)!=value for p,value in (bindings|code).items()):raise ValueError('Bound source or pipeline code changed')
    def record(**values):
        state.update(values,observed_utc=datetime.now(timezone.utc).isoformat());save(state_path,state)
    def step(label,command):
        if cancelled():raise InterruptedError('User cancellation marker')
        validate();record(status=label,child_pid=None)
        with (logs/(label+'.log')).open('xb') as log:
            child=subprocess.Popen(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                env=os.environ|{'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
            created=psutil.Process(child.pid).create_time();record(child_pid=child.pid,child_created=created)
            while child.poll() is None:
                if cancelled():
                    proc=psutil.Process(child.pid)
                    if abs(proc.create_time()-created)>.1:raise RuntimeError('Owned child identity changed')
                    owned=proc.children(recursive=True)+[proc]
                    for item in reversed(owned):
                        try:item.terminate()
                        except psutil.NoSuchProcess:pass
                    psutil.wait_procs(owned,timeout=10)
                    raise InterruptedError('User cancellation marker')
                time.sleep(2)
        if child.returncode:raise RuntimeError(label+' failed; inspect preserved stage log and output')
        validate();record(child_pid=None,last_completed_stage=label)
    record(**status)
    try:
        last_bucket=None
        while status['status']!='ready':
            if cancelled():raise InterruptedError('User cancellation marker')
            validate();status=collection_readiness(root,frozen)
            if status['status']=='cancelled':raise InterruptedError('User cancellation marker')
            record(**status)
            bucket=status.get('completed_requests',0)//100
            if bucket!=last_bucket:print(json.dumps(status),flush=True);last_bucket=bucket
            if status['status']!='ready':time.sleep(15)
        validate()
        python=[sys.executable,'-u'];base=dataset/'sam3_baseline/candidates_by_topic.json';receipt=root/'baseline_prediction_receipt.json'
        remaining=root/'raw_certificate_validation_remaining_dense_v1'
        frames=','.join(map(str,frozen['remaining_frames']))
        step('replay_remaining',python+[str(repo/'tools/replay_multiscale_candidates.py'),'--dataset',str(dataset),
            '--candidates-json',str(base),'--prediction-receipt',str(receipt),'--study',str(previous),
            '--study',str(remaining),'--frames',frames,'--diagnostic-all-proposals','--output',str(pool)])
        pool_progress=read(pool/'progress.json')
        if pool_progress['status']!='complete' or pool_progress['completed_images']!=frozen['remaining_images']:
            raise ValueError('Remaining replay is incomplete')
        step('select_remaining',python+[str(repo/'tools/select_raw_coverage_candidates.py'),'--dataset',str(dataset),
            '--baseline-json',str(base),'--baseline-receipt',str(receipt),
            '--pool-json',str(pool/'all_multiscale_proposals_candidates.json'),'--pool-receipt',str(pool/'all_multiscale_proposals_receipt.json'),
            '--anchor-json',str(pool/'four_crop_control_candidates.json'),'--anchor-receipt',str(pool/'four_crop_control_receipt.json'),
            '--state-file',str(Path(args.state_file).resolve()),'--output',str(predictions)])
        step('merge_full',python+[str(repo/'tools/merge_frozen_prediction_exports.py'),'--dataset',str(dataset),
            '--frozen-plan',str(full_path),'--baseline-json',str(base),'--baseline-receipt',str(receipt),
            '--study',str(first),'--study',str(predictions),'--output',str(merged)])
        progress=read(merged/'progress.json');merge_plan=read(merged/'study_plan.json')
        if progress['status']!='complete' or progress['images']!=frozen['full_validation_images'] or progress['source_studies']!=2:
            raise ValueError('Full merged cohort is incomplete')
        if merge_plan['frozen_validation_plan_sha256']!=sha(full_path):raise ValueError('Merged plan differs from frozen full plan')
        control=read(merged/'four_crop_control_candidates.json')
        if canonical(control)!=canonical(read(previous/'tiles_s50_b50_candidates.json')):
            raise ValueError('Full four-crop predictions differ from the historical cache')
        methods=['pallet_baseline',*merge_plan['methods']]
        score_bindings={merged/(m+'_candidates.json'):sha(merged/(m+'_candidates.json')) for m in merge_plan['methods']}
        record(all_full_predictions_frozen_before_scoring=True,full_merge_plan_sha256=sha(merged/'study_plan.json'))
        reports={};per_image={};maps={}
        for method in methods:
            candidate=base if method=='pallet_baseline' else merged/(method+'_candidates.json')
            bound=receipt if method=='pallet_baseline' else merged/(method+'_receipt.json')
            output=merged/('evaluation_'+method)
            step('score_'+method,python+[str(repo/'tools/evaluate_projected_support.py'),'--dataset',str(dataset),
                '--candidates-json',str(candidate),'--prediction-receipt',str(bound),
                '--frames',','.join(map(str,frozen['full_validation_frames'])),'--cameras',','.join(map(str,frozen['cameras'])),
                '--output',str(output)])
            reports[method]=read(output/'summary.json');per_image[method]=read(output/'per_image.json');maps[method]=read(candidate)
        if any(sha(p)!=value for p,value in score_bindings.items()):raise ValueError('Full predictions changed during scoring')
        if cancelled():raise InterruptedError('User cancellation marker')
        comparison=build_comparison(frozen,reports,per_image,maps,read(previous/'evaluation_frozen/summary.json'),
                                    read(previous/'evaluation_frozen/per_image.json'))
        comparison.update(observed_utc=datetime.now(timezone.utc).isoformat(),frozen_validation_plan_sha256=sha(full_path),
            metric_sha256=frozen['metric_sha256'],frozen_primary_before_new_model_calls=True,
            new_model_requests_this_pipeline=0,source_bindings_unchanged=True)
        validate();save(merged/'comparison.json',comparison)
        record(status='complete',primary_passed=comparison['primary_passed'],winner=comparison['winner'],
            comparison_file=str(merged/'comparison.json'),new_model_requests=0,workbench_annotation_writes=False)
        print(json.dumps({k:v for k,v in state.items() if k!='comparison_file'}),flush=True)
    except InterruptedError:
        record(status='cancelled');raise
    except Exception as error:
        record(status='failed',error_type=type(error).__name__,error=str(error));raise
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','artifact-root','state-file'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--wait',action='store_true')
    parser.add_argument('--run-id',default='v1')
    try:result=run(parser.parse_args());print(json.dumps(result),flush=True)
    except InterruptedError:raise SystemExit(2)
