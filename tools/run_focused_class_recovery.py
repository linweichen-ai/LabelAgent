"""Isolated fresh-SAM Agent diagnostic using byte-bound real focus responses.

The source panel may have failed its quality gate. This tool only diagnoses
controlled class-flip recovery; it never installs a filter or writes workbench
state. Cached model proposals and new HTTP calls are counted separately.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time

from PIL import Image

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'tools')]
from connect_local_vlm import loopback_url,require_real_service
from evaluate_duplicate_recovery import geometry_counter
from evaluate_instance_focus import checked_focus_fixture,consensus_category,largest_mask_candidate,render_instance_views
from evaluate_point_merges import file_sha,read
from evaluate_vlm_grounding import build_payload,validated_response
from run_local_agent_acceptance import load_image,save
from source.agent import workflow


def current_view_hashes(image,shape):
    views,_=render_instance_views(image,shape,padding=.1)
    hashes=[]
    for view in views:
        encoded=io.BytesIO();view.save(encoded,'JPEG',quality=90)
        hashes.append(hashlib.sha256(encoded.getvalue()).hexdigest())
    return tuple(hashes)


def matching_operation(hashes,lookup,shape,labels):
    proposal=lookup.get(tuple(hashes))
    if proposal not in labels or proposal==shape['label']:return None
    return {'operation':'reclassify','shape_ids':[shape['id']],'label':proposal,
        'reason':'Both byte-bound focused image views identify this class.'}


def load_focus_cache(fixture_path,study_path):
    fixture,assets=checked_focus_fixture(fixture_path)
    study=Path(study_path)
    plan=read(study/'study_plan.json');summary=read(study/'summary.json')
    if (plan['fixture_sha256']!=file_sha(fixture_path) or summary['status']!='complete'
            or summary['valid_real_CUDA_responses']!=2*len(fixture['cases'])
            or plan['source_code_sha256']!=file_sha(REPO/'tools/evaluate_instance_focus.py')):
        raise ValueError('A complete, frozen real focus study is required')
    lookup={};bindings={}
    for index,case in enumerate(fixture['cases']):
        categories=[]
        for offset,kind in enumerate(('single','context'),1):
            path=study/f'response_{index*2+offset:03d}.json'
            receipt=read(path)
            _,fingerprint=build_payload(fixture,assets,case,kind,plan['model_alias'])
            if receipt['case_id']!=case['id'] or receipt['view']!=kind or receipt['input_fingerprint_excluding_model_alias']!=fingerprint:
                raise ValueError('Focus response is not bound to the requested image bytes and prompt')
            parsed=validated_response(receipt['raw_response'],fixture['categories'],len(case['views'][kind]))
            if parsed!=receipt['parsed']:raise ValueError('Stored focus decision differs from the raw model response')
            categories.append(parsed['category']);bindings[str(path.resolve())]=file_sha(path)
        hashes=tuple(ref['sha256'] for ref in case['views']['context'])
        category=consensus_category(*categories)
        if hashes in lookup and lookup[hashes]!=category:raise ValueError('Conflicting focus cache entries')
        lookup[hashes]=category
    return lookup,bindings,read(study/'backend_receipt.json'),summary['all_predeclared_checks_passed']


@contextmanager
def focused_review_policy(agent,lookup,on_check):
    original=workflow.review
    def review(*args,**kwargs):
        if agent.shapes and not kwargs.get('sam_comparison') and not kwargs.get('holes'):
            target=largest_mask_candidate(agent.shapes,agent.width,agent.height)
            hashes=current_view_hashes(agent.image,target)
            operation=matching_operation(hashes,lookup,target,agent.labels)
            on_check({'shape_id':target['id'],'cache_match':hashes in lookup,
                'cached_consensus':lookup.get(hashes),'current_class':target['label'],
                'operation_proposed':operation is not None,'model_requests_added':0})
            if operation is not None:
                decision={'status':'continue','reason':'Focused image evidence supports one class correction.',
                    'hole_assessment':[],'operations':[operation]}
                return workflow._validate_review_result(decision,[],args[4],agent.labels,agent.width,agent.height,None)
        return original(*args,**kwargs)
    workflow.review=review
    try:yield
    finally:workflow.review=original


def evaluate(args):
    if not 2<=args.max_rounds<=5:raise ValueError('Use two to five rounds so edits can be observed')
    lookup,bindings,cache_health,panel_passed=load_focus_cache(args.fixture_json,args.focus_output)
    dataset=Path(args.dataset).resolve();manifest=read(dataset/'import_manifest.json')
    frames=[int(value) for value in args.frames.split(',')]
    if len(frames)!=2 or len(set(frames))!=2 or args.reference_frame in frames:raise ValueError('Use two distinct target frames')
    labels=['pallet','person']
    records={row['frame']:row for row in manifest['records'] if row['topic']==args.topic}
    images={frame:load_image(dataset,records[frame]) for frame in [args.reference_frame,*frames]}
    annotations=read(Path(args.reference_json));workflow.reference_labels(annotations,labels)
    reference={'image':images[args.reference_frame],'marked_image':workflow.reference_overlay(images[args.reference_frame],annotations),'annotations':annotations}
    description=workflow.restrict_description(read(Path(args.description_json)),labels)
    config={'llm_url':loopback_url(args.base),'llm_model':args.model,'llm_key':'local',
        'llm_prompt_profile':'compact','sam3_url':loopback_url(args.sam)}
    output=Path(args.output).resolve()
    if output.is_relative_to(dataset):raise ValueError('Keep diagnostic output outside source data')
    output.mkdir(parents=True,exist_ok=False)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'frames':frames,'generation':'fresh_real_SAM3',
        'reference_kind':'unconfirmed_functional_fixture','description_source':'cached_prior_real_VLM_description',
        'max_rounds':args.max_rounds,'focus_fixture_sha256':file_sha(args.fixture_json),
        'focus_response_bindings':bindings,'focus_source_panel_passed':panel_passed,
        'focus_study_plan_sha256':file_sha(Path(args.focus_output)/'study_plan.json'),
        'focus_study_summary_sha256':file_sha(Path(args.focus_output)/'summary.json'),
        'focus_backend_receipt_sha256':file_sha(Path(args.focus_output)/'backend_receipt.json'),
        'description_sha256':file_sha(args.description_json),'reference_sha256':file_sha(args.reference_json),
        'source_code_sha256':file_sha(__file__),'workflow_sha256':file_sha(workflow.__file__),
        'focus_renderer_sha256':file_sha(REPO/'tools/evaluate_instance_focus.py'),
        'predeclared_checks':['four fresh-SAM runs with real fullscene VLM review',
            'both natural cases exactly preserve fresh seed geometry and labels',
            'both known class flips exactly recover seeds and have an actual edit plus later review',
            'all per-round class-proposal lookups match exact current JPEG bytes',
            'stored originals, source inputs and production workflow unchanged'],
        'workbench_modified':False,'quality_filter_adoption_allowed':False,
        'pixel_accuracy_evaluated':False,'independent_validation':False}
    save(output/'study_plan.json',plan);plan_sha=file_sha(output/'study_plan.json')
    before=file_sha(REPO/'work/state.json')
    summary={'status':'running','pid':__import__('os').getpid(),'cases':[],
        'fresh_SAM3_requests':0,'fresh_VLM_reviews':0,'cached_focus_check_requests_added':0,
        'source_focus_panel_passed':panel_passed,'quality_filter_adopted':False,'workbench_modified':False}
    save(output/'summary.json',summary)
    class StopSignal:
        def is_set(self):return (output/'STOP').exists()
    try:
        health=require_real_service(config['llm_url'],args.model)
        if (health.get('model_config_sha256')!=cache_health['model_config_sha256']
                or health.get('quantization')!='nf4' or health.get('image_token_budget')!=256):
            raise ValueError('Current physical VLM differs from the cached focus experiment')
        for frame in frames:
            for kind in ('natural','class_flip_control'):
                if (output/'STOP').exists():raise workflow.RunCancelled('Diagnostic stop requested')
                folder=output/f'{frame}_{kind}';folder.mkdir()
                traces={};checks=[]
                def trace(event):traces[event['id']]=event;save(folder/'trace.json',list(traces.values()))
                def progress(event):save(folder/'progress.json',{'phase':event['phase'],'round':event['round'],'shape_count':len(event['shapes'])})
                def checked(event):checks.append(event);save(folder/'focus_checks.json',checks)
                class DiagnosticAgent(workflow.AnnotationAgent):
                    def generate(self):
                        super().generate()
                        self.seed=deepcopy(self.shapes)
                        self.target=largest_mask_candidate(self.shapes,self.width,self.height)
                        save(folder/'fresh_seed.json',self.seed)
                        if kind=='class_flip_control':
                            self.target['label']='pallet' if self.target['label']=='person' else 'person'
                        save(folder/'challenged.json',self.shapes)
                agent=DiagnosticAgent(config,images[frame],description,labels,args.max_rounds,progress,reference=reference)
                started=time.monotonic()
                with focused_review_policy(agent,lookup,checked),workflow.trace_calls(trace),workflow.cancellation(StopSignal()):result=agent.execute()
                save(folder/'result.json',result)
                restored=geometry_counter(agent.seed,agent.width,agent.height)==geometry_counter(result['shapes'],agent.width,agent.height)
                completed=[item for item in traces.values() if item.get('status')=='succeeded']
                edits=sum(item.get('status')=='applied' for entry in result['agent']['audit'] for item in entry['tool_results'])
                row={'frame':frame,'kind':kind,'seconds':round(time.monotonic()-started,3),
                    'fresh_SAM3_requests':sum(item['service']=='SAM 3' for item in completed),
                    'fresh_VLM_reviews':sum(item['service']=='GPT' for item in completed),
                    'cached_focus_lookups':len(checks),'cached_focus_matches':sum(item['cache_match'] for item in checks),
                    'cached_class_operations_returned':sum(item['operation_proposed'] for item in checks),
                    'fresh_seed_candidates':len(agent.seed),'remaining_candidates':len(result['shapes']),
                    'seed_geometry_and_classes_restored':restored,'actual_edit_operations':edits,
                    'rounds':result['agent']['rounds'],'converged':bool(result['agent']['converged']),
                    'stop_reason':result['agent']['stop_reason'],'source_images_unchanged':load_image(dataset,records[frame])==images[frame]}
                row['known_flip_recovered_then_observed']=bool(kind=='class_flip_control' and restored and edits and row['converged']
                    and row['stop_reason']=='done' and row['rounds']>=2 and row['fresh_VLM_reviews']>=1)
                summary['cases'].append(row)
                summary['fresh_SAM3_requests']+=row['fresh_SAM3_requests'];summary['fresh_VLM_reviews']+=row['fresh_VLM_reviews']
                save(output/'summary.json',summary);print(json.dumps(row),flush=True)
        summary['status']='complete'
    except workflow.RunCancelled:summary['status']='stopped'
    except Exception as error:
        summary.update(status='failed',error_type=type(error).__name__)
        (output/'local_error.txt').write_text(__import__('traceback').format_exc(),encoding='utf-8')
    finally:
        rows=summary['cases'];complete=len(rows)==4 and summary['status']=='complete'
        summary['checks']={'four_real_fresh_generation_and_review_runs':bool(complete and all(row['fresh_SAM3_requests']>=2 and row['fresh_VLM_reviews']>=1 for row in rows)),
            'both_natural_seeds_preserved_and_converged':bool(complete and all(row['seed_geometry_and_classes_restored'] and row['converged'] and row['stop_reason']=='done' for row in rows if row['kind']=='natural')),
            'both_known_flips_recovered_and_observed':bool(complete and all(row['known_flip_recovered_then_observed'] for row in rows if row['kind']=='class_flip_control')),
            'all_focus_lookups_match_current_pixels':bool(complete and all(row['cached_focus_lookups'] and row['cached_focus_lookups']==row['cached_focus_matches'] for row in rows))}
        summary['state_unchanged']=file_sha(REPO/'work/state.json')==before
        summary['workflow_unchanged']=file_sha(workflow.__file__)==plan['workflow_sha256']
        summary['plan_unchanged']=file_sha(output/'study_plan.json')==plan_sha
        summary['cached_responses_unchanged']=all(file_sha(path)==digest for path,digest in bindings.items())
        summary['source_inputs_unchanged']=(all(load_image(dataset,records[frame])==images[frame] for frame in images)
            and file_sha(args.fixture_json)==plan['focus_fixture_sha256']
            and file_sha(args.description_json)==plan['description_sha256']
            and file_sha(args.reference_json)==plan['reference_sha256']
            and file_sha(Path(args.focus_output)/'study_plan.json')==plan['focus_study_plan_sha256']
            and file_sha(Path(args.focus_output)/'summary.json')==plan['focus_study_summary_sha256']
            and file_sha(Path(args.focus_output)/'backend_receipt.json')==plan['focus_backend_receipt_sha256']
            and file_sha(__file__)==plan['source_code_sha256']
            and file_sha(REPO/'tools/evaluate_instance_focus.py')==plan['focus_renderer_sha256'])
        summary['functional_checks_passed']=all(summary['checks'].values()) and all(summary[key] for key in ('state_unchanged','workflow_unchanged','plan_unchanged','cached_responses_unchanged','source_inputs_unchanged'))
        summary['finished_utc']=datetime.now(timezone.utc).isoformat();save(output/'summary.json',summary)
    return 0 if summary['status']=='complete' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','fixture-json','focus-output','reference-json','description-json','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--frames',default='76,151');parser.add_argument('--reference-frame',type=int,default=0)
    parser.add_argument('--topic',default='/capture/camera_1/image');parser.add_argument('--max-rounds',type=int,default=3)
    parser.add_argument('--base',default='http://127.0.0.1:18868/v1');parser.add_argument('--model',default='labelagent-vlm')
    parser.add_argument('--sam',default='http://127.0.0.1:8766')
    raise SystemExit(evaluate(parser.parse_args()))
