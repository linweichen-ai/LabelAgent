"""Compare real VLM tool decisions with appended candidate evidence.

The evidence bundle adds three labelled-ID crops and explanatory text. It never
reuses cached category proposals, changes operation schema, or fabricates edits.
Only SAM3 seeds and the prior reference description are cached.
"""
import argparse
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
from evaluate_point_merges import checked_predictions,file_sha,read
from candidate_evidence import scoped_evidence_policy
from run_local_agent_acceptance import load_image,save
from source.agent import workflow,protocol_schema
from source.agent.edit_agent import _rasterize


def sha_value(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def result_metrics(seed,result,width,height,target_id,identity_targets):
    expected=geometry_counter(seed,width,height)
    actual=geometry_counter(result['shapes'],width,height)
    applied=0
    for entry in result['agent']['audit']:
        applied+=sum(row.get('status')=='applied' for row in entry['tool_results'])
    done=bool(result['agent']['converged'] and result['agent']['stop_reason']=='done')
    return {'original_candidate_geometry_and_classes_restored':actual==expected,
        'known_class_flip_recovered':actual==expected if target_id is not None else None,
        'natural_seed_preserved':actual==expected if target_id is None else None,
        'identity_reclassification_targets':identity_targets,'applied_edit_operations':applied,
        'rounds':result['agent']['rounds'],'converged':done,'stop_reason':result['agent']['stop_reason'],
        'multi_round_recovery_observed':bool(target_id is not None and actual==expected and applied and done and result['agent']['rounds']>=2)}


def evaluate(args):
    if not 1<=args.max_rounds<=5:raise ValueError('Use one to five review rounds')
    dataset=Path(args.dataset).resolve()
    manifest=read(dataset/'import_manifest.json')
    predictions,binding=checked_predictions(args.candidates_json,args.prediction_receipt,manifest)
    frames=[int(value) for value in args.frames.split(',')]
    if len(frames)!=2 or len(set(frames))!=2 or args.reference_frame in frames:
        raise ValueError('Use two distinct targets, separate from the reference')
    labels=['pallet','person']
    records={row['frame']:row for row in manifest['records'] if row['topic']==args.topic}
    images={frame:load_image(dataset,records[frame]) for frame in [args.reference_frame,*frames]}
    description=workflow.restrict_description(read(Path(args.description_json)),labels)
    annotations=read(Path(args.reference_json))
    workflow.reference_labels(annotations,labels)
    reference={'image':images[args.reference_frame],
        'marked_image':workflow.reference_overlay(images[args.reference_frame],annotations),'annotations':annotations}
    prepared=[]
    for frame in frames:
        if binding['image_sha256'].get(args.topic,{}).get(str(frame))!=records[frame]['sha256']:
            raise ValueError('Predictions are not bound to the selected image')
        width,height=Image.open(io.BytesIO(images[frame])).size
        seed=deepcopy(predictions[args.topic][str(frame)])
        if not seed or len({shape['id'] for shape in seed})!=len(seed) or any(
                type(shape['id']) is not int or shape['label'] not in labels or not workflow.valid_shape(shape,width,height) for shape in seed):
            raise ValueError('Invalid cached candidate facts')
        person=max((shape for shape in seed if shape['label']=='person'),
            key=lambda shape:int(_rasterize(shape,width,height).sum()))
        for kind in ('natural','class_flip_control'):
            challenged=deepcopy(seed)
            if kind=='class_flip_control':
                next(shape for shape in challenged if shape['id']==person['id'])['label']='pallet'
            prepared.append({'id':f'{frame}_{kind}','frame':frame,'kind':kind,'seed':seed,
                'challenged':challenged,'target_id':person['id'] if kind=='class_flip_control' else None,
                'width':width,'height':height})
    base=loopback_url(args.base)
    config={'llm_url':base,'llm_model':args.model,'llm_key':'local','llm_prompt_profile':'compact',
        'sam3_url':loopback_url(args.sam)}
    output=Path(args.output).resolve()
    if output.is_relative_to(dataset):raise ValueError('Keep output outside the imported dataset')
    output.mkdir(parents=True,exist_ok=False)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'model_alias':args.model,'model_config_sha256':args.expected_model_config_sha256,
        'quantization':'nf4','image_token_budget':256,'reference_kind':'unconfirmed_functional_fixture',
        'frames':frames,'max_rounds':args.max_rounds,'cached_generation':True,'fresh_SAM3_generation_requests':0,
        'source_prediction_sha256':file_sha(args.candidates_json),'description_sha256':file_sha(args.description_json),
        'reference_sha256':file_sha(args.reference_json),'workflow_sha256':file_sha(workflow.__file__),
        'protocol_schema_sha256':file_sha(protocol_schema.__file__),'source_code_sha256':file_sha(__file__),
        'evidence_helper_sha256':file_sha(REPO/'tools/candidate_evidence.py'),
        'image_sha256':{str(frame):records[frame]['sha256'] for frame in [args.reference_frame,*frames]},
        'cases':[{key:case[key] for key in ('id','frame','kind','target_id')} for case in prepared],
        'policies':['baseline','candidate_crops'],'model_input_intervention':'three_ID_crop_images_plus_explanatory_text; original input prefix and response schema retained',
        'predeclared_checks':['all eight cases evaluated with real VLM calls',
            'original prompt/image prefix and operation schema equal across each policy pair',
            'candidate evidence policy produces zero identity class changes',
            'both injected class flips exactly recovered with a real edit and subsequent converged review',
            'both natural controls preserve geometry/classes and converge',
            'production workflow, stored workbench and source inputs unchanged'],
        'pixel_accuracy_evaluated':False,'independent_validation':False,'workbench_modified':False}
    save(output/'study_plan.json',plan)
    plan_sha=file_sha(output/'study_plan.json')
    state_path=REPO/'work/state.json'
    state_before=file_sha(state_path)
    summary={'status':'running','pid':__import__('os').getpid(),'cases':[],
        'successful_real_VLM_calls':0,'successful_SAM3_tool_calls':0,'fresh_SAM3_generation_requests':0,
        'workbench_modified':False,'pixel_accuracy_evaluated':False}
    save(output/'summary.json',summary)
    class StopSignal:
        def is_set(self):return (output/'STOP').exists()
    try:
        health=require_real_service(base,args.model)
        if (health.get('model_config_sha256')!=args.expected_model_config_sha256
                or health.get('quantization')!='nf4' or health.get('image_token_budget')!=256):
            raise ValueError('Physical model or decoding budget differs from the plan')
        save(output/'backend_receipt.json',health)
        for index,case in enumerate(prepared):
            order=['baseline','candidate_crops'] if index in (0,3) else ['candidate_crops','baseline']
            for policy in order:
                if StopSignal().is_set():raise workflow.RunCancelled('Experiment stop requested')
                folder=output/(case['id']+'_'+policy)
                folder.mkdir()
                save(folder/'seed.json',case['seed'])
                save(folder/'challenged.json',case['challenged'])
                traces={}
                bundles=[]
                def bundle(event):
                    bundles.append(event);save(folder/'evidence_bundles.json',bundles)
                identity_targets=[]
                def trace(event):
                    traces[event['id']]=event
                    save(folder/'trace.json',list(traces.values()))
                def progress(event):
                    operation=event.get('operation',{})
                    if event['phase']=='editing' and operation.get('operation')=='reclassify':
                        current={shape['id']:shape['label'] for shape in event['shapes']}
                        identity_targets.extend(identifier for identifier in operation['shape_ids'] if current.get(identifier)==operation['label'])
                    save(folder/'progress.json',{'phase':event['phase'],'round':event['round'],'shape_count':len(event['shapes'])})
                class SeededAgent(workflow.AnnotationAgent):
                    def generate(self):
                        self.shapes=deepcopy(case['challenged'])
                        self.next_id=max(shape['id'] for shape in self.shapes)+1
                        self.progress('cached_real_candidates_loaded')
                started=time.monotonic()
                agent=SeededAgent(config,images[case['frame']],description,labels,args.max_rounds,progress,reference=reference)
                with scoped_evidence_policy(policy,agent,bundle),workflow.trace_calls(trace),workflow.cancellation(StopSignal()):
                    result=agent.execute()
                save(folder/'result.json',result)
                (folder/'result.jpg').write_bytes(workflow.candidate_overlay(images[case['frame']],result['shapes']))
                completed=[row for row in traces.values() if row.get('status')=='succeeded']
                first=next(row['input'] for row in traces.values() if row['service']=='GPT')
                first=deepcopy(first)
                save(folder/'first_model_input.json',first)
                schema=first.pop('response_schema',None)
                base_first=deepcopy(first)
                if policy=='candidate_crops' and bundles and bundles[0]['appended']:
                    base_first['instruction']=base_first['instruction'].split('\nAdditional images ',1)[0]
                    base_first['images']=base_first['images'][:bundles[0]['base_image_count']]
                row={key:case[key] for key in ('id','frame','kind','target_id')}
                row.update(policy=policy,seconds=round(time.monotonic()-started,3),
                    first_prompt_and_images_sha256=sha_value(first),original_prefix_sha256=sha_value(base_first),first_schema_sha256=sha_value(schema),
                    first_image_count=len(first['images']),evidence_bundles_added=len(bundles),
                    all_evidence_bundles_added=bool(bundles and all(item['appended'] for item in bundles)),
                    successful_real_VLM_calls=sum(item['service']=='GPT' for item in completed),
                    successful_SAM3_tool_calls=sum(item['service']=='SAM 3' for item in completed),
                    **result_metrics(case['seed'],result,case['width'],case['height'],case['target_id'],len(identity_targets)))
                summary['cases'].append(row)
                summary['successful_real_VLM_calls']+=row['successful_real_VLM_calls']
                summary['successful_SAM3_tool_calls']+=row['successful_SAM3_tool_calls']
                save(output/'summary.json',summary)
                print(json.dumps(row),flush=True)
        summary['status']='complete'
    except workflow.RunCancelled:
        summary['status']='stopped'
    except Exception as error:
        summary.update(status='failed',error_type=type(error).__name__)
        (output/'local_error.txt').write_text(__import__('traceback').format_exc(),encoding='utf-8')
    finally:
        rows=summary['cases']
        constrained=[row for row in rows if row['policy']=='candidate_crops']
        pairs={case['id']:[row for row in rows if row['id']==case['id']] for case in prepared}
        complete=summary['status']=='complete' and len(rows)==8 and all(row['successful_real_VLM_calls'] for row in rows)
        summary['checks']={'all_cases_have_real_responses':bool(complete),
            'original_prompt_images_and_schema_preserved':bool(complete and all(len(pair)==2 and pair[0]['original_prefix_sha256']==pair[1]['original_prefix_sha256'] and pair[0]['first_schema_sha256']==pair[1]['first_schema_sha256'] for pair in pairs.values())),
            'evidence_policy_identity_reclassifications_zero':bool(complete and all(row['identity_reclassification_targets']==0 for row in constrained)),
            'all_enhanced_runs_receive_three_crop_images':bool(complete and all(row['all_evidence_bundles_added'] and row['first_image_count']==7 for row in constrained)),
            'both_class_flip_controls_recovered_with_observation':bool(complete and all(row['multi_round_recovery_observed'] for row in constrained if row['kind']=='class_flip_control')),
            'both_natural_controls_preserved_and_converged':bool(complete and all(row['natural_seed_preserved'] and row['converged'] for row in constrained if row['kind']=='natural'))}
        summary['state_unchanged']=file_sha(state_path)==state_before
        summary['workflow_unchanged']=file_sha(workflow.__file__)==plan['workflow_sha256']
        summary['plan_unchanged']=file_sha(output/'study_plan.json')==plan_sha
        summary['source_inputs_unchanged']=(all(load_image(dataset,records[frame])==images[frame] for frame in images)
            and file_sha(args.candidates_json)==plan['source_prediction_sha256']
            and file_sha(args.description_json)==plan['description_sha256']
            and file_sha(args.reference_json)==plan['reference_sha256']
            and file_sha(protocol_schema.__file__)==plan['protocol_schema_sha256']
            and file_sha(REPO/'tools/candidate_evidence.py')==plan['evidence_helper_sha256']
            and file_sha(__file__)==plan['source_code_sha256'])
        summary['all_predeclared_checks_passed']=all(summary['checks'].values()) and all(summary[key] for key in ('state_unchanged','workflow_unchanged','plan_unchanged','source_inputs_unchanged'))
        summary['finished_utc']=datetime.now(timezone.utc).isoformat()
        save(output/'summary.json',summary)
    print(json.dumps({key:summary[key] for key in ('status','successful_real_VLM_calls','checks','all_predeclared_checks_passed')}),flush=True)
    return 0 if summary['status']=='complete' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','candidates-json','prediction-receipt','description-json','reference-json','output','expected-model-config-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--frames',default='76,151')
    parser.add_argument('--topic',default='/capture/camera_1/image')
    parser.add_argument('--reference-frame',type=int,default=0)
    parser.add_argument('--max-rounds',type=int,default=3)
    parser.add_argument('--base',default='http://127.0.0.1:18868/v1')
    parser.add_argument('--model',default='labelagent-vlm')
    parser.add_argument('--sam',default='http://127.0.0.1:8766')
    raise SystemExit(evaluate(parser.parse_args()))
