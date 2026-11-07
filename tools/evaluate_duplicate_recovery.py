"""Measure recovery from an exact duplicate injected into local SAM3 candidates.

The seed is unconfirmed. Metrics compare the result to that seed only, and do
not measure real segmentation accuracy. All outputs go to a new local folder.
The default replays cached real SAM3; --generation fresh runs real SAM3 first.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from connect_local_vlm import loopback_url, require_real_service
from run_local_agent_acceptance import load_image, save


def geometry_counter(shapes, width, height):
    from source.agent.edit_agent import _rasterize
    return Counter((shape['label'], hashlib.sha256(_rasterize(shape,width,height).tobytes()).hexdigest())
                   for shape in shapes)


def recovery_metrics(seed, result, width, height, target=None):
    """IDs/order may change; labels, masks and per-mask multiplicity must match."""
    expected = geometry_counter(seed,width,height)
    actual = geometry_counter(result['shapes'],width,height)
    target_key = next(iter(geometry_counter([target],width,height))) if target is not None else None
    unrelated_before = Counter({key:value for key,value in expected.items() if key != target_key})
    unrelated_after = Counter({key:value for key,value in actual.items() if key != target_key})
    applied = sum(item.get('status')=='applied' for row in result['agent']['audit'] for item in row['tool_results'])
    return {'known_injection_recovered': actual==expected if target is not None else None,
            'control_preserved': actual==expected if target is None else None,
            'unrelated_candidate_geometry_preserved': unrelated_before==unrelated_after,
            'applied_edits':applied,
            'multi_round_correction_observed':bool(applied and result['agent']['rounds']>=2 and result['agent']['converged']),
            'pixel_accuracy_evaluated':False}


def evaluate(args):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root))
    from source.agent import workflow
    from source.agent.edit_agent import _rasterize
    if args.workflow_file:
        specification = importlib.util.spec_from_file_location('source.agent.workflow_benchmark',args.workflow_file)
        workflow = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(workflow)
    if not 1 <= args.max_rounds <= 20:
        raise ValueError('Review budget must be between 1 and 20 rounds')
    labels = [label.strip() for label in args.labels.split(',') if label.strip()]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('Labels must be distinct')
    cases = []
    for value in args.cases.split(','):
        frame_text, label = value.split(':')
        if label != 'none' and label not in labels:
            raise ValueError('Injected class must be allowed, or none for a control')
        cases.append((int(frame_text),label))
    dataset = Path(args.dataset).resolve()
    manifest = json.loads((dataset/'import_manifest.json').read_text(encoding='utf-8'))
    records = {row['frame']:row for row in manifest['records'] if row['topic']==args.topic}
    reference_records = [row for row in manifest['records'] if row['topic']==args.reference_topic and row['frame']==args.reference_frame]
    if len(reference_records) != 1 or any(frame not in records for frame,_ in cases):
        raise ValueError('Requested images must match the import manifest')
    reference_bytes = Path(args.reference_json).read_bytes()
    annotations = json.loads(reference_bytes)
    workflow.reference_labels(annotations,labels)
    reference_image = load_image(dataset,reference_records[0])
    reference = {'image':reference_image,'marked_image':workflow.reference_overlay(reference_image,annotations),
                 'annotations':annotations}
    description_bytes = Path(args.description_json).read_bytes()
    description = workflow.restrict_description(json.loads(description_bytes),labels)
    cache_bytes = Path(args.candidates_json).read_bytes() if args.generation=='cached' else None
    cached = json.loads(cache_bytes)[args.topic] if cache_bytes is not None else {}
    images = {frame:load_image(dataset,records[frame]) for frame,_ in cases}
    config = {'llm_url':loopback_url(args.base),'llm_model':args.model,'llm_key':'local',
              'llm_prompt_profile':'compact','sam3_url':loopback_url(args.sam),
              'llm_duplicate_review':args.duplicate_review}
    output = Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    report = {'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
              'scope':'known_exact_duplicate_recovery','reference_kind':args.reference_kind,
              'generation':args.generation,'topic':args.topic,'cases':[],
              'experimental_features':{'duplicate_review':args.duplicate_review},
              'review_backend':'real_local_CUDA_VLM','pixel_accuracy_evaluated':False,'workbench_modified':False,
              'workflow_sha256':hashlib.sha256(Path(workflow.__file__).read_bytes()).hexdigest(),
              'reference_json_sha256':hashlib.sha256(reference_bytes).hexdigest(),
              'description_sha256':hashlib.sha256(description_bytes).hexdigest(),
              'cached_candidates_sha256':hashlib.sha256(cache_bytes).hexdigest() if cache_bytes is not None else None}
    save(output/'summary.json',report)
    try:
        require_real_service(config['llm_url'],args.model)
        for case_index,(frame,label) in enumerate(cases):
            folder = output/f'{case_index:02d}_frame_{frame}_{label}'
            folder.mkdir()
            image = images[frame]
            width,height = Image.open(io.BytesIO(image)).size
            class ChallengeAgent(workflow.AnnotationAgent):
                def generate(self):
                    if args.generation=='fresh':
                        super().generate()
                    else:
                        self.shapes = deepcopy(cached[str(frame)])
                        for index,shape in enumerate(self.shapes,1):
                            shape['id']=index
                        self.next_id=max([shape['id'] for shape in self.shapes],default=0)+1
                    if not self.shapes or any(shape.get('label') not in labels or
                        not workflow.valid_shape(shape,width,height) for shape in self.shapes):
                        raise ValueError('Seed candidates must have allowed labels and valid geometry')
                    self.seed=deepcopy(self.shapes)
                    self.target=None
                    if label!='none':
                        eligible=[shape for shape in self.shapes if shape['label']==label]
                        if not eligible:
                            raise ValueError('The requested class has no seed candidate')
                        self.target=deepcopy(max(eligible,key=lambda shape:int(_rasterize(shape,width,height).sum())))
                        duplicate=deepcopy(self.target)
                        duplicate['id']=self.next_id
                        self.next_id+=1
                        self.shapes.append(duplicate)
                    save(folder/'seed.json',self.seed)
                    save(folder/'challenged.json',self.shapes)
                    (folder/'before.jpg').write_bytes(workflow.candidate_overlay(image,self.shapes))
                    self.progress('known_injection_prepared')
            traces={}
            def trace(event):
                traces[event['id']]=event
                save(folder/'trace.json',list(traces.values()))
            def progress(event):
                save(folder/'progress.json',{'phase':event['phase'],'round':event['round'],'shape_count':len(event['shapes'])})
            row={'frame':frame,'injected_class':None if label=='none' else label}
            started=time.monotonic()
            try:
                with workflow.trace_calls(trace):
                    agent=ChallengeAgent(config,image,description,labels,args.max_rounds,progress,reference=reference)
                    result=agent.execute()
                save(folder/'result.json',result)
                (folder/'after.jpg').write_bytes(workflow.candidate_overlay(image,result['shapes']))
                if not hasattr(agent,'seed'):
                    raise ValueError('Candidate generation did not finish')
                vlm_calls=sum(event.get('service')=='GPT' and event.get('status')=='succeeded' for event in traces.values())
                sam_calls=sum(event.get('service')=='SAM 3' and event.get('status')=='succeeded' for event in traces.values())
                metrics=recovery_metrics(agent.seed,result,width,height,agent.target)
                row.update(metrics,
                    call_completed=bool(result['agent']['stop_reason']!='error' and vlm_calls),
                    seed_candidates=len(agent.seed),challenged_candidates=len(agent.seed)+int(agent.target is not None),
                    remaining_candidates=len(result['shapes']),
                    rounds=result['agent']['rounds'],converged=result['agent']['converged'],
                    stop_reason=result['agent']['stop_reason'],
                    successful_real_vlm_calls=vlm_calls,successful_sam_calls=sam_calls,
                    fresh_generation_and_multi_round_correction=bool(sam_calls and vlm_calls>=2 and
                        metrics['multi_round_correction_observed'] and metrics['known_injection_recovered'] is True))
            except Exception as error:
                row.update(call_completed=False,error_type=type(error).__name__)
                (folder/'error.txt').write_text(str(error),encoding='utf-8')
            row['seconds']=round(time.monotonic()-started,3)
            report['cases'].append(row)
            save(output/'summary.json',report)
            print(json.dumps(row),flush=True)
        report['status']='complete'
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__)
        (output/'local_error.txt').write_text(str(error),encoding='utf-8')
    finally:
        try:
            report['source_images_unchanged']=all(images[frame]==load_image(dataset,records[frame]) for frame,_ in cases)
            report['reference_image_unchanged']=reference_image==load_image(dataset,reference_records[0])
        except Exception:
            report.update(source_images_unchanged=False,status='failed')
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        save(output/'summary.json',report)
    return 0 if report['status']=='complete' and all(row.get('call_completed') for row in report['cases']) else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--candidates-json')
    parser.add_argument('--reference-json',required=True)
    parser.add_argument('--reference-kind',required=True,choices=['human_reviewed','unconfirmed_functional_fixture'])
    parser.add_argument('--description-json',required=True)
    parser.add_argument('--topic',default='/capture/camera_1/image')
    parser.add_argument('--reference-topic',default='/capture/camera_1/image')
    parser.add_argument('--reference-frame',type=int,default=0)
    parser.add_argument('--labels',default='pallet,person')
    parser.add_argument('--cases',default='76:none,76:person,76:pallet')
    parser.add_argument('--generation',choices=['cached','fresh'],default='cached')
    parser.add_argument('--duplicate-review',action='store_true',help='Enable experimental VLM-directed exact-duplicate review')
    parser.add_argument('--workflow-file',help='Optional trusted local workflow source for an isolated comparison')
    parser.add_argument('--max-rounds',type=int,default=3)
    parser.add_argument('--base',default='http://127.0.0.1:18868/v1')
    parser.add_argument('--model',default='labelagent-vlm')
    parser.add_argument('--sam',default='http://127.0.0.1:8766')
    parser.add_argument('--output',required=True)
    arguments=parser.parse_args()
    if arguments.generation=='cached' and not arguments.candidates_json:
        parser.error('--candidates-json is required for cached generation')
    raise SystemExit(evaluate(arguments))
