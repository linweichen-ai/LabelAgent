"""Run real SAM3 -> tiled proposals -> geometric merge -> production VLM Agent.

All results are isolated, unconfirmed exports. No workbench or truth API is
called. Geometric preprocessing and VLM-requested edits are counted separately.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time

from PIL import Image

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from connect_local_vlm import loopback_url,require_real_service
from merge_candidate_overlaps import merge_added_overlaps
from run_local_agent_acceptance import evaluate_run,load_image,save
from tiled_candidate_utils import augment_candidates,remap_tile_result,tile_grid
from source.agent import workflow


def sha(value):
    return hashlib.sha256(value).hexdigest()


def evaluate(args):
    frames=[int(value) for value in args.frames.split(',')]
    cameras=[int(value) for value in args.cameras.split(',')]
    if (not frames or len(set(frames))!=len(frames) or not cameras or len(set(cameras))!=len(cameras)
            or any(camera not in range(4) for camera in cameras) or not 1<=args.max_rounds<=20
            or not .3<=args.merge_threshold<=1):
        raise ValueError('Invalid frame, camera, round budget or merge threshold')
    dataset=Path(args.dataset).resolve()
    manifest=json.loads((dataset/'import_manifest.json').read_text(encoding='utf-8'))
    records={(row['topic'],row['frame']):row for row in manifest['records'] if row['topic'].endswith('/image')}
    keys=[(f'/capture/camera_{camera}/image',frame) for frame in frames for camera in cameras]
    reference_key=(args.reference_topic,args.reference_frame)
    if reference_key in keys or any(key not in records for key in [reference_key,*keys]):
        raise ValueError('Reference and distinct target images must exist in the manifest')
    reference_raw=Path(args.reference_json).read_bytes()
    annotations=json.loads(reference_raw)
    labels=['pallet','person']
    workflow.reference_labels(annotations,labels)
    reference_image=load_image(dataset,records[reference_key])
    width,height=Image.open(io.BytesIO(reference_image)).size
    if (not isinstance(annotations,list) or not annotations or len(annotations)>200
            or any(not workflow.valid_shape(shape,width,height) for shape in annotations)
            or len({shape['id'] for shape in annotations})!=len(annotations)):
        raise ValueError('Reference must contain valid unique integer-ID 2D shapes')
    images={key:load_image(dataset,records[key]) for key in keys}
    base,sam=loopback_url(args.base),loopback_url(args.sam)
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    config={'llm_url':base,'llm_model':args.model,'llm_key':'local','sam3_url':sam,
        'llm_prompt_profile':'compact','llm_duplicate_review':False}
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'frames':frames,'cameras':cameras,
        'reference_kind':args.reference_kind,'reference_topic':args.reference_topic,'reference_frame':args.reference_frame,
        'reference_json_sha256':sha(reference_raw),'max_rounds':args.max_rounds,
        'queries':{'pallet':'pallet','person':'person'},'tile_fraction':.6,'internal_crop_edge_margin':2,
        'minimum_score':.5,'addition_ratio':.5,'NMS_IoU':.8,'merge_threshold':args.merge_threshold,
        'workflow_sha256':sha(Path(workflow.__file__).read_bytes()),
        'source_code_sha256':sha(Path(__file__).read_bytes()),'reference_3D_labels_used_for_prediction':False,
        'independent_scene_test':False,'candidates_confirmed_as_truth':False,'workbench_modified':False}
    save(output/'study_plan.json',plan)
    summary={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
        'planned_images':len(keys),'completed_images':0,'cases':[],
        'plan_sha256':sha((output/'study_plan.json').read_bytes()),'reference_kind':args.reference_kind,
        'full_agent_loops_completed':False,'pixel_accuracy_evaluated':False,'workbench_modified':False}
    save(output/'summary.json',summary)
    traces={}
    def record(event):
        traces[event['id']]=event
        save(output/'model_trace.json',list(traces.values()))
    exports={phase:{} for phase in ('base','tiled','merged','final')}
    inputs={}
    try:
        save(output/'backend_receipt.json',require_real_service(base,args.model))
        marked=workflow.reference_overlay(reference_image,annotations)
        reference={'image':reference_image,'marked_image':marked,'annotations':annotations}
        with workflow.trace_calls(record):
            description=workflow.describe(config,reference_image,annotations,labels,marked_image=marked)
            # Freeze category queries to compare geometry processing fairly;
            # the real VLM's detailed description remains the review rule.
            description['prompts']=[{'label':label,'text':plan['queries'][label]} for label in labels]
            save(output/'description.json',description)
            for topic,frame in keys:
                if (output/'STOP').exists():
                    summary['status']='stopped'
                    break
                folder=output/f"{topic.split('/')[2]}_frame_{frame:04d}"
                folder.mkdir()
                started=time.monotonic()
                before_traces=set(traces)
                phases={}
                geometry={}
                events=[]
                def progress(event):
                    events.append({'phase':event['phase'],'round':event['round'],'shapes':len(event['shapes'])})
                    save(folder/'events.json',events)
                class FreshUnionAgent(workflow.AnnotationAgent):
                    def generate(self):
                        super().generate()
                        seed=deepcopy(self.shapes)
                        phases['base']=seed
                        save(folder/'base_shapes.json',seed)
                        source=Image.open(io.BytesIO(self.image)).convert('RGB')
                        proposals=[]
                        for tile_index,bounds in enumerate(tile_grid(self.width,self.height,plan['tile_fraction'])):
                            buffer=io.BytesIO()
                            source.crop(bounds).save(buffer,'PNG')
                            raw=buffer.getvalue()
                            response=workflow.sam_text(raw,'pallet',self.config)
                            save(folder/f'tile_{tile_index}.json',{'bounds':bounds,'input_sha256':sha(raw),'results':response})
                            for value in response:
                                shape=remap_tile_result(value,bounds,self.width,self.height,self.next_id,plan['internal_crop_edge_margin'])
                                if shape is not None:
                                    proposals.append(shape)
                                    self.next_id+=1
                        tiled,tile_info=augment_candidates(seed,proposals,self.width,self.height,
                            minimum_score=plan['minimum_score'],addition_ratio=plan['addition_ratio'],nms_iou=plan['NMS_IoU'])
                        merged,merge_info=merge_added_overlaps(seed,tiled[len(seed):],self.width,self.height,plan['merge_threshold'])
                        phases.update(tiled=deepcopy(tiled),merged=deepcopy(merged))
                        geometry.update(tile_info=tile_info,merge_info=merge_info)
                        self.shapes=merged
                        self.generated=len(merged)
                        save(folder/'tiled_shapes.json',tiled)
                        save(folder/'merged_shapes.json',merged)
                        save(folder/'geometry_preprocessing.json',geometry)
                        self.progress('generated',geometric_merges=merge_info['merged_additions'])
                agent=FreshUnionAgent(config,images[topic,frame],description,labels,
                    max_rounds=args.max_rounds,on_progress=progress,reference=reference)
                result=agent.execute()
                phases['final']=deepcopy(result['shapes'])
                save(folder/'result.json',result)
                evidence=evaluate_run(result,[value for identifier,value in traces.items() if identifier not in before_traces])
                case={'topic':topic,'frame':frame,'seconds':round(time.monotonic()-started,3),
                    'source_image_sha256':records[topic,frame]['sha256'],
                    'stage_candidates':{phase:len(shapes) for phase,shapes in phases.items()},
                    'geometric_merge_edits':geometry.get('merge_info',{}).get('merged_additions',0),
                    'VLM_tool_edits':evidence['applied_edits'],**evidence}
                summary['cases'].append(case)
                summary['completed_images']+=1
                for phase,shapes in phases.items():exports[phase].setdefault(topic,{})[str(frame)]=shapes
                inputs.setdefault(topic,{})[str(frame)]=records[topic,frame]['sha256']
                save(output/'summary.json',summary)
                print(json.dumps(case),flush=True)
        if summary['status']=='running':
            summary['full_agent_loops_completed']=bool(summary['cases'] and all(row['loop_completed'] for row in summary['cases']))
            summary['status']='complete' if summary['full_agent_loops_completed'] else 'incomplete'
        for phase,predictions in exports.items():
            file=output/(phase+'_candidates.json')
            save(file,predictions)
            save(output/(phase+'_receipt.json'),{'archive_sha256':manifest['archive_sha256'],
                'prediction_json_sha256':sha(file.read_bytes()),'image_sha256':inputs,'plan_sha256':summary['plan_sha256'],
                'source':'real_SAM3_tiled_union_and_production_VLM_Agent','phase':phase,'candidates_confirmed_as_truth':False})
    except Exception as error:
        summary.update(status='failed',error_type=type(error).__name__)
        (output/'local_error.txt').write_text(str(error),encoding='utf-8')
    finally:
        summary['finished_utc']=datetime.now(timezone.utc).isoformat()
        summary['plan_unchanged']=sha((output/'study_plan.json').read_bytes())==summary['plan_sha256']
        summary['source_images_unchanged']=all(sha(load_image(dataset,records[key]))==records[key]['sha256'] for key in [reference_key,*keys])
        save(output/'summary.json',summary)
    print(json.dumps({'status':summary['status'],'completed_images':summary['completed_images'],
        'full_agent_loops_completed':summary['full_agent_loops_completed']}),flush=True)
    return 0 if summary['status']=='complete' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--reference-json',required=True)
    parser.add_argument('--reference-kind',required=True,choices=['unconfirmed_functional_fixture','human_reviewed_2D_reference'])
    parser.add_argument('--reference-topic',default='/capture/camera_1/image')
    parser.add_argument('--reference-frame',type=int,default=0)
    parser.add_argument('--frames',default='76,119')
    parser.add_argument('--cameras',default='0,1,2,3')
    parser.add_argument('--merge-threshold',type=float,default=.5)
    parser.add_argument('--max-rounds',type=int,default=3)
    parser.add_argument('--base',default='http://127.0.0.1:18868/v1')
    parser.add_argument('--sam',default='http://127.0.0.1:8766')
    parser.add_argument('--model',default='labelagent-vlm')
    parser.add_argument('--output',required=True)
    raise SystemExit(evaluate(parser.parse_args()))
