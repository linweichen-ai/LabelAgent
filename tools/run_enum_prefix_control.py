"""Owned, bounded local GPU measurements; never mutates the shared service."""
import argparse,gc,hashlib,json,math,os,sys,time,traceback
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'tools')]
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(args):
    if sha(args.plan)!=args.plan_sha256:raise ValueError('Plan changed')
    plan=read(args.plan);assert all(sha(p)==h for p,h in plan['source_bindings'].items())
    from tools.local_vlm_server import configure_runtime_caches,arm_probe_deadline
    configure_runtime_caches(plan['cache_dir']);os.environ['CUDA_VISIBLE_DEVICES']=plan['gpu_uuid'];os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from tools.vlm_resources import inventory,require_local_gpu
    from tools.evaluate_confirmed_region_context import check_idle
    from tools.atomic_progress import save_json as save
    from tools.local_vlm_server import TransformersBackend,decode_messages,actual_image_encoding,require_runtime_probe_headroom
    from tools.enum_score_prefix_control import trace_generation,score_label,select_sequence
    import numpy as np,psutil,torch
    sys.path.insert(0,plan['task_dir']);from verify_source_last_20261107 import verify_protected
    protection_plan=read(plan['protection_plan']);verify_protected(protection_plan)
    check_idle();rows=inventory();selected=require_local_gpu(rows,plan['gpu_uuid'],plan['minimum_free_mib'])
    if selected['utilization_percent']>10:raise InterruptedError('Local GPU has other active compute')
    output=Path(args.output);output.mkdir(exist_ok=False)
    runtime=dict(status='loading',pid=os.getpid(),create_time=psutil.Process().create_time(),plan_sha256=args.plan_sha256,generate_calls=0,score_forwards=0,
        traces=[],score_rows=[],gpu_before=selected,maximum_generate_calls=1,maximum_score_forwards=108,original_annotations_written=False,default_enabled=False)
    save(output/'runtime.json',runtime);deadline=arm_probe_deadline(plan['deadline_seconds']);engine=None;started=time.monotonic()
    def before(identity):
        if (output/'STOP').exists():raise InterruptedError('Own diagnostic stop requested')
        check_idle();require_runtime_probe_headroom(torch,plan['memory_fraction'],plan['reserve_mib'])
        save(output/'operation_inflight_PRIVATE.json',identity)
    def after():
        (output/'operation_inflight_PRIVATE.json').unlink();save(output/'runtime.json',runtime)
    def prepare(job):
        assert sha(job['request_path'])==job['request_sha256'];body=read(job['request_path'])['body']
        messages,images=decode_messages(body['messages']);text=engine.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=engine.processor(text=[text],images=images,return_tensors='pt',padding=True);inputs.pop('token_type_ids',None)
        grid,counts=actual_image_encoding(inputs,engine.processor,len(images));assert counts==[256]*2
        assert len(inputs['input_ids'][0])==job['prompt_tokens'] and grid==job['image_grid_thw']
        return body,messages,images,inputs.to(engine.model.device)
    def generate(job,guard=False):
        identity=dict(kind='instrumented_generate',case_id=job['case_id'],view=job['view'],guard=guard)
        before(identity);body=read(job['request_path'])['body'];assert sha(job['request_path'])==job['request_sha256']
        messages,images=decode_messages(body['messages']);sink={};original,wrapped=trace_generation(engine.model,engine.processor.tokenizer,sink)
        engine.model.generate=wrapped
        try:content,prompt_tokens,tokens=engine.generate(messages,images,body['max_tokens'],response_schema=body['response_format']['json_schema']['schema'])
        finally:engine.model.generate=original
        runtime['generate_calls']+=1;assert runtime['generate_calls']<=1
        parsed=json.loads(content);assert parsed['category']==sink['category'] and set(parsed)=={'category','reason'}
        original_raw=read(job['response_path'])['raw_response'];previous=original_raw['choices'][0]['message']['content']
        name=f"trace_{runtime['generate_calls']:03d}";matrix=output/(name+'_PRIVATE.npz')
        np.savez_compressed(matrix,raw=sink.pop('raw'),processed=sink.pop('processed'))
        record=dict(**identity,request_path=job['request_path'],request_sha256=job['request_sha256'],content=content,parsed=parsed,
            same_text_as_previous=content==previous,same_category_as_previous=parsed['category']==json.loads(previous)['category'],prompt_tokens=prompt_tokens,
            completion_tokens=tokens,inference_stats=engine.last_inference_stats,model_config_sha256=engine.model_config_sha256,
            quantization=engine.quantization,device=str(engine.model.device),matrices=str(matrix),matrices_sha256=sha(matrix),trace=sink)
        ref=output/(name+'_PRIVATE.json');save(ref,record);runtime['traces'].append(str(ref));after()
        del record;torch.cuda.empty_cache()
    try:
        engine=TransformersBackend(plan['model_dir'],'labelagent-vlm',max_pixels=256*32*32,context_budget=4096,quantization='nf4',
            memory_fraction=plan['memory_fraction'],probe_reserve_mib=plan['reserve_mib'])
        assert engine.model_config_sha256==plan['model_config_sha256']
        runtime.update(status='running',model_memory_bytes=engine.model_memory_bytes,model_config_sha256=engine.model_config_sha256,
            quantization=engine.quantization,device=str(engine.model.device));save(output/'runtime.json',runtime)
        for job in plan['trace_jobs']:
            generate(job);print(json.dumps(dict(kind='trace_saved',calls=runtime['generate_calls'],case_id=job['case_id'],view=job['view'])),flush=True)
        for job in plan['score_jobs']:
            body,messages,images,inputs=prepare(job);scored={};refs=[]
            for label in plan['categories']:
                identity=dict(kind='teacher_forced_label_forward',case_id=job['case_id'],view=job['view'],label=label)
                before(identity);torch.cuda.reset_peak_memory_stats()
                result,logits=score_label(engine,inputs,label,body['response_format']['json_schema']['schema'])
                runtime['score_forwards']+=1;assert runtime['score_forwards']<=108
                name=f"score_{runtime['score_forwards']:03d}";matrix=output/(name+'_PRIVATE.npz');np.savez_compressed(matrix,logits=logits)
                result.update(identity,request_path=job['request_path'],request_sha256=job['request_sha256'],model_config_sha256=engine.model_config_sha256,
                    quantization=engine.quantization,device=str(engine.model.device),matrices=str(matrix),matrices_sha256=sha(matrix),peak_allocated_bytes=int(torch.cuda.max_memory_allocated()))
                ref=output/(name+'_PRIVATE.json');save(ref,result);refs.append(str(ref));scored[label]=result['statistics'];after()
                del logits,result
            row=dict(case_id=job['case_id'],view=job['view'],primary_raw_sequence_argmax=select_sequence(scored),secondary_schema_normalized_argmax=select_sequence(scored,'schema_normalized_sum'),scores=scored,receipts=refs)
            runtime['score_rows'].append(row);save(output/'runtime.json',runtime);del inputs;torch.cuda.empty_cache()
            print(json.dumps(dict(kind='sequence_scores_saved',conditions=len(runtime['score_rows']),primary=row['primary_raw_sequence_argmax'])),flush=True)
        generate(plan['guard_job'],True)
        assert runtime['generate_calls']==1 and runtime['score_forwards']==108 and len(runtime['score_rows'])==36
        assert all(sha(p)==h for p,h in plan['source_bindings'].items());verify_protected(protection_plan)
        runtime.update(status='complete',elapsed_seconds=round(time.monotonic()-started,3),source_bindings_unchanged=True,originals_and_references_unchanged=True)
    except InterruptedError as error:runtime.update(status='stopped',stop_reason=str(error))
    except Exception as error:
        runtime.update(status='failed',error_type=type(error).__name__);(output/'error_PRIVATE.txt').write_text(traceback.format_exc(),encoding='utf8')
    finally:
        engine=None;gc.collect();torch.cuda.empty_cache();runtime['own_allocator_bytes_before_exit']=int(torch.cuda.memory_allocated());deadline.cancel()
        save(output/'runtime.json',runtime);save(output/'summary_PRIVATE.json',runtime)
    print(json.dumps(dict(status=runtime['status'],generations=runtime['generate_calls'],score_forwards=runtime['score_forwards'])),flush=True)
    return 0 if runtime['status']=='complete' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',required=True)
    raise SystemExit(run(p.parse_args()))
