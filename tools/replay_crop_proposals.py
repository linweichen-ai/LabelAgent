"""Replay every bound nontruncated proposal from one complete SAM crop study."""
import argparse,hashlib,io,json,os,sys
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.crop_response_binding import check_crop_receipt
from tools.tiled_candidate_utils import remap_tile_result,tile_grid
from tools.lidar_candidate_evidence import ReferenceReadAudit
from tools.run_local_agent_acceptance import load_image,save

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())

def replay_image(seed,image,receipts,context):
    width,height=image.size;grid=context['grid_spec']
    bounds_list=tile_grid(width,height,grid['crop_fraction'],grid['divisions'])
    if len(receipts)!=len(bounds_list):raise ValueError('All crop responses required')
    values=deepcopy(seed);raw=0;rejected=0
    for index,(bounds,receipt) in enumerate(zip(bounds_list,receipts)):
        buffer=io.BytesIO();image.crop(bounds).save(buffer,format='PNG')
        expected={'topic':context['topic'],'frame':context['frame'],'tile_index':index,'bounds':list(bounds),
            'source_image_sha256':context['source_image_sha256'],'crop_png_sha256':hashlib.sha256(buffer.getvalue()).hexdigest(),
            'query':context['query'],'plan_sha256':context['plan_sha256']}
        check_crop_receipt(receipt,expected)
        for position,result in enumerate(receipt['response']['results']):
            raw+=1
            mapped=remap_tile_result(result,bounds,width,height,
                f"cropraw:{context['query'].replace(' ','_')}:{context['frame']}:{context['topic'].split('/')[2]}:{index}:{position}",
                context['internal_edge_margin_pixels'])
            if mapped is None:rejected+=1
            else:values.append(mapped)
    if len({s['id'] for s in values})!=len(values) or values[:len(seed)]!=seed:raise ValueError('Candidate prefix or identity changed')
    return values,{'raw_proposals':raw,'edge_rejected':rejected,'nontruncated_proposals':raw-rejected}

def run(args):
    dataset=Path(args.dataset).resolve();study=Path(args.study).resolve();output=Path(args.output).resolve()
    if output.is_relative_to(dataset):raise ValueError('Isolate output from source data')
    manifest=read(dataset/'import_manifest.json');source_plan=read(study/'study_plan.json');source_progress=read(study/'progress.json')
    baseline=read(args.candidates_json);receipt=read(args.prediction_receipt)
    if (source_progress['status']!='complete' or source_progress.get('plan_unchanged') is not True or source_progress.get('baseline_cache_unchanged') is not True
        or source_progress['completed_requests']!=source_plan['planned_SAM3_requests']
        or source_progress['completed_images']!=source_plan['planned_images'] or sha(study/'study_plan.json')!=source_progress['plan_sha256']):
        raise ValueError('Complete bound crop study required')
    if (source_plan['query']!=args.expected_prompt or source_plan['archive_sha256']!=manifest['archive_sha256']
        or receipt['archive_sha256']!=manifest['archive_sha256'] or sha(args.candidates_json)!=receipt['prediction_json_sha256']
        or source_plan['baseline_prediction_sha256']!=receipt['prediction_json_sha256']
        or source_plan['manifest_sha256']!=sha(dataset/'import_manifest.json')):raise ValueError('Source query or baseline differs')
    if source_plan['postprocessor_sha256']!=sha(REPO/'tools/tiled_candidate_utils.py'):raise ValueError('Crop remapping changed')
    grid=source_plan['grid_spec'];pairs={(f'/capture/camera_{c}/image',str(f)) for f in source_plan['frames'] for c in source_plan['cameras']}
    if len(pairs)!=source_plan['planned_images']:raise ValueError('Incomplete image cohort')
    bindings={str(Path(p).resolve()):sha(p) for p in (dataset/'import_manifest.json',args.candidates_json,args.prediction_receipt,study/'study_plan.json',study/'progress.json')}
    helpers={str(REPO/'tools'/n):sha(REPO/'tools'/n) for n in ('crop_response_binding.py','tiled_candidate_utils.py','run_local_agent_acceptance.py','lidar_candidate_evidence.py')}
    helpers[str(REPO/'source/agent/edit_agent.py')]=sha(REPO/'source/agent/edit_agent.py')
    output.mkdir(parents=True,exist_ok=False)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'code_sha256':sha(__file__),'helper_sha256':helpers,
        'input_bindings':bindings.copy(),'source_plan_sha256':source_progress['plan_sha256'],'query':args.expected_prompt,
        'grid_spec':grid,'planned_images':len(pairs),'planned_responses':source_plan['planned_SAM3_requests'],
        'per_image_addition_budget':None,'new_model_requests':0,'reference_inputs':False,'workbench_writes':False}
    save(output/'study_plan.json',plan);plan_hash=sha(output/'study_plan.json')
    progress={'status':'running','pid':os.getpid(),'completed_images':0,'verified_responses':0,'raw_proposals':0,'edge_rejected':0,'nontruncated_proposals':0,'new_model_requests':0}
    save(output/'progress.json',progress);audit=ReferenceReadAudit(dataset);audit.start()
    try:
        records={(r['topic'],str(r['frame'])):r for r in manifest['records'] if r['topic'].endswith('/image')}
        result={};images={}
        for topic,key in sorted(pairs,key=lambda p:(int(p[1]),p[0])):
            if (output/'STOP').exists():raise InterruptedError('Stop requested')
            row=records[topic,key]
            if receipt['image_sha256'][topic][key]!=row['sha256']:raise ValueError('Source image binding differs')
            image=Image.open(io.BytesIO(load_image(dataset,row))).convert('RGB');receipts=[]
            for index in range(grid['divisions']**2):
                file=study/'responses'/f"{topic.split('/')[2]}_{int(key):04d}_tile_{index}.json"
                bindings[str(file)]=sha(file);receipts.append(read(file))
            values,stats=replay_image(baseline[topic][key],image,receipts,{'topic':topic,'frame':int(key),
                'source_image_sha256':row['sha256'],'query':source_plan['query'],'plan_sha256':source_progress['plan_sha256'],
                'grid_spec':grid,'internal_edge_margin_pixels':source_plan['internal_edge_margin_pixels']})
            result.setdefault(topic,{})[key]=values;images.setdefault(topic,{})[key]=row['sha256']
            progress['completed_images']+=1;progress['verified_responses']+=len(receipts)
            for k,v in stats.items():progress[k]+=v
            save(output/'progress.json',progress)
        if progress['verified_responses']!=source_plan['planned_SAM3_requests']:raise ValueError('Response coverage incomplete')
        if any(sha(p)!=h for p,h in bindings.items()) or any(sha(p)!=h for p,h in helpers.items()):raise ValueError('Input changed')
        file=output/'all_crop_proposals_candidates.json';save(file,result);save(output/'response_bindings.json',bindings)
        save(output/'all_crop_proposals_receipt.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
            'archive_sha256':manifest['archive_sha256'],'prediction_json_sha256':sha(file),'image_sha256':images,
            'plan_sha256':plan_hash,'source':'all_bound_nontruncated_crop_proposals','query':source_plan['query'],
            'new_model_requests':0,'candidates_confirmed_as_truth':False})
        progress.update(status='complete',all_original_prefixes_exact=True,plan_sha256=plan_hash)
    except InterruptedError:progress['status']='stopped'
    except Exception as error:
        progress.update(status='failed',error_type=type(error).__name__);(output/'local_error.txt').write_text(str(error),encoding='utf-8')
    finally:
        audit.stop();progress.update(reference_files_opened=len(audit.denied),input_bindings_unchanged=all(sha(p)==h for p,h in bindings.items()),
            code_unchanged=sha(__file__)==plan['code_sha256'] and all(sha(p)==h for p,h in helpers.items()),plan_unchanged=sha(output/'study_plan.json')==plan_hash)
        save(output/'progress.json',progress)
    print(json.dumps(progress),flush=True);return 0 if progress['status']=='complete' else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','study','candidates-json','prediction-receipt','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--expected-prompt',required=True,choices=('pallet','wooden pallet','plastic pallet','pallet base'))
    raise SystemExit(run(parser.parse_args()))
