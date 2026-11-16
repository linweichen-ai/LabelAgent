"""Fill missing responses for an existing frozen crop plan, without rerunning cache."""
import argparse,base64,hashlib,io,json,os,sys,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.crop_response_binding import check_crop_receipt
from tools.sam_busy_retry import post_when_available
from tools.tiled_candidate_utils import tile_grid
from tools.run_local_agent_acceptance import load_image,save
from tools.lidar_candidate_evidence import ReferenceReadAudit

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(args):
    dataset=Path(args.dataset).resolve();study=Path(args.study).resolve();output=Path(args.output).resolve()
    manifest=read(dataset/'import_manifest.json');plan=read(study/'study_plan.json');source_progress=read(study/'progress.json');plan_hash=sha(study/'study_plan.json')
    if plan_hash!=source_progress['plan_sha256'] or plan['manifest_sha256']!=sha(dataset/'import_manifest.json'):raise ValueError('Frozen plan differs')
    if plan['collector_sha256']!=sha(REPO/'tools/collect_tiled_candidates.py') or plan['postprocessor_sha256']!=sha(REPO/'tools/tiled_candidate_utils.py'):raise ValueError('Frozen collector changed')
    if plan['query']!='wooden pallet' or plan['planned_SAM3_requests']!=80:raise ValueError('This bounded recovery expects the frozen80 wooden-pallet plan')
    output.mkdir(parents=True,exist_ok=False)
    binding={str(p):sha(p) for p in (dataset/'import_manifest.json',study/'study_plan.json',REPO/'tools/sam_busy_retry.py',REPO/'tools/crop_response_binding.py')}
    save(output/'recovery_plan.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'code_sha256':sha(__file__),
        'input_bindings':binding,'source_plan_sha256':plan_hash,'sam_url':'http://127.0.0.1:8766',
        'retry_only':'503 with exact manager busy message, before activation','cached_successes_repeated':False,
        'query_or_crops_changed':False,'source_study_plan_rewritten':False})
    progress={'status':'running','pid':os.getpid(),'verified_responses':0,'cached_responses':0,'new_successful_requests':0,'busy_rejections':0}
    def report():save(output/'progress.json',progress)
    def stop():
        if any((p/'STOP').exists() for p in (output,study)):raise InterruptedError('Stop requested')
    def busy():stop();progress['busy_rejections']+=1;report()
    report();audit=ReferenceReadAudit(dataset);audit.start();opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        pairs={(f'/capture/camera_{c}/image',f) for c in plan['cameras'] for f in plan['frames']}
        rows=[r for r in manifest['records'] if (r['topic'],r['frame']) in pairs]
        if len(rows)!=plan['planned_images']:raise ValueError('Missing source image')
        for row in sorted(rows,key=lambda r:(r['frame'],r['topic'])):
            stop();image=Image.open(io.BytesIO(load_image(dataset,row))).convert('RGB')
            for index,bounds in enumerate(tile_grid(*image.size,plan['crop_fraction'],plan['grid_spec']['divisions'])):
                stop();buffer=io.BytesIO();image.crop(bounds).save(buffer,format='PNG');png=buffer.getvalue()
                expected={'topic':row['topic'],'frame':row['frame'],'tile_index':index,'bounds':list(bounds),
                    'source_image_sha256':row['sha256'],'crop_png_sha256':hashlib.sha256(png).hexdigest(),
                    'query':plan['query'],'plan_sha256':plan_hash}
                file=study/'responses'/f"{row['topic'].split('/')[2]}_{row['frame']:04d}_tile_{index}.json"
                if file.exists():check_crop_receipt(read(file),expected);progress['cached_responses']+=1
                else:
                    started=time.monotonic();body=json.dumps({'image':base64.b64encode(png).decode('ascii'),'prompt':plan['query']}).encode()
                    result=post_when_available(opener,'http://127.0.0.1:8766/segment_text',body,on_busy=busy)
                    value={**expected,'response':result,'seconds':round(time.monotonic()-started,3),
                        'cache_acquisition_code_sha256':sha(__file__),'cache_acquisition_plan_sha256':sha(output/'recovery_plan.json')}
                    check_crop_receipt(value,expected)
                    if file.exists():raise ValueError('Another writer created this response; do not overwrite')
                    save(file,value);progress['new_successful_requests']+=1
                progress['verified_responses']+=1;report()
        if progress['verified_responses']!=80:raise ValueError('Incomplete response coverage')
        if any(sha(p)!=h for p,h in binding.items()):raise ValueError('Frozen input changed')
        progress['status']='complete'
    except InterruptedError:progress['status']='stopped'
    except Exception as error:
        progress.update(status='failed',error_type=type(error).__name__);(output/'local_error.txt').write_text(repr(error),encoding='utf-8')
    finally:audit.stop();progress.update(reference_files_opened=len(audit.denied),source_plan_unchanged=sha(study/'study_plan.json')==plan_hash);report()
    print(json.dumps(progress),flush=True);return 0 if progress['status']=='complete' else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','study','output'):parser.add_argument('--'+name,required=True)
    raise SystemExit(run(parser.parse_args()))
