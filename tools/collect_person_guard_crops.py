"""Collect real loopback SAM3 person crop seeds into an isolated guard export.

No annotations or workbench writes. Successful responses are immutable caches;
uncertain missing responses block resume instead of repeating accepted work.
"""
import argparse,base64,hashlib,io,json,os,shutil,signal,sys,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path
import psutil
from PIL import Image
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.connect_local_vlm import loopback_url
from tools.run_local_agent_acceptance import load_image,save
from tools.tiled_candidate_utils import tile_grid
from tools.person_guard_crops import PURPOSE,check_person_guard_receipt,guard_pending_request,remap_person_guard_result
from tools.sam_busy_retry import post_when_available
from tools.lidar_candidate_evidence import ReferenceReadAudit

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()

def collect(args):
    dataset=Path(args.dataset).resolve();manifest_path=dataset/'import_manifest.json';manifest=read(manifest_path)
    baseline=Path(args.baseline_json).resolve();receipt=Path(args.baseline_receipt).resolve();binding=read(receipt)
    assert sha(baseline)==binding['prediction_json_sha256'] and binding['archive_sha256']==manifest['archive_sha256']
    frames=[int(v) for v in args.frames.split(',')];cameras=[int(v) for v in args.cameras.split(',')]
    if (not frames or not cameras or len(set(frames))!=len(frames) or len(set(cameras))!=len(cameras)
            or any(f not in range(len(manifest['timestamps_ns'])) for f in frames) or any(c not in range(4) for c in cameras)):
        raise ValueError('Distinct valid frames and cameras required')
    topics={f'/capture/camera_{c}/image' for c in cameras}
    rows=sorted([r for r in manifest['records'] if r['topic'] in topics and r['frame'] in frames],key=lambda r:(r['frame'],r['topic']))
    if len(rows)!=len(frames)*len(cameras):raise ValueError('Complete rectangular image cohort required')
    for row in rows:
        if binding['image_sha256'].get(row['topic'],{}).get(str(row['frame']))!=row['sha256']:raise ValueError('Unbound source image')
    endpoint=loopback_url(args.sam);workbench=loopback_url(args.workbench)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def get(url):
        with opener.open(url,timeout=8) as response:return json.load(response)
    if get(workbench+'/api/run/status')['status'] not in ('completed','idle'):raise RuntimeError('Workbench is active; defer this experiment')
    if get(endpoint+'/health')['status']!='ok':raise RuntimeError('SAM service is not healthy')
    output=Path(args.output).resolve()
    if output.is_relative_to(dataset):raise ValueError('Keep guard outputs outside the source dataset')
    parent=output.parent
    while not parent.exists():parent=parent.parent
    if shutil.disk_usage(parent).free<2*1024**3:raise RuntimeError('At least2GiB free on the experiment output disk required')
    files=[Path(__file__),manifest_path,baseline,receipt,*[REPO/'tools'/n for n in
        ('person_guard_crops.py','crop_response_binding.py','sam_busy_retry.py','tiled_candidate_utils.py','run_local_agent_acceptance.py','connect_local_vlm.py','lidar_candidate_evidence.py')],REPO/'source/agent/edit_agent.py']
    hashes={str(p):sha(p) for p in files}
    requested={'version':1,'frames':frames,'cameras':cameras,'planned_images':len(rows),'planned_requests':len(rows)*4,
        'query':'person','label':'person','purpose':PURPOSE,'grid_spec':{'divisions':2,'crop_fraction':.6},
        'minimum_score':.5,'internal_edge_fragments':'retained as guard seeds, not complete instance claims',
        'all_parts_and_holes_preserved':True,'nms_or_instance_budget':None,'sam_endpoint':endpoint,'workbench_endpoint':workbench,
        'input_and_code_bindings':hashes,'archive_sha256':manifest['archive_sha256'],
        'prediction_reads_reference':False,'workbench_writes':False,'new_VLM_calls':0,
        'scope':'New model person guard evidence on existing images, not annotation truth or accuracy'}
    if args.resume:
        plan=read(output/'study_plan.json')
        if any(plan.get(k)!=v for k,v in requested.items()):raise ValueError('Resume inputs, code or parameters changed')
        prior=read(output/'progress.json')
        if prior['status']=='complete':raise ValueError('Completed collection must not be repeated')
        try:
            p=psutil.Process(prior['pid'])
            if p.is_running() and p.create_time()==prior['process_create_time']:raise ValueError('Collection is already live')
        except psutil.NoSuchProcess:pass
        pending_path=output/'request_inflight.json'
        if pending_path.exists():
            pending=read(pending_path);name=pending['response_filename']
            if Path(name).name!=name:raise ValueError('Invalid pending response path')
            cached=output/'responses'/name
            if cached.exists():check_person_guard_receipt(read(cached),pending['expected'])
            guard_pending_request(pending,cached.exists())
    else:
        output.mkdir(parents=True,exist_ok=False);(output/'responses').mkdir()
        plan={**requested,'created_utc':utc()};save(output/'study_plan.json',plan)
    plan_sha=sha(output/'study_plan.json')
    state={'status':'running','pid':os.getpid(),'process_create_time':psutil.Process().create_time(),'planned_requests':plan['planned_requests'],
        'completed_requests':0,'completed_images':0,'new_requests_this_process':0,'cached_reused_this_process':0,'busy_rejections':0,
        'seed_shapes':0,'requests':[],'workbench_writes':False,'new_VLM_calls':0}
    def update(**kw):state.update(kw,observed_utc=utc());save(output/'progress.json',state)
    update();stop={'value':False}
    def request_stop(*_):stop['value']=True
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    def check_ready():
        if stop['value'] or (output/'STOP').exists():raise InterruptedError('Stop requested')
        if get(workbench+'/api/run/status')['status'] not in ('completed','idle'):raise InterruptedError('Defer to active workbench')
    audit=ReferenceReadAudit(dataset);audit.start();active={'value':True};denied=[]
    forbidden=[Path(p).resolve() for p in args.deny_reference_root]
    def guard(event,values):
        if event=='open' and active['value'] and values and isinstance(values[0],(str,bytes)):
            path=Path(values[0].decode() if isinstance(values[0],bytes) else values[0]).resolve()
            if any(path.is_relative_to(p) for p in forbidden):denied.append(str(path));raise PermissionError('Approved-reference input forbidden')
    sys.addaudithook(guard)
    maps={};images={};pending=None
    try:
        for row in rows:
            raw=load_image(dataset,row)
            with Image.open(io.BytesIO(raw)) as source:image=source.convert('RGB')
            width,height=image.size;topic=row['topic'];frame=str(row['frame']);camera=topic.split('/')[2];seeds=[]
            for tile,bounds in enumerate(tile_grid(width,height,.6,2)):
                check_ready();buffer=io.BytesIO();image.crop(bounds).save(buffer,format='PNG');png=buffer.getvalue()
                expected={'topic':topic,'frame':row['frame'],'tile_index':tile,'bounds':list(bounds),
                    'source_image_sha256':row['sha256'],'crop_png_sha256':hashlib.sha256(png).hexdigest(),'query':'person','plan_sha256':plan_sha}
                filename=f'{camera}_{row["frame"]:04d}_tile_{tile}.json';cached=output/'responses'/filename
                if cached.exists():
                    response=read(cached);check_person_guard_receipt(response,expected);state['cached_reused_this_process']+=1
                else:
                    pending={**expected,'expected':expected,'response_filename':filename,'status':'awaiting_response','started_utc':utc()}
                    save(output/'request_inflight.json',pending)
                    def busy():
                        state['busy_rejections']+=1;update()
                        try:check_ready()
                        except InterruptedError:
                            pending['status']='rejected_busy_before_activation';save(output/'request_inflight.json',pending);raise
                    started=time.monotonic()
                    body=json.dumps({'image':base64.b64encode(png).decode('ascii'),'prompt':'person'}).encode('utf-8')
                    try:payload=post_when_available(opener,endpoint+'/segment_text',body,timeout=600,busy_wait=300,pause=1,on_busy=busy)
                    except RuntimeError as error:
                        if str(error)=='Model manager remained busy; no accepted request was retried':
                            pending['status']='rejected_busy_before_activation';save(output/'request_inflight.json',pending)
                        raise
                    response={**expected,'label':'person','purpose':PURPOSE,'seconds':round(time.monotonic()-started,4),'response':payload}
                    check_person_guard_receipt(response,expected);save(cached,response)
                    pending['status']='complete';save(output/'request_inflight.json',pending);state['new_requests_this_process']+=1
                raw_count=len(response['response']['results']);kept=0
                for index,item in enumerate(response['response']['results']):
                    shape=remap_person_guard_result(item,bounds,width,height,f'guard-person-{camera}-{frame}-{tile}-{index}')
                    if shape['score']>=.5:seeds.append(shape);kept+=1
                state['completed_requests']+=1
                state['requests'].append({'frame':row['frame'],'camera':camera,'tile_index':tile,'raw_results':raw_count,'kept_seeds':kept,'seconds':response['seconds']})
                update()
            maps.setdefault(topic,{})[frame]=seeds;images.setdefault(topic,{})[frame]=row['sha256']
            state['completed_images']+=1;state['seed_shapes']+=len(seeds);update()
        assert state['completed_requests']==plan['planned_requests']
        file=output/'person_guard_seeds.json';save(file,maps)
        save(output/'person_guard_seeds_receipt.json',{'created_utc':utc(),'archive_sha256':manifest['archive_sha256'],
            'prediction_json_sha256':sha(file),'image_sha256':images,'plan_sha256':plan_sha,'purpose':PURPOSE,'label':'person',
            'baseline_prediction_sha256':binding['prediction_json_sha256'],'new_successful_model_responses':state['completed_requests'],
            'source':'real_local_SAM3_person_crops_guard_only','candidates_confirmed_as_truth':False})
        update(status='complete',images_with_seeds=sum(bool(v) for fs in maps.values() for v in fs.values()))
    except InterruptedError as error:update(status='stopped',stop_reason=str(error))
    except Exception as error:
        (output/'local_error.txt').write_text(str(error),encoding='utf-8');update(status='failed',error_type=type(error).__name__)
    finally:
        audit.stop();active['value']=False
        unchanged=all(sha(Path(p))==h for p,h in hashes.items()) and sha(output/'study_plan.json')==plan_sha
        state.update(input_code_and_plan_unchanged=unchanged,reference_files_opened=len(audit.denied),approved_reference_files_opened=len(denied))
        if not unchanged:state.update(status='failed',error_type='InputOrCodeChanged')
        update()
    print(json.dumps({k:v for k,v in state.items() if k!='requests'}),flush=True)
    return 0 if state['status']=='complete' else 2 if state['status']=='stopped' else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','baseline-json','baseline-receipt','output','frames'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--cameras',default='0,1,2,3');parser.add_argument('--sam',default='http://127.0.0.1:8766')
    parser.add_argument('--workbench',default='http://127.0.0.1:8765');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--deny-reference-root',action='append',default=[])
    raise SystemExit(collect(parser.parse_args()))
