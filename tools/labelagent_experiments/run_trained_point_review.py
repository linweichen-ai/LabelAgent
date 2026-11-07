"""Frozen source-only part review for cues from unchanged trained outputs."""
import datetime,hashlib,json,os,shutil,sys,time,urllib.request,urllib.error
from pathlib import Path
from PIL import Image
from paired_source_cue import body,external_live,SHORT,old
from semantic_subject_guard import compatible_pallet_subject

BASE=Path(os.environ['LABELAGENT_TRAINED_POINTS_OUTPUT'])
OUT=Path(os.environ.get('LABELAGENT_TRAINED_REVIEW_OUTPUT',str(BASE/'semantic')))
PAIRED=Path(os.environ['LABELAGENT_PAIRED_CUE_OUTPUT'])

def key(b):return hashlib.sha256(json.dumps(b,sort_keys=True).encode()).hexdigest()

def prepare():
    if OUT.exists():raise FileExistsError('Preserve prepared review')
    if shutil.disk_usage(OUT.parent).free<64*2**20:raise RuntimeError('64MiB reserve before new study')
    prepared=old.read(BASE/'prepared.json');source=old.read(SHORT/'frozen_plan.json');previous=old.read(PAIRED/'frozen_plan.json');rows=[];protected={str(BASE/'prepared.json'):old.digest(BASE/'prepared.json'),str(BASE/'frozen_plan.json'):old.digest(BASE/'frozen_plan.json'),str(Path(__file__)):old.digest(__file__),str(Path(__file__).with_name('paired_source_cue.py')):old.digest(Path(__file__).with_name('paired_source_cue.py'))};OUT.mkdir()
    for i,r in enumerate(prepared['rows']):
        camera=int(r['case'].split('_')[0][6:]);photo_row=next(x for x in source['native_sources'] if x['camera']==camera and x['frame']==76);photo=Image.open(photo_row['path']).convert('RGB');protected[photo_row['path']]=old.digest(photo_row['path']);ax,ay=r['source_point'];folder=OUT/'inputs'/f'cue{i+1:02d}';folder.mkdir(parents=True);views=[];bindings=[]
        for name,side in [('near',129),('context',385)]:
            half=side//2;box=[ax-half,ay-half,ax+half+1,ay+half+1];clip=[max(0,box[0]),max(0,box[1]),min(photo.width,box[2]),min(photo.height,box[3])];canvas=Image.new('RGB',(side,side),(127,127,127));canvas.paste(photo.crop(tuple(clip)),(clip[0]-box[0],clip[1]-box[1]));file=folder/(name+'.png');canvas.resize((512,512),Image.Resampling.LANCZOS).save(file);views.append(str(file));bindings.append({'source_canvas_box':box,'clipped_box':clip,'side':side,'image':str(file),'sha256':old.digest(file)})
        request=body(*views);k=key(request);requestfile=OUT/'prepared_requests'/(k+'_PRIVATE.json');old.save(requestfile,request);rows.append({**r,'index':i+1,'source_image':photo_row['path'],'bindings':bindings,'request_file':str(requestfile),'request_sha256':k})
    old.save(OUT/'frozen_plan.json',{'at':datetime.datetime.now().astimezone().isoformat(),'rows':rows,'protected':protected,'model_config_sha256':previous['model_config_sha256'],'actual_image_tokens':[256,256],'same_prompt_schema_and_context_as_previous_matched_study':True,'point_generation_does_not_read_reference':True,'max_unique_new_calls':len({r['request_sha256'] for r in rows}),'original_Agent_round_limit':3,'new_Agent_rounds':0,'no_final_gates_changed':True});print({'prepared':len(rows),'new_model_calls':0},flush=True)

def run():
    if (OUT/'completed.json').exists():raise FileExistsError('Do not repeat completed review')
    status,live=external_live()
    if live:raise RuntimeError('External named model worker still running; no duplicate GPU worker')
    p=old.read(OUT/'frozen_plan.json');assert all(old.digest(f)==h for f,h in p['protected'].items());previous=old.read(PAIRED/'completed.json');cache={r['request_sha256']:r for r in previous['rows']};rows=[];fresh=0
    for r in p['rows']:
        assert all(old.digest(b['image'])==b['sha256'] for b in r['bindings']);request=body(*[b['image'] for b in r['bindings']]);assert key(request)==r['request_sha256'] and request==old.read(r['request_file']);file=OUT/'responses'/(r['request_sha256']+'.json');cached=False
        if not file.exists():
            if r['request_sha256'] in cache:
                oldrow=cache[r['request_sha256']];reply=old.read(oldrow['response']);assert old.digest(oldrow['response'])==oldrow['response_sha256'];assert old.read(oldrow['request_file'])==request;cached=True
            else:
                if shutil.disk_usage(OUT).free<8*2**20:raise RuntimeError('Keep raw reply reserve; resume only missing requests')
                for _ in range(1500):
                    if (BASE/'STOP').exists():raise RuntimeError('Explicit STOP')
                    try:
                        with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18868/v1/chat/completions',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'}),timeout=300) as f:reply=json.load(f)
                        fresh+=1;break
                    except urllib.error.HTTPError as e:
                        if e.code==503:time.sleep(.2);continue
                        raise
                else:raise RuntimeError('Shared service busy')
            assert reply['local_vlm']['model_config_sha256']==p['model_config_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256,256];old.save(file,{'raw_response':reply,'exact_prior_payload_reuse':cached})
        saved=old.read(file);reply=saved['raw_response'];assert reply['local_vlm']['model_config_sha256']==p['model_config_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256,256];decision=json.loads(reply['choices'][0]['message']['content']);row={**r,'response_file':str(file),'response_sha256':old.digest(file),'decision':decision,'PAL_subject_evidence':compatible_pallet_subject(decision),'exact_prior_payload_reuse':saved['exact_prior_payload_reuse']};rows.append(row);old.save(OUT/'progress.json',{'rows':rows,'actual_calls_this_execution':fresh});print({'index':r['index'],'teacher':r['teacher_key'],'side':r['side'],'object':decision['object'],'PAL_subject_evidence':row['PAL_subject_evidence'],'cached':row['exact_prior_payload_reuse']},flush=True)
    assert all(old.digest(f)==h for f,h in p['protected'].items());old.save(OUT/'completed.json',{'status':'complete_trained_cue_part_review','rows':rows,'actual_VLM_calls_total':sum(not r['exact_prior_payload_reuse'] for r in rows),'actual_calls_this_execution':fresh,'exact_prior_reuse':sum(r['exact_prior_payload_reuse'] for r in rows),'PAL_subject_cues':[r['index'] for r in rows if r['PAL_subject_evidence']],'new_SAM_calls':0,'new_masks':0,'new_Agent_runs':0,'not_final_instance_acceptance':True,'adopted':False})

if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
