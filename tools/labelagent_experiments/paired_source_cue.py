"""Matched clean-neighborhood/context proposal test, after the other run ends."""
import base64,datetime,hashlib,io,json,os,shutil,sys,time,urllib.request,urllib.error
from pathlib import Path
import numpy as np
from PIL import Image
from run_pallet_chain import ROOT,old
from semantic_subject_guard import compatible_pallet_subject

SHORT=Path(os.environ.get('LABELAGENT_SHORT_CHAIN_SOURCE',str(ROOT/'outputs/labelagent_chain_rgb_20261108')))
COUNTER=Path(os.environ.get('LABELAGENT_COUNTER_OUTPUT',str(ROOT/'outputs/labelagent_counter_material_20261108')))
OUT=Path(os.environ.get('LABELAGENT_PAIRED_CUE_OUTPUT',str(ROOT/'outputs/labelagent_paired_source_cue_20261108')))
EXTERNAL=Path('D:/Apps/LabelAgent/work/experiments/labelagent_projection_20261104/native_source_pair_control_20261108_v1')
PROMPT=('Two unmarked source photographs show the SAME center pixel. Describe the actual physical object or surface AT that pixel. '
        'Use wider structure only when visibly connected to that selected source component; do not borrow the identity of a separate nearby object. '
        'State concrete material, structure and color in one short English sentence. A small low-texture component in a real object scene '
        'is not by itself an empty scene. If both images are uniform without an actual object, describe that uniform image. '
        'Do not infer an object from the task or a class list. Return exactly one JSON key object.')
SCHEMA={'type':'object','properties':{'object':{'type':'string','minLength':1,'maxLength':200}},'required':['object'],'additionalProperties':False}


def png(im):
    b=io.BytesIO();im.save(b,'PNG');return b.getvalue()


def body(first,second):
    return {'model':'labelagent-vlm','temperature':0,'max_tokens':256,'messages':[{'role':'user','content':[{'type':'text','text':PROMPT}]+[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(Path(p).read_bytes()).decode()}} for p in [first,second]]}],'response_format':{'type':'json_schema','json_schema':{'name':'labelagent','schema':SCHEMA,'strict':True}}}


def prepare():
    if OUT.exists():raise FileExistsError('Preserve matched point/context trial')
    if shutil.disk_usage(OUT.parent).free<256*2**20:raise RuntimeError('Output space guard')
    s=old.read(SHORT/'frozen_plan.json');counter=old.read(COUNTER/'frozen_plan.json');model=old.read(SHORT/'semantic/frozen_plan.json')['model_config_sha256'];protected={str(SHORT/'frozen_plan.json'):old.digest(SHORT/'frozen_plan.json'),str(COUNTER/'frozen_plan.json'):old.digest(COUNTER/'frozen_plan.json'),str(Path(__file__)):old.digest(__file__)};cases=[]
    for seed in s['seeds']:
        camera=int(seed['case'].split('_')[0][6:]);source=next(x for x in s['native_sources'] if x['camera']==camera and x['frame']==76);cases.append({'name':f"original_seed{seed['index']:02d}",'kind':'original_seed','seed':seed['index'],'source':source['path'],'point':seed['initial_points'][0],'known_class_truth':False})
    for r in counter['proposals']:cases.append({'name':f"corner_seed{r['seed']:02d}_"+r['cue'],'kind':'counter_positive_cue','seed':r['seed'],'cue':r['cue'],'source':r['source_image'],'point':r['source_point'],'original_counter_key':r['key'],'known_class_truth':False})
    controls=ROOT/'outputs/labelagent_semantic_recovery_20261105/neutral_controls'
    for name in ['wood_pallet_base','raw_robot','raw_person','raw_floor','gray','white']:
        p=controls/(name+'.png');im=Image.open(p);cases.append({'name':'control_'+name,'kind':'control','source':str(p),'point':[im.width//2,im.height//2],'expected_PAL':name=='wood_pallet_base','expected_uniform':name in ['gray','white'],'derived_control_image_not_native_sensor_scale':True})
    OUT.mkdir();requests=[]
    for case in cases:
        source=Path(case['source']);protected[str(source)]=old.digest(source);photo=Image.open(source).convert('RGB');ax,ay=[int(round(x)) for x in case['point']];folder=OUT/'inputs'/case['name'];folder.mkdir(parents=True);views={};bindings=[]
        for name,side in [('near',129),('context',385)]:
            half=side//2;box=[ax-half,ay-half,ax+half+1,ay+half+1];clip=[max(0,box[0]),max(0,box[1]),min(photo.width,box[2]),min(photo.height,box[3])];canvas=Image.new('RGB',(side,side),(127,127,127));canvas.paste(photo.crop(tuple(clip)),(clip[0]-box[0],clip[1]-box[1]));file=folder/(name+'.png');canvas.resize((512,512),Image.Resampling.LANCZOS).save(file);views[name]=str(file);bindings.append({'view':name,'source_canvas_box':box,'clipped_box':clip,'source_side':side,'same_center_source_point':[ax,ay],'output_image':[512,512],'resampling_does_not_add_detail':True})
        for arm,second in [('repeated_near',views['near']),('near_plus_context',views['context'])]:
            request=body(views['near'],second);key=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest();requestfile=OUT/'prepared_requests'/(key+'_PRIVATE.json');old.save(requestfile,request);requests.append({**case,'arm':arm,'first_image':views['near'],'second_image':second,'first_sha256':old.digest(views['near']),'second_sha256':old.digest(second),'bindings':bindings,'request_sha256':key,'request_file':str(requestfile)})
    for a,b in zip(requests[::2],requests[1::2]):
        ra=old.read(a['request_file']);rb=old.read(b['request_file']);ra['messages'][0]['content'][2]=rb['messages'][0]['content'][2];assert ra==rb and a['first_sha256']==b['first_sha256']
    old.save(OUT/'frozen_plan.json',{'frozen_at':datetime.datetime.now().astimezone().isoformat(),'requests':requests,'protected':protected,'model_config_sha256':model,'cases':len(cases),'logical_conditions':len(requests),'max_unique_new_requests':len({r['request_sha256'] for r in requests}),'same_prompt_schema_weights_decoding_and_two_images_per_arm':True,'only_second_image_changes_within_pair':True,'expected_actual_image_token_counts':[256,256],'visual_tokens_each_arm':512,'known_real_point_truth_not_inferred_from_old_caption':True,'proposal_test_not_final_mask_acceptance':True,'source_original_negative_labels_still_unconfirmed':True,'no_reference_labels_points_pose_depth_calibration_or_area_inputs':True,'new_Agent_rounds':0,'Agent_round_limit':3,'all_final_gates_unchanged':True,'external_run_guard_before_network':str(EXTERNAL)});print({'prepared_cases':len(cases),'logical_conditions':len(requests),'unique_requests':len({r['request_sha256'] for r in requests}),'GPU_requests':0},flush=True)


def external_live():
    import psutil
    state=old.read(EXTERNAL/'supervisor.json');live=[]
    for pidkey,timekey in [('pid','create_time'),('child_pid','child_create_time')]:
        if not isinstance(state.get(pidkey),int) or not isinstance(state.get(timekey),(int,float)):
            live.append(False);continue
        try:
            p=psutil.Process(int(state[pidkey]));live.append(p.is_running() and abs(p.create_time()-float(state[timekey]))<.01)
        except (KeyError,psutil.NoSuchProcess):live.append(False)
    return state['status'],any(live)


def run():
    if (OUT/'completed.json').exists():raise FileExistsError('Do not repeat completed matched captions')
    status,live=external_live()
    if live:print({'waiting_external_run':True,'status':status,'new_requests':0},flush=True);raise SystemExit(75)
    plan=old.read(OUT/'frozen_plan.json');assert all(old.digest(p)==h for p,h in plan['protected'].items());rows=[];fresh=0;seen=set()
    for r in plan['requests']:
        assert old.digest(r['first_image'])==r['first_sha256'] and old.digest(r['second_image'])==r['second_sha256'];request=body(r['first_image'],r['second_image']);assert hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()==r['request_sha256'];file=OUT/'responses'/(r['request_sha256']+'.json')
        if not file.exists():
            for _ in range(1500):
                if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
                try:
                    with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18868/v1/chat/completions',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'}),timeout=300) as f:reply=json.load(f)
                    old.save(file,reply);fresh+=1;break
                except urllib.error.HTTPError as e:
                    if e.code==503:time.sleep(.2);continue
                    raise
            else:raise RuntimeError('Shared manager busy; resume missing replies only')
        reply=old.read(file);assert reply['local_vlm']['model_config_sha256']==plan['model_config_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256,256];decision=json.loads(reply['choices'][0]['message']['content']);row={**r,'response':str(file),'response_sha256':old.digest(file),'decision':decision,'PAL_guard':compatible_pallet_subject(decision),'exact_same_batch_reuse':r['request_sha256'] in seen};seen.add(r['request_sha256']);rows.append(row);old.save(OUT/'progress.json',{'rows':rows,'unique_saved_replies':len(seen),'new_requests_this_execution':fresh});print({'case':r['name'],'arm':r['arm'],'object':decision['object'],'PAL_guard':row['PAL_guard'],'reused':row['exact_same_batch_reuse']},flush=True)
    assert all(old.digest(p)==h for p,h in plan['protected'].items());old.save(OUT/'completed.json',{'status':'complete_matched_point_context_captions','rows':rows,'logical_conditions':len(rows),'unique_model_calls_total':len(seen),'new_requests_this_execution':fresh,'exact_reused_logical_conditions':sum(r['exact_same_batch_reuse'] for r in rows),'new_SAM_calls':0,'new_Agent_runs':0,'adopted':False,'no_real_point_truth_or_complete_instance_claim':True})


if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
