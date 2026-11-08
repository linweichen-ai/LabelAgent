"""Bounded neutral reviews and paired native SAM captures of frozen grid cues."""
import base64,datetime,hashlib,io,json,os,shutil,sys,time,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
cfg=json.loads((ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json').read_text('utf-8'));sys.path.insert(0,cfg['dependency_dir'])
import numpy as np
from PIL import Image
from run_rgb_layers import read,save
from deliver_context_graph import sha
from paired_source_cue import body,external_live
from semantic_subject_guard import compatible_pallet_subject
from native_prompt_guard import evaluate
from prepare_fixed_grid_followup import BASE,OUT
SOURCE=Path('D:/Apps/LabelAgent/source/sam3_service/point_segment.py')
SOURCE_SHA='ab8861c57fbd275e8d16b5cc9bfbf734b3e4d7226d02b93d2b0ef9173fc49a81'

def initialize():
    p=read(OUT/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());file=OUT/'execution_plan.json'
    hashes={str(Path(__file__)):sha(__file__),str(Path(__file__).with_name('semantic_subject_guard.py')):sha(Path(__file__).with_name('semantic_subject_guard.py')),str(Path(__file__).with_name('native_prompt_guard.py')):sha(Path(__file__).with_name('native_prompt_guard.py'))}
    if not file.exists():save(file,{'at':datetime.datetime.now().astimezone().isoformat(),'source_plan_sha256':sha(OUT/'frozen_plan.json'),'protected':hashes,'max_new_neutral_real_reviews':32,'max_new_neutral_controls':2,'max_SAM_calls':32,'no_mask_edits':True})
    execution=read(file);assert execution['source_plan_sha256']==sha(OUT/'frozen_plan.json') and execution['protected']==hashes
    return p

def request_json(url,payload):
    for _ in range(1500):
        if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
        try:
            request=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=600) as f:return json.load(f)
        except urllib.error.HTTPError as e:
            if e.code==503:time.sleep(.2);continue
            raise
    raise RuntimeError('Shared model busy; preserve completed responses, resume missing only')

def review():
    if (OUT/'review_completed.json').exists():raise FileExistsError('Do not repeat completed neutral reviews')
    p=initialize();_,live=external_live()
    if live:raise RuntimeError('Named external GPU run is live; no competing activation')
    rows=[];fresh=0;seen=set()
    for r in p['controls']+p['rows']:
        assert all(sha(b['image'])==b['sha256'] for b in r['bindings']);payload=body(*[b['image'] for b in r['bindings']]);key=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();assert key==r['request_sha256'] and payload==read(r['request_file']);file=OUT/'responses'/(key+'.json')
        if not file.exists():
            if shutil.disk_usage(OUT).free<16*2**20:raise RuntimeError('16MiB reply reserve')
            reply=request_json('http://127.0.0.1:18868/v1/chat/completions',payload);save(file,reply);fresh+=1
        reply=read(file);assert reply['local_vlm']['model_config_sha256']==p['model_config_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256,256]
        decision=json.loads(reply['choices'][0]['message']['content']);row=r|{'decision':decision,'PAL_subject_evidence':compatible_pallet_subject(decision),'response_file':str(file),'response_sha256':sha(file),'same_batch_exact_reuse':key in seen};seen.add(key);rows.append(row);save(OUT/'review_progress.json',{'rows':rows,'new_calls_this_execution':fresh});print({'cue':r['index'],'object':decision['object'],'PAL_subject_evidence':row['PAL_subject_evidence'],'same_batch_reuse':row['same_batch_exact_reuse']},flush=True)
    controls=[r for r in rows if r.get('known_no_real_object')];assert len(controls)==2;controls_ok=not any(r['PAL_subject_evidence'] for r in controls)
    save(OUT/'review_completed.json',{'status':'complete_neutral_source_point_reviews','rows':rows,'actual_new_VLM_calls_total':len(seen),'new_calls_this_execution':fresh,'same_batch_exact_reuses':sum(r['same_batch_exact_reuse'] for r in rows),'plain_controls_reject_PAL':controls_ok,'not_complete_instance_semantics':True,'new_SAM_calls':0,'new_Agent_runs':0,'adopted':False})
    assert controls_ok,'Plain control failure; keep actual replies, do not admit SAM proposals'

def prepare_SAM():
    target=OUT/'SAM'
    if target.exists():raise FileExistsError('Preserve native paired SAM preparation')
    p=initialize();d=read(OUT/'review_completed.json');assert d['plain_controls_reject_PAL'];selected=[];seen=set();requests=[]
    for r in p['rows']:
        rr=next(v for v in d['rows'] if v['index']==r['index']);assert r['both_stock_panoptics_predict_PAL']
        if not rr['PAL_subject_evidence']:continue
        point_key=(r['case'],*r['source_point'])
        if point_key in seen:continue
        if len(selected)>=16:break
        seen.add(point_key);selected.append(r)
        for arm in ['original_source','explicit_padded_view']:
            if arm=='original_source':image=r['source_image'];point=r['source_point'];virtual=[0,0,1920,1536];observed=[0,0,1920,1536]
            else:image=r['padded_input'];point=r['input_point'];virtual=r['source_virtual_box'];observed=r['observed_region_in_input']
            payload={'image':base64.b64encode(Path(image).read_bytes()).decode(),'points':[{'x':point[0],'y':point[1],'label':1}],'include_raw_capture':True};key=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();f=target/'prepared_requests'/(key+'_PRIVATE.json');save(f,payload);requests.append(r|{'arm':arm,'input':image,'input_sha256':sha(image),'points':payload['points'],'virtual_box':virtual,'observed_rectangle':observed,'request_file':str(f),'request_sha256':key,'neutral_response':rr['response_file'],'neutral_response_sha256':rr['response_sha256']})
    assert len(requests)<=32 and sha(SOURCE)==SOURCE_SHA
    old=ROOT/'outputs/labelagent_registry_zoom_pool_20261108/SAM_weights_identity.json';w=read(old);weight=Path(w['path']);assert weight.stat().st_size==w['bytes'] and weight.stat().st_mtime_ns==w['mtime_ns']
    save(target/'weights_same_stat_binding.json',{'original_streamed_identity':str(old),'original_receipt_sha256':sha(old),'current_size_mtime_exact':True,'not_a_new_full_weight_hash':True})
    protected={str(OUT/'frozen_plan.json'):sha(OUT/'frozen_plan.json'),str(OUT/'review_completed.json'):sha(OUT/'review_completed.json'),str(Path(__file__)):sha(__file__),str(SOURCE):SOURCE_SHA,str(target/'weights_same_stat_binding.json'):sha(target/'weights_same_stat_binding.json')}
    for r in requests:protected[r['input']]=r['input_sha256'];protected[r['request_file']]=sha(r['request_file']);protected[r['neutral_response']]=r['neutral_response_sha256']
    save(target/'frozen_plan.json',{'rows':requests,'protected':protected,'max_unique_point_candidates':16,'actual_point_candidates':len(selected),'max_new_SAM_calls':len({r['request_sha256'] for r in requests}),'no_prior_mask':True,'source_selected_index_only':True,'save_complete_native_before_any_mapping':True,'reject_all_foreground_on_unobserved_padding':True,'all_final_gates_unchanged':True,'Agent_round_limit':3,'new_Agent_runs':0});print({'prepared_points':len(selected),'logical_paired_requests':len(requests),'max_unique_new_SAM_calls':len({r['request_sha256'] for r in requests})},flush=True)

def SAM():
    target=OUT/'SAM'
    if (target/'generation_completed.json').exists():raise FileExistsError('Do not repeat complete native paired captures')
    p=read(target/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());rows=[];fresh=0;seen=set()
    for r in p['rows']:
        assert sha(r['input'])==r['input_sha256'];payload=read(r['request_file']);assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==r['request_sha256'];folder=target/'captures'/r['request_sha256'];file=folder/'response.json'
        if not file.exists():
            if shutil.disk_usage(target).free<48*2**20:raise RuntimeError('48MiB complete-array reserve')
            d=request_json('http://127.0.0.1:8766/segment_points',payload);save(file,d);fresh+=1
        d=read(file);capture=d['raw_capture'];meta=capture['metadata'];assert meta['source_sha256']==SOURCE_SHA and meta['prior_mask_sha256'] is None;save(folder/'metadata.json',meta);rawbytes=base64.b64decode(capture['arrays_npz_base64']);(folder/'all_arrays.npz').write_bytes(rawbytes);arrays=np.load(io.BytesIO(rawbytes),allow_pickle=False);row=r|{'folder':str(folder),'response_sha256':sha(file),'metadata_file':str(folder/'metadata.json'),'raw_array_file':str(folder/'all_arrays.npz'),'valid':False,'same_batch_reuse':r['request_sha256'] in seen};seen.add(r['request_sha256'])
        if meta.get('raw_capture_complete') and meta.get('selected_index_observed'):
            raw=arrays[f"{meta['selected_prediction_call']}_masks"][meta['selected_index']].squeeze();assert np.isin(raw,[0,1]).all();native=(raw!=0).astype('uint8');photo=Image.open(r['input']);assert native.shape==(photo.height,photo.width);np.save(folder/'complete_native_mask.npy',native);Image.fromarray(native*255).save(folder/'native.png');guard=evaluate(native,r['points']);x1,y1,x2,y2=r['observed_rectangle'];outside=native.copy();outside[y1:y2,x1:x2]=0;pad_pixels=int(outside.sum());row.update(native_pixels=int(native.sum()),foreground_on_synthetic_padding=pad_pixels,native_prompt_guard=guard,native_index=meta['selected_index'],prediction_call=meta['selected_prediction_call'])
            # Test first, map later: never manufacture validity by deleting
            # foreground in unobserved padding or by clipping a mask budget.
            if pad_pixels==0:
                vx,vy,_,_=r['virtual_box'];ax,ay,bx,by=x1+vx,y1+vy,x2+vx,y2+vy;assert 0<=ax<bx<=1920 and 0<=ay<by<=1536;full=np.zeros((1536,1920),np.uint8);full[ay:by,ax:bx]=native[y1:y2,x1:x2];assert int(full.sum())==int(native.sum());np.save(folder/'complete_full_mask.npy',full);row.update(full_mask_sha256=hashlib.sha256(full.tobytes()).hexdigest(),all_native_foreground_preserved_in_source=True)
            row['valid']=meta['status']=='segment_returned' and not d.get('warnings') and guard['valid'] and pad_pixels==0
        rows.append(row);save(target/'progress.json',{'rows':rows,'new_SAM_calls_this_execution':fresh});print({'cue':r['index'],'arm':r['arm'],'native_pixels':row.get('native_pixels'),'padding_pixels':row.get('foreground_on_synthetic_padding'),'valid':row['valid']},flush=True)
    assert all(sha(f)==h for f,h in p['protected'].items());save(target/'generation_completed.json',{'status':'complete_native_paired_point_SAM_captures','rows':rows,'actual_new_SAM_calls_total':len(seen),'new_calls_this_execution':fresh,'logical_requests':len(rows),'same_batch_reuses':sum(r['same_batch_reuse'] for r in rows),'no_final_mask_clipping':True,'new_Agent_runs':0,'new_accepted_instances':0,'adopted':False})

if __name__=='__main__':{'review':review,'prepare_SAM':prepare_SAM,'SAM':SAM}[sys.argv[1]]()
