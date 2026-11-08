"""Unchanged neutral gate, paired source-negative prompts, strict native capture."""
import base64,datetime,hashlib,io,json,os,shutil,sys
from pathlib import Path
from prepare_balanced_palette_parts import OUT
from prepare_palette_parts import ROOT,BASE
import numpy as np
from PIL import Image
from run_rgb_layers import read,save
from deliver_context_graph import sha
from semantic_subject_guard import compatible_pallet_subject
from native_prompt_guard import evaluate
import run_fixed_grid_followup as api

def initialize():
    p=read(OUT/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());file=OUT/'execution_plan.json';protected={str(Path(__file__)):sha(__file__),str(Path(api.__file__)):sha(api.__file__)}
    if not file.exists():save(file,{'protected':protected,'source_plan_sha256':sha(OUT/'frozen_plan.json'),'max_positive_reviews':32,'max_SAM_calls':32,'final_native_border_test_required':True})
    e=read(file);assert e['protected']==protected and e['source_plan_sha256']==sha(OUT/'frozen_plan.json');api.OUT=OUT;return p

def review():
    if (OUT/'review_completed.json').exists():raise FileExistsError('Do not repeat completed palette reviews')
    p=initialize();_,live=api.external_live()
    if live:raise RuntimeError('Named external GPU run still active')
    catalogue={}
    for folder in ['labelagent_fixed_grid_followup_20261108','labelagent_single_teacher_grid_20261108']:
        f=BASE/folder/'review_completed.json'
        for r in read(f)['rows']:catalogue[r['request_sha256']]=r
    cache=[];rows=[];seen=set();fresh=0
    for r in p['controls']+p['rows']:
        payload=api.body(*[b['image'] for b in r['bindings']]);key=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();assert key==r['request_sha256'] and payload==read(r['request_file']);file=OUT/'responses'/(key+'.json');cached=False
        if not file.exists():
            previous=catalogue.get(key)
            if previous:
                old=Path(previous['response_file']);assert sha(old)==previous['response_sha256'] and read(previous['request_file'])==payload;reply=read(old);cached=True;cache.append({'request_sha256':key,'source_response':str(old),'source_response_sha256':sha(old)})
            else:
                if shutil.disk_usage(OUT).free<16*2**20:raise RuntimeError('Original16MiB reply reserve')
                reply=api.request_json('http://127.0.0.1:18868/v1/chat/completions',payload);fresh+=1
            save(file,{'raw_response':reply,'historical_exact_reuse':cached})
        saved=read(file);reply=saved['raw_response'];assert reply['local_vlm']['model_config_sha256']==p['model_config_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256,256];decision=json.loads(reply['choices'][0]['message']['content']);rows.append(r|{'decision':decision,'PAL_subject_evidence':compatible_pallet_subject(decision),'response_file':str(file),'response_sha256':sha(file),'historical_exact_reuse':saved['historical_exact_reuse'],'same_batch_reuse':key in seen});seen.add(key);save(OUT/'review_progress.json',{'rows':rows,'new_calls_this_execution':fresh});print({'cue':r['index'],'object':decision['object'],'PAL_subject_evidence':rows[-1]['PAL_subject_evidence'],'historical_reuse':rows[-1]['historical_exact_reuse']},flush=True)
    ok=not any(r['PAL_subject_evidence'] for r in rows if r.get('known_no_real_object'));save(OUT/'review_completed.json',{'status':'complete_neutral_palette_source_points','rows':rows,'actual_new_VLM_calls_total':len({r['request_sha256'] for r in rows if not r['historical_exact_reuse']}),'exact_historical_VLM_reuses':len({r['request_sha256'] for r in rows if r['historical_exact_reuse']}),'new_calls_this_execution':fresh,'plain_controls_reject_PAL':ok,'negative_component_physical_identity_not_reviewed_or_asserted':True,'new_Agent_runs':0,'adopted':False});save(OUT/'neutral_exact_prior_cache.json',{'rows':cache});assert ok

def prepare_SAM():
    target=OUT/'SAM'
    if target.exists():raise FileExistsError('Preserve paired component-negative plan')
    if shutil.disk_usage(OUT).free<96*2**20:raise RuntimeError('Original96MiB preparation reserve')
    p=initialize();rv=read(OUT/'review_completed.json');assert rv['plain_controls_reject_PAL'];assert sha(api.SOURCE)==api.SOURCE_SHA;wfile=ROOT/'outputs/labelagent_registry_zoom_pool_20261108/SAM_weights_identity.json';w=read(wfile);weight=Path(w['path']);assert weight.stat().st_size==w['bytes'] and weight.stat().st_mtime_ns==w['mtime_ns'];history={}
    for name in ['labelagent_fixed_grid_followup_20261108','labelagent_single_teacher_grid_20261108']:
        for r in read(BASE/name/'SAM/generation_completed.json')['rows']:history[r['request_sha256']]=r
    selected=[];requests=[];protected={str(OUT/'frozen_plan.json'):sha(OUT/'frozen_plan.json'),str(OUT/'review_completed.json'):sha(OUT/'review_completed.json'),str(Path(__file__)):sha(__file__),str(api.SOURCE):api.SOURCE_SHA}
    for r in p['rows']:
        rr=next(v for v in rv['rows'] if v['index']==r['index'])
        if not rr['PAL_subject_evidence']:continue
        if len(selected)>=16:break
        selected.append(r);arms=['one_positive']+(['source_component_negatives'] if r['negative_hypotheses'] else [])
        for arm in arms:
            positive={'x':r['source_point'][0],'y':r['source_point'][1],'label':1};points=[positive]+(r['negative_hypotheses'] if arm!='one_positive' else []);payload={'image':base64.b64encode(Path(r['source_image']).read_bytes()).decode(),'points':points,'include_raw_capture':True};key=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();file=target/'prepared_requests'/(key+'_PRIVATE.json');save(file,payload);old=history.get(key);cached=None
            if old:
                response=Path(old['folder'])/'response.json';d=read(response);meta=d.get('raw_capture',{}).get('metadata',{});assert old['points']==points and sha(old['input'])==sha(r['source_image'])
                if meta.get('source_sha256')==api.SOURCE_SHA and meta.get('prior_mask_sha256') is None and meta.get('raw_capture_complete') and meta.get('selected_index_observed'):cached=str(response);protected[cached]=sha(cached)
            requests.append(r|{'arm':arm,'input':r['source_image'],'input_sha256':sha(r['source_image']),'points':points,'virtual_box':[0,0,1920,1536],'observed_rectangle':[0,0,1920,1536],'request_file':str(file),'request_sha256':key,'exact_previous_capture':cached,'neutral_response':rr['response_file'],'neutral_response_sha256':rr['response_sha256']});protected[r['source_image']]=sha(r['source_image']);protected[str(file)]=sha(file);protected[rr['response_file']]=rr['response_sha256']
    save(target/'weights_same_stat_binding.json',{'original_streamed_receipt':str(wfile),'receipt_sha256':sha(wfile),'current_size_and_mtime_exact':True,'not_a_new_full_weight_hash':True});protected[str(target/'weights_same_stat_binding.json')]=sha(target/'weights_same_stat_binding.json');assert len(requests)<=32;save(target/'frozen_plan.json',{'rows':requests,'protected':protected,'actual_point_candidates':len(selected),'max_new_SAM_calls':len({r['request_sha256'] for r in requests if not r['exact_previous_capture']}),'source_component_negatives_are_unconfirmed_identity_hypotheses':True,'same_RGB_same_positive_only_negatives_change_within_pair':True,'no_prior_mask':True,'all_native_arrays_and_source_selected_output_preserved':True,'original_final_input_border_and_point_constraints_explicit':True,'all_final_gates_unchanged':True,'new_Agent_runs':0,'Agent_round_limit':3});print({'neutral_admitted_points':len(selected),'logical_SAM_requests':len(requests),'max_new_SAM_calls':read(target/'frozen_plan.json')['max_new_SAM_calls']},flush=True)

def SAM():
    target=OUT/'SAM'
    if (target/'generation_completed.json').exists():raise FileExistsError('Never repeat completed source-negative trial')
    p=read(target/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());rows=[];seen=set();fresh=0
    for r in p['rows']:
        folder=target/'captures'/r['request_sha256'];file=folder/'response.json';payload=read(r['request_file']);assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==r['request_sha256']
        if not file.exists():
            if shutil.disk_usage(OUT).free<48*2**20:raise RuntimeError('Original48MiB raw array reserve')
            if r['exact_previous_capture']:folder.mkdir(parents=True,exist_ok=True);shutil.copy2(r['exact_previous_capture'],file)
            else:d=api.request_json('http://127.0.0.1:8766/segment_points',payload);save(file,d);fresh+=1
        d=read(file);capture=d['raw_capture'];meta=capture['metadata'];assert meta['source_sha256']==api.SOURCE_SHA and meta['prior_mask_sha256'] is None and meta['image_sha256']==r['input_sha256'];save(folder/'metadata.json',meta);rawbytes=base64.b64decode(capture['arrays_npz_base64']);(folder/'all_arrays.npz').write_bytes(rawbytes);a=np.load(io.BytesIO(rawbytes),allow_pickle=False);row=r|{'folder':str(folder),'metadata_file':str(folder/'metadata.json'),'raw_array_file':str(folder/'all_arrays.npz'),'response_sha256':sha(file),'valid':False,'same_batch_reuse':r['request_sha256'] in seen,'historical_exact_SAM_reuse':r['exact_previous_capture'] is not None};seen.add(r['request_sha256'])
        if meta.get('raw_capture_complete') and meta.get('selected_index_observed'):
            raw=a[f"{meta['selected_prediction_call']}_masks"][meta['selected_index']].squeeze();assert raw.shape==(1536,1920) and np.isin(raw,[0,1]).all();m=(raw!=0).astype('uint8');np.save(folder/'complete_native_mask.npy',m);np.save(folder/'complete_full_mask.npy',m);Image.fromarray(m*255).save(folder/'native.png');guard=evaluate(m,r['points']);border=bool(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any());row.update(native_pixels=int(m.sum()),native_index=meta['selected_index'],prediction_call=meta['selected_prediction_call'],native_prompt_guard=guard,touches_input_border=border,foreground_on_synthetic_padding=0,full_mask_sha256=hashlib.sha256(m.tobytes()).hexdigest(),all_native_foreground_preserved_in_source=True,valid=meta['status']=='segment_returned' and not d.get('warnings') and guard['valid'] and not border)
        rows.append(row);save(target/'progress.json',{'rows':rows,'new_calls_this_execution':fresh});print({'cue':r['index'],'arm':r['arm'],'pixels':row.get('native_pixels'),'input_border':row.get('touches_input_border'),'points_valid':row.get('native_prompt_guard',{}).get('valid'),'valid':row['valid']},flush=True)
    assert all(sha(f)==h for f,h in p['protected'].items());save(target/'generation_completed.json',{'status':'complete_strict_native_source_component_negative_trial','rows':rows,'actual_new_SAM_calls_total':len({r['request_sha256'] for r in rows if not r['historical_exact_SAM_reuse']}),'exact_historical_SAM_reuses':len({r['request_sha256'] for r in rows if r['historical_exact_SAM_reuse']}),'new_calls_this_execution':fresh,'all_input_border_tests_explicit':True,'no_final_mask_cut':True,'new_accepted_instances':0,'new_Agent_runs':0,'adopted':False})

def main():
    api.OUT=OUT;phase=sys.argv[1];file=OUT/(phase+'_runtime_process.json');import psutil
    if file.exists():
        old=read(file)
        try:
            process=psutil.Process(old['pid']);live=process.is_running() and abs(process.create_time()-old['create_time'])<.01
        except psutil.NoSuchProcess:live=False
        if live:raise RuntimeError('Exact phase is live; poll original handle')
    state={'pid':os.getpid(),'create_time':psutil.Process().create_time(),'phase':phase,'status':'running','at':datetime.datetime.now().astimezone().isoformat()};save(file,state)
    try:{'review':review,'prepare_SAM':prepare_SAM,'SAM':SAM}[phase]();state['status']='complete'
    except BaseException as error:state.update(status='failed',error=str(error));raise
    finally:save(file,state|{'last_at':datetime.datetime.now().astimezone().isoformat()})

if __name__=='__main__':main()
