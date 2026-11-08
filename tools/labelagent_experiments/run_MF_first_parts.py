"""Execute frozen MF-first proposals; every native pixel and final gate survives."""
import base64,datetime,hashlib,io,json,os,shutil,sys,urllib.error
from pathlib import Path
from prepare_mf_first_parts import OUT
from prepare_palette_parts import ROOT,BASE
import numpy as np
from PIL import Image
from run_rgb_layers import read,save
from deliver_context_graph import sha
from native_review_feedback import describe_native,SOURCE_SHA
import run_fixed_grid_followup as api

def initialize():
    p=read(OUT/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());parent=read(BASE/'labelagent_balanced_palette_parts_20261108/frozen_plan.json');assert all(sha(f)==h for f,h in parent['protected'].items());assert sha(api.SOURCE)==SOURCE_SHA;file=OUT/'execution_plan.json'
    if not file.exists():
        wfile=ROOT/'outputs/labelagent_registry_zoom_pool_20261108/SAM_weights_identity.json';w=read(wfile);weight=Path(w['path']);assert weight.stat().st_size==w['bytes'] and weight.stat().st_mtime_ns==w['mtime_ns'];catalogue={};sources={}
        for name in ['labelagent_fixed_grid_followup_20261108','labelagent_single_teacher_grid_20261108','labelagent_balanced_palette_parts_20261108','labelagent_concise_palette_candidates_20261108']:
            path=BASE/name/'SAM/generation_completed.json';sources[str(path)]=sha(path)
            for row in read(path)['rows']:
                response=Path(row['folder'])/'response.json';d=read(response);meta=d.get('raw_capture',{}).get('metadata',{})
                if meta.get('source_sha256')==SOURCE_SHA and meta.get('prior_mask_sha256') is None and meta.get('raw_capture_complete') and meta.get('selected_index_observed'):
                    catalogue[row['request_sha256']]={'response':str(response),'response_sha256':sha(response),'input_sha256':row['input_sha256'],'points':row['points']}
        protected={str(Path(__file__)):sha(__file__),str(Path(__file__).with_name('native_review_feedback.py')):sha(Path(__file__).with_name('native_review_feedback.py')),str(Path(__file__).with_name('native_prompt_guard.py')):sha(Path(__file__).with_name('native_prompt_guard.py')),str(Path(api.__file__)):sha(api.__file__),str(api.SOURCE):SOURCE_SHA,str(wfile):sha(wfile)};save(file,{'at':datetime.datetime.now().astimezone().isoformat(),'source_plan_sha256':sha(OUT/'frozen_plan.json'),'protected':protected,'cache':catalogue,'cache_catalogue_sources':sources,'SAM_weights_prior_identity':w,'SAM_weights_same_stat_not_fresh_full_hash':True,'max_requests':32,'no_early_caption_PAL_flag_fabricated':True,'original_source_selected_default_primary':True,'all_native_predictions_retained_for_fixed_index_diagnostic_only':True,'alternative_diagnostic_arms_predeclared':[0,1,2],'never_count_three_alternatives_as_three_annotations':True,'no_alternative_selection_from_GT_area_or_support':True,'all_final_gates_unchanged':True,'new_Agent_runs':0})
    e=read(file);assert e['source_plan_sha256']==sha(OUT/'frozen_plan.json') and all(sha(f)==h for f,h in e['protected'].items());w=e['SAM_weights_prior_identity'];weight=Path(w['path']);assert weight.stat().st_size==w['bytes'] and weight.stat().st_mtime_ns==w['mtime_ns'];return p,e

def run():
    if (OUT/'generation_completed.json').exists():raise FileExistsError('Do not repeat completed MF-first inference')
    if shutil.disk_usage(OUT).free<192*2**20:raise RuntimeError('Original192MiB stage output reserve')
    p,e=initialize();api.OUT=OUT;rows=[];seen=set();fresh=0
    for r in p['rows']:
        payload=read(r['request_file']);key=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();assert key==r['request_sha256'] and sha(r['input'])==r['input_sha256'];folder=OUT/'captures'/key;file=folder/'response.json';marker=folder/'network_request_started.json';previous=e['cache'].get(key);cached=False
        if not file.exists():
            if marker.exists():raise RuntimeError('A request completion is uncertain; inspect the original request/worker instead of repeating inference')
            if previous:
                assert previous['input_sha256']==r['input_sha256'] and previous['points']==r['points'] and sha(previous['response'])==previous['response_sha256'];d=read(previous['response']);save(file,{'raw_response':d,'historical_exact_reuse':True});cached=True
            else:
                if shutil.disk_usage(OUT).free<48*2**20:raise RuntimeError('Original48MiB full raw-array reserve')
                save(marker,{'sent_at':datetime.datetime.now().astimezone().isoformat(),'request_sha256':key,'body_file':r['request_file'],'do_not_repeat_without_authoritative_completion_state':True})
                status=200
                try:d=api.request_json('http://127.0.0.1:8766/segment_points',payload)
                except urllib.error.HTTPError as error:
                    status=error.code;raw=error.read();d=json.loads(raw)
                save(file,{'raw_response':d,'historical_exact_reuse':False,'HTTP_status':status});fresh+=1
        saved=read(file);d=saved['raw_response'];capture=d.get('raw_capture');row=r|{'folder':str(folder),'response_sha256':sha(file),'historical_exact_reuse':saved['historical_exact_reuse'],'same_batch_reuse':key in seen,'valid':False,'source_selected_observed':False};seen.add(key)
        if capture:
            meta=capture['metadata'];assert meta['source_sha256']==SOURCE_SHA and meta['prior_mask_sha256'] is None and meta['image_sha256']==r['input_sha256'];save(folder/'metadata.json',meta);rawbytes=base64.b64decode(capture['arrays_npz_base64']);(folder/'all_arrays.npz').write_bytes(rawbytes);arrays=np.load(io.BytesIO(rawbytes),allow_pickle=False);row.update(metadata_file=str(folder/'metadata.json'),raw_array_file=str(folder/'all_arrays.npz'),source_status=meta['status'],actual_predictor_calls_observed=len(meta.get('predictions',[])))
            if meta.get('raw_capture_complete') and meta.get('selected_index_observed'):
                call=meta['selected_prediction_call'];index=meta['selected_index'];all_masks=arrays[f'{call}_masks'];assert hashlib.sha256(all_masks.tobytes()).hexdigest()==meta['predictions'][call]['masks']['sha256'];raw=all_masks[index].squeeze();im=Image.open(r['input']);assert raw.shape==(im.height,im.width) and np.isin(raw,[0,1]).all();np.testing.assert_array_equal(arrays[f'{call}_point_coords'],[[q['x'],q['y']] for q in r['points']]);np.testing.assert_array_equal(arrays[f'{call}_point_labels'],[q['label'] for q in r['points']]);native=(raw!=0).astype('uint8');np.save(folder/'complete_native_mask.npy',native);Image.fromarray(native*255).save(folder/'native.png');feedback=describe_native(native,r['points']);x1,y1,x2,y2=r['observed_rectangle'];outside=native.copy();outside[y1:y2,x1:x2]=0;pad_pixels=int(outside.sum());row.update(source_selected_observed=True,prediction_call=call,native_index=index,native_pixels=int(native.sum()),native_prompt_guard=feedback['native_prompt_guard'],native_feedback=feedback,touches_input_border=feedback['touches_input_border'],foreground_on_synthetic_padding=pad_pixels,valid=meta['status']=='segment_returned' and not d.get('warnings') and feedback['native_prompt_passed'] and pad_pixels==0)
                # The full native output is saved first. Mapping is permitted
                # only if it discards no foreground, even for invalid masks.
                if pad_pixels==0:
                    vx,vy,_,_=r['virtual_box'];ax,ay,bx,by=x1+vx,y1+vy,x2+vx,y2+vy;assert 0<=ax<bx<=1920 and 0<=ay<by<=1536;full=np.zeros((1536,1920),'uint8');full[ay:by,ax:bx]=native[y1:y2,x1:x2];assert int(full.sum())==int(native.sum());np.save(folder/'complete_full_mask.npy',full);row.update(full_mask_sha256=hashlib.sha256(full.tobytes()).hexdigest(),all_native_foreground_preserved_in_source=True)
        rows.append(row);save(OUT/'progress.json',{'rows':rows,'fresh_requests_this_execution':fresh});print({'cue':r['index'],'arm':r['arm'],'pixels':row.get('native_pixels'),'padding_FG':row.get('foreground_on_synthetic_padding'),'input_border':row.get('touches_input_border'),'valid':row['valid'],'exact_prior_reuse':row['historical_exact_reuse'],'original_point_caption_PAL':r['original_neutral_point_PAL_evidence']},flush=True)
    assert len(rows)==32 and all(sha(f)==h for f,h in p['protected'].items());save(OUT/'generation_completed.json',{'status':'complete_original_source_selected_MF_first_SAM_capture','rows':rows,'logical_requests':len(rows),'unique_request_payloads':len(seen),'actual_new_SAM_capture_requests':len({r['request_sha256'] for r in rows if not r['historical_exact_reuse'] and r.get('actual_predictor_calls_observed',0)>0}),'exact_historical_capture_reuses':len({r['request_sha256'] for r in rows if r['historical_exact_reuse']}),'source_default_unobserved_requests':sum(not r['source_selected_observed'] for r in rows),'native_valid_rows':sum(r['valid'] for r in rows),'new_MF_calls':0,'new_Agent_runs':0,'new_accepted_instances':0,'all_previous_point_captions_kept_faithfully':True,'no_output_mask_cut':True,'adopted':False});print({k:v for k,v in read(OUT/'generation_completed.json').items() if k!='rows'},flush=True)

def main():
    import psutil
    file=OUT/'runtime_process.json'
    if file.exists():
        old=read(file)
        try:
            p=psutil.Process(old['pid']);live=p.is_running() and abs(p.create_time()-old['create_time'])<.01
        except psutil.NoSuchProcess:live=False
        if live:raise RuntimeError('Exact original process is live; poll instead of restart')
    state={'pid':os.getpid(),'create_time':psutil.Process().create_time(),'at':datetime.datetime.now().astimezone().isoformat(),'status':'running'};save(file,state)
    try:run();state['status']='complete'
    except BaseException as error:state.update(status='failed',error=str(error));raise
    finally:save(file,state|{'last_at':datetime.datetime.now().astimezone().isoformat()})

if __name__=='__main__':main()
