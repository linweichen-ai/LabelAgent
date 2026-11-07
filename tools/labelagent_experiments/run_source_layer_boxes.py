"""Source layer boxes on identical five-point inputs, with full native capture."""
import base64,datetime,hashlib,io,json,shutil,sys,time,urllib.request,urllib.error
from pathlib import Path
import numpy as np
from PIL import Image
from run_rgb_layers import ROOT,read,save,sha
from source_layer_box_policy import propose
from native_prompt_guard import evaluate
OUT=ROOT/'outputs/labelagent_source_layer_boxes_20261108';PREVIOUS=ROOT/'outputs/labelagent_two_sided_layers_20261108'
SERVICE=Path('D:/Apps/LabelAgent/source/sam3_service')
SOURCE_SHA='ab8861c57fbd275e8d16b5cc9bfbf734b3e4d7226d02b93d2b0ef9173fc49a81'

def body(r):return {'image':base64.b64encode(Path(r['input']).read_bytes()).decode(),'points':r['points'],'include_raw_capture':True,'diagnostic_box':r['diagnostic_box']}
def key(b):return hashlib.sha256(json.dumps(b,sort_keys=True).encode()).hexdigest()

def prepare():
    if OUT.exists():raise FileExistsError('Preserve source layer box comparison')
    if shutil.disk_usage(OUT.parent).free<96*2**20:raise RuntimeError('Keep original output reserve')
    assert sha(SERVICE/'point_segment.py')==SOURCE_SHA;old=read(PREVIOUS/'SAM/generation_completed.json');rows=[];diagnostics=[];protected={str(PREVIOUS/'SAM/generation_completed.json'):sha(PREVIOUS/'SAM/generation_completed.json')}
    for f in [Path(__file__),Path(__file__).with_name('source_layer_box_policy.py'),Path(__file__).with_name('native_prompt_guard.py'),SERVICE/'point_segment.py',SERVICE/'boxed_capture.py',SERVICE/'raw_capture_adapter.py']:protected[str(f)]=sha(f)
    for original in old['rows']:
        im=Image.open(original['input']);box,diag=propose(original['points'],*im.size);diagnostics.append({'partition':original['source_partition_index'],**diag})
        if box is None:continue
        row={k:original[k] for k in ['case','box','input','prior','source_image','points','source_partition_index','teacher_key','source_point']};row.update(index=original['source_partition_index'],diagnostic_box=box,baseline_folder=original['folder'],baseline_mask_sha256=original['mask_sha256'],baseline_generation_sha256=sha(PREVIOUS/'SAM/generation_completed.json'));request=body(row);row['request_sha256']=key(request);row['exact_prior_capture']=None
        # Exact full boxed requests only. Unboxed historical masks cannot be reused.
        for f in (ROOT/'outputs').glob('labelagent_*/generation_completed.json'):
            for previous in read(f).get('rows',[]):
                if previous.get('points')!=row['points'] or previous.get('diagnostic_box')!=box or previous.get('box')!=row['box']:continue
                if previous.get('input') and Path(previous['input']).is_file() and sha(previous['input'])==sha(row['input']):
                    capture=Path(previous.get('folder',''))/'response.json'
                    if capture.is_file() and key(body(previous))==row['request_sha256']:row['exact_prior_capture']=str(capture);protected[str(capture)]=sha(capture)
        for f in [row['input'],row['source_image'],str(Path(original['folder'])/'complete_full_mask.npy')]:protected[f]=sha(f)
        rows.append(row)
    assert len(rows)==3;OUT.mkdir();save(OUT/'frozen_plan.json',{'at':datetime.datetime.now().astimezone().isoformat(),'rows':rows,'diagnostics':diagnostics,'protected':protected,'source_sha256':SOURCE_SHA,'box_adapter_sha256':sha(SERVICE/'boxed_capture.py'),'same_original_images_and_all_five_points':True,'only_box_prompt_added':True,'source_selection_not_from_reference':True,'native_prompt_guard_required':True,'no_output_mask_clipping':True,'new_VLM_calls':0,'new_Mask2Former_forwards':0,'new_Agent_rounds':0,'Agent_round_limit':3,'max_new_SAM_calls':sum(r['exact_prior_capture'] is None for r in rows),'all_final_gates_unchanged':True});print({'prepared':len(rows),'cache_hits':sum(r['exact_prior_capture'] is not None for r in rows),'boxes':[r['diagnostic_box'] for r in rows],'new_model_calls':0},flush=True)

def run():
    if (OUT/'generation_completed.json').exists():raise FileExistsError('Do not repeat completed boxed requests')
    p=read(OUT/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());rows=[];fresh=0
    for r in p['rows']:
        assert key(body(r))==r['request_sha256'];folder=OUT/'captures'/r['request_sha256'];folder.mkdir(parents=True,exist_ok=True);file=folder/'response.json'
        if not file.exists():
            if r['exact_prior_capture']:shutil.copy2(r['exact_prior_capture'],file)
            else:
                if shutil.disk_usage(OUT).free<32*2**20:raise RuntimeError('Raw reply reserve')
                for _ in range(1500):
                    if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
                    try:
                        with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8766/segment_points',data=json.dumps(body(r)).encode(),headers={'Content-Type':'application/json'}),timeout=600) as f:d=json.load(f);http_status=f.status
                        fresh+=1;break
                    except urllib.error.HTTPError as e:
                        if e.code==503:time.sleep(.2);continue
                        d=json.loads(e.read());http_status=e.code;fresh+=1;break
                else:raise RuntimeError('Shared manager busy; resume missing replies only')
                save(file,d);save(folder/'transport.json',{'http_status':http_status,'request_sha256':r['request_sha256']})
        d=read(file);row={**r,'folder':str(folder),'response_sha256':sha(file),'valid':False,'service_reported_valid':False,'source_selected':True,'new_SAM_call':r['exact_prior_capture'] is None}
        capture=d.get('raw_capture')
        if capture:
            meta=capture['metadata'];assert meta['source_sha256']==SOURCE_SHA and meta['box_adapter_sha256']==p['box_adapter_sha256'];assert meta['experimental_box_prompt'] and meta['diagnostic_box']==r['diagnostic_box'] and meta['box_is_prompt_not_output_clip'];assert meta['actual_box_calls'] and all(c['box']==r['diagnostic_box'] and c['multimask_output'] is False for c in meta['actual_box_calls']);save(folder/'metadata.json',meta);rawbytes=base64.b64decode(capture['arrays_npz_base64']);(folder/'all_arrays.npz').write_bytes(rawbytes);a=np.load(io.BytesIO(rawbytes),allow_pickle=False);row.update(metadata_file=str(folder/'metadata.json'),raw_array_file=str(folder/'all_arrays.npz'),status=meta['status'])
            if meta.get('raw_capture_complete') and meta.get('selected_index_observed'):
                native=a[f"{meta['selected_prediction_call']}_masks"][meta['selected_index']].squeeze();assert np.isin(native,[0,1]).all();m=(native!=0).astype('uint8');x1,y1,x2,y2=r['box'];assert m.shape==(y2-y1,x2-x1);full=np.zeros((1536,1920),np.uint8);full[y1:y2,x1:x2]=m;assert int(full.sum())==int(m.sum());np.save(folder/'complete_full_mask.npy',full);Image.fromarray(m*255).save(folder/'raw.png');border=bool(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any());guard=evaluate(m,r['points']);service_valid=meta['status']=='segment_returned' and not border and not d.get('warnings');yy,xx=np.indices(m.shape);bx1,by1,bx2,by2=r['diagnostic_box'];inside=(xx>=bx1)&(xx<=bx2)&(yy>=by1)&(yy<=by2);outside=int(((m!=0)&~inside).sum());row.update(valid=service_valid and guard['valid'],service_reported_valid=service_valid,native_prompt_guard=guard,native_index=meta['selected_index'],prediction_call=meta['selected_prediction_call'],mask_sha256=hashlib.sha256(full.tobytes()).hexdigest(),raw_pixels=int(m.sum()),touches_border=border,pixels_outside_prompt_box_preserved=outside,all_native_arrays_retained=True)
        else:row['error']=d.get('error','No raw capture')
        rows.append(row);save(OUT/'progress.json',{'rows':rows,'actual_new_calls_this_execution':fresh});print({'partition':r['index'],'valid':row['valid'],'pixels':row.get('raw_pixels'),'native_guard':row.get('native_prompt_guard',{}).get('valid'),'outside_box_pixels_kept':row.get('pixels_outside_prompt_box_preserved'),'status':row.get('status',row.get('error'))},flush=True)
    assert all(sha(f)==h for f,h in p['protected'].items());save(OUT/'generation_completed.json',{'status':'complete_box_prompt_full_native_capture','rows':rows,'actual_new_SAM_calls':sum(r['new_SAM_call'] for r in rows),'actual_new_calls_this_execution':fresh,'exact_cache_reuses':sum(not r['new_SAM_call'] for r in rows),'new_Agent_runs':0,'adopted':False,'no_output_mask_clipping':True})

if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
