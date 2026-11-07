"""Original source-selected, complete SAM masks for semantically screened cues."""
import base64,hashlib,io,json,os,shutil,sys,time,urllib.request,urllib.error
from pathlib import Path
import numpy as np
from PIL import Image
from run_rgb_layers import ROOT,read,save,sha

REVIEW=Path(os.environ['LABELAGENT_TRAINED_REVIEW_OUTPUT'])
OUT=REVIEW.parent/'labelagent_trained_point_sam_20261108'
SOURCE='D:/Apps/LabelAgent/source/sam3_service/point_segment.py'
SOURCE_SHA='ab8861c57fbd275e8d16b5cc9bfbf734b3e4d7226d02b93d2b0ef9173fc49a81'

def body(p):return {'image':base64.b64encode(Path(p['input']).read_bytes()).decode(),'points':p['points'],'include_raw_capture':True}
def key(p):return hashlib.sha256(json.dumps(body(p),sort_keys=True).encode()).hexdigest()

def prepare():
    if OUT.exists():raise FileExistsError('Preserve SAM point study')
    if shutil.disk_usage(OUT.parent).free<96*2**20:raise RuntimeError('Output reserve')
    assert sha(SOURCE)==SOURCE_SHA;review=read(REVIEW/'completed.json');rows=[];protected={str(REVIEW/'completed.json'):sha(REVIEW/'completed.json'),str(Path(__file__)):sha(__file__)};history=[]
    for base in [ROOT/'outputs',Path(os.environ['LABELAGENT_PAIRED_CUE_OUTPUT']).parent]:
        for f in base.glob('labelagent_*/generation_completed.json'):
            d=read(f)
            for r in d.get('rows',[]):
                if r.get('input') and r.get('points') and r.get('folder') and Path(r['input']).is_file() and (Path(r['folder'])/'response.json').is_file():history.append(r)
    for r in review['rows']:
        if not r['PAL_subject_evidence']:continue
        assert r['both_stock_panoptics_predict_PAL'];protected[r['input']]=sha(r['input']);k=key(r);cached=None
        for h in history:
            if h['points']!=r['points'] or h.get('box')!=r['box'] or sha(h['input'])!=sha(r['input']):continue
            response=Path(h['folder'])/'response.json';d=read(response);meta=d.get('raw_capture',{}).get('metadata',{})
            if meta.get('source_sha256')==SOURCE_SHA and meta.get('prior_mask_sha256') is None and meta.get('raw_capture_complete') and meta.get('selected_index_observed') and key(h)==k:
                cached=str(response);protected[cached]=sha(cached);break
        rows.append({**r,'sam_request_sha256':k,'exact_previous_capture':cached})
    OUT.mkdir();save(OUT/'frozen_plan.json',{'rows':rows,'protected':protected,'max_new_calls':sum(r['exact_previous_capture'] is None for r in rows),'source_sha256':SOURCE_SHA,'point_only_no_prior':True,'no_mask_clipping':True,'source_selected_index_only':True,'no_reference_or_budget_for_generation':True,'new_Agent_rounds':0,'original_Agent_round_limit':3});print({'qualified_cues':len(rows),'exact_cache_hits':sum(r['exact_previous_capture'] is not None for r in rows),'new_SAM_calls':0},flush=True)

def run():
    if (OUT/'generation_completed.json').exists():raise FileExistsError('Never repeat completed study')
    p=read(OUT/'frozen_plan.json');assert sha(SOURCE)==SOURCE_SHA and all(sha(f)==h for f,h in p['protected'].items());rows=[];fresh=0
    for r in p['rows']:
        assert key(r)==r['sam_request_sha256'];folder=OUT/'captures'/r['sam_request_sha256'];folder.mkdir(parents=True,exist_ok=True);file=folder/'response.json'
        if not file.exists():
            if r['exact_previous_capture']:shutil.copy2(r['exact_previous_capture'],file)
            else:
                if shutil.disk_usage(OUT).free<32*2**20:raise RuntimeError('Raw capture reserve')
                for _ in range(600):
                    if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
                    req=urllib.request.Request('http://127.0.0.1:8766/segment_points',data=json.dumps(body(r)).encode(),headers={'Content-Type':'application/json'})
                    try:
                        with urllib.request.urlopen(req,timeout=600) as f:d=json.load(f)
                        fresh+=1;break
                    except urllib.error.HTTPError as e:
                        if e.code==503:time.sleep(.2);continue
                        d=json.loads(e.read());fresh+=1;break
                else:raise RuntimeError('Shared model service busy')
                save(file,d)
        d=read(file);capture=d['raw_capture'];meta=capture['metadata'];assert meta['source_sha256']==SOURCE_SHA and meta['prior_mask_sha256'] is None;save(folder/'metadata.json',meta);rawbytes=base64.b64decode(capture['arrays_npz_base64']);(folder/'all_arrays.npz').write_bytes(rawbytes);a=np.load(io.BytesIO(rawbytes),allow_pickle=False);row={**r,'folder':str(folder),'metadata_file':str(folder/'metadata.json'),'raw_array_file':str(folder/'all_arrays.npz'),'valid':False,'source_selected':True,'new_SAM_call':r['exact_previous_capture'] is None,'response_sha256':sha(file)}
        if meta.get('raw_capture_complete') and meta.get('selected_index_observed'):
            raw=a[f"{meta['selected_prediction_call']}_masks"][meta['selected_index']].squeeze();assert np.isin(raw,[0,1]).all();m=(raw!=0).astype('uint8');x1,y1,x2,y2=r['box'];assert m.shape==(y2-y1,x2-x1);full=np.zeros((1536,1920),np.uint8);full[y1:y2,x1:x2]=m;assert int(full.sum())==int(m.sum());np.save(folder/'complete_full_mask.npy',full);Image.fromarray(m*255).save(folder/'raw.png');border=bool(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any());row.update(valid=meta['status']=='segment_returned' and not border and not d.get('warnings'),raw_pixels=int(m.sum()),touches_border=border,native_index=meta['selected_index'],prediction_call=meta['selected_prediction_call'],mask_sha256=hashlib.sha256(full.tobytes()).hexdigest(),all_native_arrays_retained=True,low_res_logits_present=any('logit' in k or 'low_res' in k for k in a.files))
        rows.append(row);save(OUT/'progress.json',rows);print({'index':r['index'],'valid':row['valid'],'pixels':row.get('raw_pixels'),'border':row.get('touches_border'),'cached':not row['new_SAM_call']},flush=True)
    assert all(sha(f)==h for f,h in p['protected'].items());save(OUT/'generation_completed.json',{'status':'complete_full_untrimmed_source_selected_masks','rows':rows,'actual_new_SAM_calls':sum(r['new_SAM_call'] for r in rows),'new_calls_this_execution':fresh,'exact_reused_SAM_calls':sum(not r['new_SAM_call'] for r in rows),'new_Agent_runs':0,'adopted':False})

if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
