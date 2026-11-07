"""Freeze RGB seam box hypotheses from model-only parents, retain all masks."""
import base64,datetime,hashlib,io,json,sys,time,urllib.request,urllib.error
from pathlib import Path
import numpy as np
from PIL import Image
from native_multi_policy import place
from box_layer_policy import propose
from run_rgb_layers import ROOT,INC,read,save,sha
OUT=ROOT/'outputs/labelagent_box_layers_20261107'
OLD=ROOT/'outputs/labelagent_native_multi_20261107'


def prepare():
    if OUT.exists():raise FileExistsError('Preserve frozen experiments')
    OUT.mkdir();old=read(OLD/'generation_completed.json');protected=dict(read(OLD/'frozen_plan.json')['protected']);protected[str(OLD/'generation_completed.json')]=sha(OLD/'generation_completed.json');plans=[];diagnostics=[]
    for seed in sorted({r['seed'] for r in old['rows']}):
        options=[r for r in old['rows'] if r['seed']==seed and r['valid']]
        if not options:
            diagnostics.append({'seed':seed,'reason':'No non-border complete parent'});continue
        parent=sorted(options,key=lambda r:(-r['raw_pixels'],r['rank'],r['native_index'],r['key']))[0]
        source=Path(parent['input']);maskfile=Path(parent['folder'])/'complete_full_mask.npy';protected.update({str(source):sha(source),str(maskfile):sha(maskfile)})
        image=Image.open(source).convert('RGB');x1,y1,x2,y2=parent['box'];mask=np.load(maskfile,allow_pickle=False)[y1:y2,x1:x2];props,diag=propose(np.asarray(image),mask);folder=OUT/f'seed{seed:02d}';folder.mkdir();(folder/'input.png').write_bytes(source.read_bytes());Image.fromarray(mask*255).save(folder/'parent.png')
        diagnostics.append({'seed':seed,'parent_key':parent['key'],'parent_rule':'Largest valid whole native candidate, no reference or budget scoring',**diag})
        for rank,p in enumerate(props):
            binding={'points':p['points'],'diagnostic_box':p['diagnostic_box']};key=hashlib.sha256((sha(source)+json.dumps(binding,sort_keys=True)).encode()).hexdigest();plans.append({**p,'seed':seed,'rank':rank,'case':parent['case'],'box':parent['box'],'key':key,'input':str(folder/'input.png'),'input_sha256':sha(source),'anchor_origin':'RGB_seam_'+p['side'],'parent_key':parent['key']})
    deployed=read(ROOT/'task_workspaces/labelagent_box_layers_20261107/deployment/deployed.json');protected.update(deployed['protected_after'])
    save(OUT/'frozen_plan.json',{'at':datetime.datetime.now().astimezone().isoformat(),'driver_sha256':sha(__file__),'policy_sha256':sha(Path(__file__).with_name('box_layer_policy.py')),'protected':protected,'proposals':plans,'diagnostics':diagnostics,'max_requests':16,'no_reference_inputs':True,'reference_used_after_generation_only':True,'box_is_prompt_not_output_clip':True,'all_native_candidates_retained':True,'Agent_round_limit':3,'all_acceptance_gates_unchanged':True})
    print({'requests':len(plans),'seeds':len(diagnostics)},flush=True)


def run():
    if (OUT/'generation_completed.json').exists():
        raise FileExistsError('Do not repeat completed inference')
    plan = read(OUT/'frozen_plan.json')
    assert sha(__file__)==plan['driver_sha256'] and sha(Path(__file__).with_name('box_layer_policy.py'))==plan['policy_sha256']
    rows=[]; transport=0
    for p in plan['proposals']:
        assert sha(p['input'])==p['input_sha256']
        folder=OUT/'captures'/p['key'];folder.mkdir(parents=True,exist_ok=True)
        file=folder/'response.json'
        if not file.exists():
            body=json.dumps({'image':base64.b64encode(Path(p['input']).read_bytes()).decode(),'points':p['points'],'include_raw_capture':True,'diagnostic_box':p['diagnostic_box']}).encode()
            for attempt in range(1500):
                if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
                transport+=1
                try:
                    with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8766/segment_points',data=body,headers={'Content-Type':'application/json'}),timeout=600) as f:response=json.load(f)
                    break
                except urllib.error.HTTPError as error:
                    if error.code==503:
                        time.sleep(.2);continue
                    response=json.loads(error.read());break
            else:raise RuntimeError('Shared manager busy; resume missing requests only')
            save(file,response)
        response=read(file)
        capture=response.get('raw_capture')
        if not capture:
            rows.append({**p,'folder':str(folder),'valid':False,'error':'No raw capture','status':'capture_failed'})
            continue
        meta=capture['metadata'];save(folder/'metadata.json',meta)
        assert meta['experimental_box_prompt'] and meta['diagnostic_box']==[float(v) for v in p['diagnostic_box']]
        assert meta['actual_box_calls'] and all(c['box']==meta['diagnostic_box'] for c in meta['actual_box_calls'])
        npz=base64.b64decode(capture['arrays_npz_base64']);(folder/'all_arrays.npz').write_bytes(npz)
        arrays=np.load(io.BytesIO(npz),allow_pickle=False)
        for call,pred in enumerate(meta['predictions']):
            raw=arrays[f'{call}_masks'];scores=arrays[f'{call}_scores']
            assert pred['multimask_output'] is True
            assert hashlib.sha256(raw.tobytes()).hexdigest()==pred['masks']['sha256']
            for index,m in enumerate(raw):
                row={**p,'request_key':p['key'],'key':p['key']+f'_{call}_{index}','folder':str(folder/f'candidate{call}_{index}'),'status':meta['status'],'native_score':float(scores[index]),'native_index':index,'prediction_call':call,'source_selected':call==meta.get('selected_prediction_call') and index==meta.get('selected_index'),'all_arrays_sha256':sha(folder/'all_arrays.npz'),'valid':False}
                target=Path(row['folder']);target.mkdir(exist_ok=True)
                if np.isin(m,[0,1]).all():
                    full=place(m,p['box']);np.save(target/'complete_full_mask.npy',full);local=np.asarray(m).squeeze();Image.fromarray((local!=0).astype('uint8')*255).save(target/'raw.png')
                    border=bool(local[0].any() or local[-1].any() or local[:,0].any() or local[:,-1].any())
                    positive=all(bool(local[q['y'],q['x']]) for q in p['points'])
                    bx1,by1,bx2,by2=p['diagnostic_box'];outside=int((local!=0).sum())-int((local[by1:by2+1,bx1:bx2+1]!=0).sum())
                    row['pixels_outside_prompt_box_preserved']=outside
                    row.update(mask_sha256=hashlib.sha256(full.tobytes()).hexdigest(),raw_pixels=int(full.sum()),touches_border=border,positive_covered=positive,valid=meta['status']=='segment_returned' and positive and not border,raw_array_file=str(folder/'all_arrays.npz'),metadata_file=str(folder/'metadata.json'),lossless_full_mask=True)
                rows.append(row)
        save(OUT/'progress.json',{'rows':rows,'requests_completed':len({r.get('request_key',r['key']) for r in rows}),'transport_attempts_this_run':transport})
        print({'seed':p['seed'],'anchor':p['anchor_origin'],'native_candidates':len(meta['predictions'][0]['masks']['shape']) and meta['predictions'][0]['masks']['shape'][0],'valid':sum(r['valid'] for r in rows if r.get('request_key')==p['key'])},flush=True)
    assert all(sha(p)==h for p,h in plan['protected'].items())
    save(OUT/'generation_completed.json',{'status':'complete','rows':rows,'unique_real_SAM_calls':len(plan['proposals']),'native_candidates':len(rows),'valid_complete_outputs':sum(r['valid'] for r in rows),'protected_unchanged':True,'annotation_writes':0,'Agent_runs':0,'transport_attempts_this_run':transport})



if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
