"""Freeze image-only single clicks, then retain every native SAM candidate."""
import base64, datetime, hashlib, io, json, sys, time, urllib.request, urllib.error
from pathlib import Path
import numpy as np
from PIL import Image
from native_multi_policy import anchors, place
from run_rgb_layers import ROOT, INC, read, save, sha

OUT = ROOT/'outputs/labelagent_native_multi_20261107'


def prepare():
    if OUT.exists():
        raise FileExistsError('Keep completed and partial experiments immutable')
    OUT.mkdir()
    cohort = read(ROOT/'outputs/labelagent_point_ids_20261107/planning/cohort.json')
    cfg = read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json')
    protected = {str(p): sha(p) for p in [Path(cfg['reference_annotations']), Path(cfg['reference_description'])] + list(INC.glob('*/*/result.json'))}
    proposals = []
    for seed in cohort:
        source = INC/'inputs'/(seed['case']+'.png')
        seedfile = Path(seed['folder'])/'seed_mask.png'
        protected.update({str(source): sha(source), str(seedfile): sha(seedfile)})
        box = seed['binding']['source_box']
        x1,y1,x2,y2 = box
        image = Image.open(source).convert('RGB').crop(tuple(box))
        mask = np.asarray(Image.open(seedfile))[y1:y2,x1:x2] > 0
        folder = OUT/f"seed{seed['index']:02d}"
        folder.mkdir()
        image.save(folder/'input.png')
        Image.fromarray(mask.astype('uint8')*255).save(folder/'seed.png')
        for rank, point in enumerate(anchors(mask)):
            points = [{k:point[k] for k in ('x','y','label')}]
            key = hashlib.sha256((sha(folder/'input.png')+str(points)).encode()).hexdigest()
            proposals.append({'seed':seed['index'],'rank':rank,'case':seed['case'],'box':box,'input':str(folder/'input.png'),'input_sha256':sha(folder/'input.png'),'key':key,'points':points,'anchor_origin':point['origin'],'no_reference_inputs':True})
    protected[str(Path('D:/Apps/LabelAgent/source/sam3_service/point_segment.py'))] = sha('D:/Apps/LabelAgent/source/sam3_service/point_segment.py')
    save(OUT/'frozen_plan.json', {'at':datetime.datetime.now().astimezone().isoformat(),'driver_sha256':sha(__file__),'policy_sha256':sha(Path(__file__).with_name('native_multi_policy.py')),'protected':protected,'proposals':proposals,'max_requests':16,'all_native_candidates_retained':True,'selection_feedback_into_generation':False,'gate_changes':False,'Agent_round_limit':3,'proposal_change':'One click activates existing native multimask path; all outputs audited, no component removal or reference-derived points.'})
    print({'prepared':len(proposals)},flush=True)


def run():
    if (OUT/'generation_completed.json').exists():
        raise FileExistsError('Do not repeat completed inference')
    plan = read(OUT/'frozen_plan.json')
    assert sha(__file__)==plan['driver_sha256'] and sha(Path(__file__).with_name('native_multi_policy.py'))==plan['policy_sha256']
    rows=[]; transport=0
    for p in plan['proposals']:
        assert sha(p['input'])==p['input_sha256']
        folder=OUT/'captures'/p['key'];folder.mkdir(parents=True,exist_ok=True)
        file=folder/'response.json'
        if not file.exists():
            body=json.dumps({'image':base64.b64encode(Path(p['input']).read_bytes()).decode(),'points':p['points'],'include_raw_capture':True}).encode()
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
                    row.update(mask_sha256=hashlib.sha256(full.tobytes()).hexdigest(),raw_pixels=int(full.sum()),touches_border=border,positive_covered=positive,valid=meta['status']=='segment_returned' and positive and not border,raw_array_file=str(folder/'all_arrays.npz'),metadata_file=str(folder/'metadata.json'),lossless_full_mask=True)
                rows.append(row)
        save(OUT/'progress.json',{'rows':rows,'requests_completed':len({r.get('request_key',r['key']) for r in rows}),'transport_attempts_this_run':transport})
        print({'seed':p['seed'],'anchor':p['anchor_origin'],'native_candidates':len(meta['predictions'][0]['masks']['shape']) and meta['predictions'][0]['masks']['shape'][0],'valid':sum(r['valid'] for r in rows if r.get('request_key')==p['key'])},flush=True)
    assert all(sha(p)==h for p,h in plan['protected'].items())
    save(OUT/'generation_completed.json',{'status':'complete','rows':rows,'unique_real_SAM_calls':len(plan['proposals']),'native_candidates':len(rows),'valid_complete_outputs':sum(r['valid'] for r in rows),'protected_unchanged':True,'annotation_writes':0,'Agent_runs':0,'transport_attempts_this_run':transport})


if __name__=='__main__':
    prepare() if sys.argv[1]=='prepare' else run()
