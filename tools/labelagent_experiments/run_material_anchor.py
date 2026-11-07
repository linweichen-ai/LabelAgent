"""Image-only material anchors, neutral review, then one raw SAM call per cue."""
import base64,hashlib,io,json,os,shutil,sys,time,urllib.request,urllib.error
from pathlib import Path
import numpy as np
from PIL import Image
from material_anchor_policy import propose
from run_pallet_chain import ROOT,old
from run_visual_budget import payload
from semantic_subject_guard import compatible_pallet_subject

SHORT=Path(os.environ.get('LABELAGENT_SHORT_CHAIN_SOURCE',str(ROOT/'outputs/labelagent_chain_rgb_20261108')))
OUT=Path(os.environ.get('LABELAGENT_MATERIAL_OUTPUT',str(ROOT/'outputs/labelagent_material_anchor_20261108')))


def png(im):
    b=io.BytesIO();im.save(b,'PNG');return b.getvalue()


def prepare():
    if OUT.exists():raise FileExistsError('Preserve image-only cue trial')
    if shutil.disk_usage(OUT.parent).free<256*2**20:raise RuntimeError('Output space guard')
    previous=old.read(SHORT/'frozen_plan.json');assert all(old.digest(p)==h for p,h in previous['protected'].items());OUT.mkdir();protected={str(SHORT/'frozen_plan.json'):old.digest(SHORT/'frozen_plan.json'),str(Path(__file__)):old.digest(__file__),str(Path(__file__).with_name('material_anchor_policy.py')):old.digest(Path(__file__).with_name('material_anchor_policy.py'))};rows=[];diagnostics=[];model=old.read(ROOT/'outputs/labelagent_native_multi_20261107/semantic/frozen_plan.json')['model_config_sha256']
    for seed in previous['seeds']:
        camera=int(seed['case'].split('_')[0][6:]);source=next(s for s in previous['native_sources'] if s['camera']==camera and s['frame']==76);protected[source['path']]=source['sha256'];photo=Image.open(source['path']).convert('RGB');box=seed['binding']['source_box'];roi=photo.crop(tuple(box));anchor=[seed['initial_points'][0][0]-box[0],seed['initial_points'][0][1]-box[1]];points,cue,diag=propose(np.asarray(roi),anchor);dest=OUT/f"seed{seed['index']:02d}";dest.mkdir();roi.save(dest/'SAM_input.png');Image.fromarray(cue*255).save(dest/'RGB_cue_not_an_annotation.png');diagnostics.append({'seed':seed['index'],'case':seed['case'],**diag})
        for q in points:
            p=q['points'][0];ax,ay=int(p['x']+box[0]),int(p['y']+box[1]);canvasbox=[ax-64,ay-64,ax+65,ay+65];clip=[max(0,canvasbox[0]),max(0,canvasbox[1]),min(photo.width,canvasbox[2]),min(photo.height,canvasbox[3])];im=Image.new('RGB',(129,129),(127,127,127));im.paste(photo.crop(tuple(clip)),(clip[0]-canvasbox[0],clip[1]-canvasbox[1]));file=dest/(q['cue']+'_neutral.png');im.resize((512,512),Image.Resampling.LANCZOS).save(file);body=payload(file.read_bytes(),256);key=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest();samkey=hashlib.sha256((old.digest(dest/'SAM_input.png')+json.dumps(q['points'],sort_keys=True)).encode()).hexdigest();rows.append({**q,'key':samkey,'seed':seed['index'],'case':seed['case'],'box':box,'input':str(dest/'SAM_input.png'),'input_sha256':old.digest(dest/'SAM_input.png'),'source_image':source['path'],'source_point':[ax,ay],'semantic_image':str(file),'semantic_image_sha256':old.digest(file),'semantic_canvas_box':canvasbox,'semantic_clip_box':clip,'request_sha256':key})
    for name,value in [('gray',127),('white',255)]:
        image=np.full((129,129,3),value,np.uint8);p,_,diag=propose(image,[64,64]);assert not p;diagnostics.append({'synthetic_control':name,'cue_policy_proposals':len(p),'not_a_new_model_measurement':True})
    old.save(OUT/'frozen_plan.json',{'frozen_at':__import__('datetime').datetime.now().astimezone().isoformat(),'proposals':rows,'diagnostics':diagnostics,'protected':protected,'model_config_sha256':model,'max_real_seeds':8,'max_hypotheses_per_seed':2,'neutral_model_image_tokens':256,'neutral_object_caption_not_instance_confirmation':True,'no_reference_labels_points_pose_calibration_or_area_inputs':True,'original_mask_not_a_color_template_or_prior':True,'one_SAM_call_per_qualified_cue':True,'same_original_full_image_ROI_not_new_budget_crops':True,'all_native_candidates_retained_source_selection_only':True,'Agent_round_limit':3,'new_Agent_rounds':0,'all_final_acceptance_gates_unchanged':True,'no_post_mask_clipping':True});print({'proposals':len(rows),'seeds':8,'max_SAM_after_neutral_guard':len(rows)},flush=True)


def review():
    if (OUT/'semantic_completed.json').exists():raise FileExistsError('Preserve neutral replies')
    p=old.read(OUT/'frozen_plan.json');assert all(old.digest(f)==h for f,h in p['protected'].items());rows=[];seen=set()
    for r in p['proposals']:
        assert old.digest(r['semantic_image'])==r['semantic_image_sha256'];body=payload(Path(r['semantic_image']).read_bytes(),256);assert hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()==r['request_sha256'];file=OUT/'semantic_responses'/(r['request_sha256']+'.json');existed=file.exists()
        if not existed:
            for _ in range(1500):
                if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
                try:
                    with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18868/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=300) as f:reply=json.load(f)
                    old.save(file,reply);break
                except urllib.error.HTTPError as e:
                    if e.code==503:time.sleep(.2);continue
                    raise
            else:raise RuntimeError('Shared manager busy; preserve saved replies')
        reply=old.read(file);assert reply['local_vlm']['model_config_sha256']==p['model_config_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256];caption=json.loads(reply['choices'][0]['message']['content']);row={**r,'decision':caption,'PAL_guard':compatible_pallet_subject(caption),'response':str(file),'response_sha256':old.digest(file),'same_batch_reuse':r['request_sha256'] in seen};seen.add(r['request_sha256']);rows.append(row);old.save(OUT/'semantic_progress.json',rows);print({'seed':r['seed'],'cue':r['cue'],'caption':caption,'PAL_guard':row['PAL_guard']},flush=True)
    old.save(OUT/'semantic_completed.json',{'rows':rows,'unique_real_VLM_calls':len(seen),'logical_views':len(rows),'same_batch_reuses':sum(r['same_batch_reuse'] for r in rows),'eligible_for_SAM':sum(r['PAL_guard'] for r in rows),'no_categorical_truth_or_adoption':True})


def run():
    if (OUT/'generation_completed.json').exists():raise FileExistsError('Preserve actual raw masks')
    plan=old.read(OUT/'frozen_plan.json');assert all(old.digest(f)==h for f,h in plan['protected'].items());rows=[];calls=0
    for r in old.read(OUT/'semantic_completed.json')['rows']:
        if not r['PAL_guard']:continue
        folder=OUT/'captures'/r['key'];folder.mkdir(parents=True,exist_ok=True);file=folder/'response.json'
        if not file.exists():
            body={'image':base64.b64encode(Path(r['input']).read_bytes()).decode(),'points':r['points'],'include_raw_capture':True}
            for _ in range(1500):
                if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
                try:
                    with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8766/segment_points',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=600) as f:response=json.load(f)
                    break
                except urllib.error.HTTPError as e:
                    if e.code==503:time.sleep(.2);continue
                    response=json.loads(e.read());break
            else:raise RuntimeError('Shared manager busy; preserve completed masks')
            old.save(file,response)
        response=old.read(file);capture=response.get('raw_capture');row={**r,'kind':'real','folder':str(folder),'valid':False,'status':'no_capture'}
        if capture:
            meta=capture['metadata'];old.save(folder/'metadata.json',meta);data=base64.b64decode(capture['arrays_npz_base64']);(folder/'all_arrays.npz').write_bytes(data);a=np.load(io.BytesIO(data),allow_pickle=False);calls+=meta['predict_inst_calls'];row.update(status=meta['status'],predict_inst_calls=meta['predict_inst_calls'],source_error=meta.get('error'))
            if meta.get('raw_capture_complete') and meta.get('selected_index_observed'):
                i=meta['selected_prediction_call'];j=meta['selected_index'];m=a[f'{i}_masks'][j].squeeze();assert np.isin(m,[0,1]).all();x1,y1,x2,y2=r['box'];assert m.shape==(y2-y1,x2-x1);full=np.zeros((1536,1920),np.uint8);full[y1:y2,x1:x2]=(m!=0);assert int(full.sum())==int((m!=0).sum());np.save(folder/'complete_full_mask.npy',full);Image.fromarray((m!=0).astype('uint8')*255).save(folder/'raw.png');border=bool(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any());agree=all(bool(m[p['y'],p['x']])==bool(p['label']) for p in r['points']);row.update(raw_pixels=int(full.sum()),mask_sha256=hashlib.sha256(full.tobytes()).hexdigest(),touches_border=border,point_pixels_agree=agree,valid=meta['status']=='segment_returned' and not border and agree and not response.get('warnings'),raw_array_file=str(folder/'all_arrays.npz'),metadata_file=str(folder/'metadata.json'),prediction_call=i,native_index=j,source_selected=True)
        rows.append(row);old.save(OUT/'generation_progress.json',rows);print({'seed':r['seed'],'cue':r['cue'],'valid':row['valid'],'pixels':row.get('raw_pixels')},flush=True)
    assert all(old.digest(f)==h for f,h in plan['protected'].items());old.save(OUT/'generation_completed.json',{'status':'complete','rows':rows,'unique_real_SAM_calls':calls,'new_Agent_runs':0,'adopted':False,'no_output_mask_clipping':True,'only_source_selected_native_masks_eligible':True})


if __name__=='__main__':{'prepare':prepare,'review':review,'run':run}[sys.argv[1]]()
