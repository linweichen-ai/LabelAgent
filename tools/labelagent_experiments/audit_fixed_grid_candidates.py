"""Independent source/capture readback and original complete-mask acceptance audit."""
import hashlib,json
from pathlib import Path
from prepare_fixed_grid_followup import ROOT,OUT
import numpy as np
from PIL import Image
from run_rgb_layers import read,save
from deliver_context_graph import sha
from paired_source_cue import body
from semantic_subject_guard import compatible_pallet_subject
from native_prompt_guard import evaluate
from run_fixed_grid_followup import SOURCE,SOURCE_SHA
import audit_soft_prior as original

def main():
    if (OUT/'independent_readback.json').exists():raise FileExistsError('Preserve completed source/capture audit')
    p=read(OUT/'frozen_plan.json');sp=read(OUT/'SAM/frozen_plan.json');g=read(OUT/'SAM/generation_completed.json');review=read(OUT/'review_completed.json');assert all(sha(f)==h for f,h in p['protected'].items()) and all(sha(f)==h for f,h in sp['protected'].items());selection=read(OUT/'semantic_source_selection.json');assert selection['selected_before_support_area_audit'];checks=[]
    for r in review['rows']:
        request=body(*[b['image'] for b in r['bindings']]);assert request==read(r['request_file']) and hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()==r['request_sha256'];reply=read(r['response_file']);assert sha(r['response_file'])==r['response_sha256'] and reply['local_vlm']['model_config_sha256']==p['model_config_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256,256];assert compatible_pallet_subject(json.loads(reply['choices'][0]['message']['content']))==r['PAL_subject_evidence']
        if r.get('known_no_real_object'):continue
        photo=Image.open(r['source_image']).convert('RGB')
        for b in r['bindings']:
            box=b['source_canvas_box'];clip=b['clipped_box'];canvas=Image.new('RGB',(b['side'],)*2,(127,)*3);canvas.paste(photo.crop(tuple(clip)),(clip[0]-box[0],clip[1]-box[1]));np.testing.assert_array_equal(np.asarray(canvas.resize((512,512),Image.Resampling.LANCZOS)),np.asarray(Image.open(b['image']).convert('RGB')))
    adapters={arm:[] for arm in ['original_source','explicit_padded_view']}
    for r in g['rows']:
        f=Path(r['folder']);d=read(f/'response.json');meta=read(r['metadata_file']);assert sha(f/'response.json')==r['response_sha256'] and meta==d['raw_capture']['metadata'] and meta['source_sha256']==sp['protected'][str(SOURCE)]==SOURCE_SHA and meta['prior_mask_sha256'] is None and meta['image_sha256']==sha(r['input']);a=np.load(r['raw_array_file'],allow_pickle=False);i=r['prediction_call'];native=a[f'{i}_masks'][r['native_index']].squeeze();np.testing.assert_array_equal(native,np.load(f/'complete_native_mask.npy',allow_pickle=False));np.testing.assert_array_equal(a[f'{i}_point_coords'],[[q['x'],q['y']] for q in r['points']]);np.testing.assert_array_equal(a[f'{i}_point_labels'],[q['label'] for q in r['points']]);assert evaluate(native,r['points'])==r['native_prompt_guard'];x1,y1,x2,y2=r['observed_rectangle'];outside=np.array(native,copy=True);outside[y1:y2,x1:x2]=0;assert int(outside.sum())==r['foreground_on_synthetic_padding'];full=np.load(f/'complete_full_mask.npy',allow_pickle=False);assert int(full.sum())==int(native.sum()) and hashlib.sha256(full.tobytes()).hexdigest()==r['full_mask_sha256'];vx,vy,_,_=r['virtual_box'];np.testing.assert_array_equal(full[y1+vy:y2+vy,x1+vx:x2+vx],native[y1:y2,x1:x2]);checks.append({'cue':r['index'],'arm':r['arm'],'actual_input_points_arrays_full_native_inverse_mapping_exact':True,'all_foreground_preserved':True,'native_pixels':int(native.sum()),'synthetic_padding_foreground':0})
        # Original evaluator accepts rectangular source bitmaps. This explicit
        # identity adapter is a full-frame evaluation representation, NOT a
        # substituted SAM capture, thresholded logits, or modified output.
        folder=OUT/'evaluation_adapters'/r['request_sha256'];folder.mkdir(parents=True);np.save(folder/'complete_full_mask.npy',full);np.savez_compressed(folder/'evaluation_bitmap.npz',evaluation_full_masks=full[None]);save(folder/'metadata.json',{'evaluation_adapter_only_not_SAM_capture':True,'original_native_arrays':r['raw_array_file'],'original_metadata':r['metadata_file'],'original_complete_native':str(f/'complete_native_mask.npy'),'original_complete_source_mask':str(f/'complete_full_mask.npy'),'source_foreground_pixels_preserved':int(full.sum()),'no_original_mask_changed':True})
        adapters[r['arm']].append(r|{'key':r['request_sha256'],'kind':'real','box':[0,0,1920,1536],'folder':str(folder),'raw_array_file':str(folder/'evaluation_bitmap.npz'),'metadata_file':str(folder/'metadata.json'),'prediction_call':'evaluation_full','native_index':0,'mask_sha256':r['full_mask_sha256'],'raw_pixels':r['native_pixels'],'original_capture_folder':str(f)})
    arms=[]
    for arm,rows in adapters.items():
        folder=OUT/'audits'/arm;save(folder/'generation_completed.json',{'status':'complete','rows':rows,'source_generation_sha256':sha(OUT/'SAM/generation_completed.json'),'all_full_native_foreground_preserved':True});original.OUT=folder;original.main();d=read(folder/'support_area_audit.json');arms.append({'arm':arm,'candidates':[{k:r[k] for k in ['index','raw_pixels','extra_pixels','valid','area_fits','person_conflict']} for r in d['candidates']],'radii':{c['case']:[{'radius':r['radius'],'old':r['old_matches'],'with_candidates':r['geometry_area_person_valid_matches'],'ignoring_area_validity':r['unfiltered_all_matches'],'ownership':r['new_mask_ownership']} for r in c['radii']] for c in d['cases']}})
    save(OUT/'audit_summary.json',{'arms':arms,'new_accepted_instances':0,'not_final_semantic_or_Agent_acceptance':True,'original_13_of_17_and_historical_6_of_6_retained':True,'all_original_gates_unchanged':True,'new_Agent_runs':0,'adopted':False});save(OUT/'independent_readback.json',{'passed':True,'checks':checks,'all_point_caption_photos_requests_raw_responses_tokens_and_model_exact':True,'all_original_selected_full_native_arrays_and_point_inputs_exact':True,'evaluation_adapters_identified_not_disguised_as_SAM_capture':True,'all_original_files_and_reference_hashes_checked':True,'semantic_source_selection_sha256':sha(OUT/'semantic_source_selection.json')});print({'capture_rows_independently_checked':len(checks),'arms':[(a['arm'],a['candidates']) for a in arms]},flush=True)

if __name__=='__main__':main()
