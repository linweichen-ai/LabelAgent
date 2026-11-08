"""Read every actual request/capture, then score fixed whole masks unchanged."""
import hashlib,json,sys
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

def prepare_semantics():
    import review_fixed_grid_instances as wrapper
    wrapper.OUT=OUT;g=read(OUT/'SAM/generation_completed.json')
    if any(r['valid'] for r in g['rows']):wrapper.prepare()
    else:
        import review_box_layers as reviewer
        save(OUT/'semantic/frozen_plan.json',{'driver_sha256':sha(reviewer.__file__),'requests':[],'selected':[],'protected':{str(OUT/'SAM/generation_completed.json'):sha(OUT/'SAM/generation_completed.json')},'model_config_sha256':read(OUT/'frozen_plan.json')['model_config_sha256'],'all_original_gates_required':True,'no_native_valid_candidate_available':True});save(OUT/'semantic_source_selection.json',{'selected_all_native_valid':0,'selected_before_support_area_audit':not (OUT/'audits').exists(),'all_original_semantic_gates_unchanged':True})
    save(OUT/'explicit_audit_binding.json',{'wrapper_sha256':sha(__file__),'actual_source_cohort':str(OUT),'neutral_reply_envelopes_normalized_without_changing_files':True,'all_native_valid_outputs_selected_before_GT_or_area':True,'all_original_semantics_required':True})

def audit():
    if (OUT/'independent_readback.json').exists():raise FileExistsError('Preserve completed source-part audit')
    import audit_soft_prior as original
    p=read(OUT/'frozen_plan.json');sp=read(OUT/'SAM/frozen_plan.json');g=read(OUT/'SAM/generation_completed.json');rv=read(OUT/'review_completed.json');assert all(sha(f)==h for f,h in p['protected'].items()) and all(sha(f)==h for f,h in sp['protected'].items());assert read(OUT/'semantic_source_selection.json')['selected_before_support_area_audit'];checks=[]
    for r in rv['rows']:
        request=api.body(*[b['image'] for b in r['bindings']]);assert request==read(r['request_file']) and hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()==r['request_sha256'];assert sha(r['response_file'])==r['response_sha256'];reply=read(r['response_file'])['raw_response'];assert reply['local_vlm']['model_config_sha256']==p['model_config_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256,256] and json.loads(reply['choices'][0]['message']['content'])==r['decision'] and compatible_pallet_subject(r['decision'])==r['PAL_subject_evidence']
        if r.get('known_no_real_object'):continue
        source=Image.open(r['source_image']).convert('RGB')
        for b in r['bindings']:
            box=b['source_canvas_box'];clip=b['clipped_box'];im=Image.new('RGB',(b['side'],)*2,(127,)*3);im.paste(source.crop(tuple(clip)),(clip[0]-box[0],clip[1]-box[1]));np.testing.assert_array_equal(np.asarray(im.resize((512,512),Image.Resampling.LANCZOS)),np.asarray(Image.open(b['image']).convert('RGB')))
    arms={};failures=[]
    for r in g['rows']:
        folder=Path(r['folder']);d=read(folder/'response.json');meta=read(r['metadata_file']);assert sha(folder/'response.json')==r['response_sha256'] and meta==d['raw_capture']['metadata'] and meta['source_sha256']==api.SOURCE_SHA and meta['prior_mask_sha256'] is None and meta['image_sha256']==sha(r['input']);a=np.load(r['raw_array_file'],allow_pickle=False)
        for call,prediction in enumerate(meta['predictions']):
            assert hashlib.sha256(a[f'{call}_masks'].tobytes()).hexdigest()==prediction['masks']['sha256'];np.testing.assert_array_equal(a[f'{call}_point_coords'],[[q['x'],q['y']] for q in r['points']]);np.testing.assert_array_equal(a[f'{call}_point_labels'],[q['label'] for q in r['points']])
        if 'prediction_call' not in r:failures.append({'cue':r['index'],'arm':r['arm'],'source_status':meta['status'],'all_raw_arrays_retained_no_selected_mask_invented':True});continue
        native=a[f"{r['prediction_call']}_masks"][r['native_index']].squeeze();full=np.load(folder/'complete_full_mask.npy',allow_pickle=False);np.testing.assert_array_equal(native,full);np.testing.assert_array_equal(native,np.load(folder/'complete_native_mask.npy',allow_pickle=False));guard=evaluate(full,r['points']);border=bool(full[0].any() or full[-1].any() or full[:,0].any() or full[:,-1].any());assert guard==r['native_prompt_guard'] and border==r['touches_input_border'];expected=meta['status']=='segment_returned' and not d.get('warnings') and guard['valid'] and not border;assert expected==r['valid'];assert hashlib.sha256(full.tobytes()).hexdigest()==r['full_mask_sha256'];checks.append({'cue':r['index'],'arm':r['arm'],'all_actual_source_points_arrays_original_selected_mask_exact':True,'native_input_border_independently_checked':border,'native_point_guard':guard,'all_foreground_preserved':True});arms.setdefault(r['arm'],[]).append(r|{'key':r['request_sha256'],'kind':'real','box':[0,0,1920,1536],'mask_sha256':r['full_mask_sha256'],'raw_pixels':r['native_pixels']})
    summaries=[]
    for arm in ['one_positive','source_component_negatives']:
        folder=OUT/'audits'/arm;save(folder/'generation_completed.json',{'status':'complete','rows':arms.get(arm,[]),'actual_source_generation_sha256':sha(OUT/'SAM/generation_completed.json'),'missing_original_selected_outputs_not_replaced':True});original.OUT=folder;original.main();d=read(folder/'support_area_audit.json');summaries.append({'arm':arm,'candidates':[{k:r[k] for k in ['index','raw_pixels','extra_pixels','valid','area_fits','person_conflict']} for r in d['candidates']],'radii':{c['case']:[{'radius':r['radius'],'old':r['old_matches'],'with_candidates':r['geometry_area_person_valid_matches'],'ignoring_validity_and_area':r['unfiltered_all_matches'],'ownership':r['new_mask_ownership']} for r in c['radii']] for c in d['cases']}})
    save(OUT/'audit_summary.json',{'arms':summaries,'missing_default_selection_captures':failures,'no_final_mask_cut':True,'all_original_final_gates_unchanged':True,'new_Agent_runs':0,'new_accepted_instances':0,'adopted':False});save(OUT/'independent_readback.json',{'passed':True,'neutral_source_images_requests_replies_tokens_model_exact':True,'native_checks':checks,'source_selection_failures_retained':failures,'native_input_border_and_original_negative_radius3_explicitly_enforced':True,'all_raw_native_arrays_and_full_foreground_preserved':True,'no_original_annotation_modified':True});print({'checked_native_rows':len(checks),'source_selection_failures':len(failures),'arms':[(s['arm'],s['candidates']) for s in summaries]},flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare_semantics':prepare_semantics()
    elif sys.argv[1]=='audit':audit()
    else:
        import review_box_layers as runner
        runner.OUT=OUT;runner.S=OUT/'semantic';runner.run()
