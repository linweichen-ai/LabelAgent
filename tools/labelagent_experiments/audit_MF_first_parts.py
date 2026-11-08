"""Source-pixel/capture readback and unchanged original support/area evaluation."""
import hashlib,json,sys
from pathlib import Path
from prepare_mf_first_parts import OUT
from prepare_palette_parts import ROOT,BASE
import numpy as np
from PIL import Image
from run_rgb_layers import read,save
from deliver_context_graph import sha
from native_review_feedback import describe_native,SOURCE_SHA
import audit_soft_prior as original

def score_arm(name,rows):
    folder=OUT/'audits'/name
    if (folder/'support_area_audit.json').exists():raise FileExistsError('Preserve completed arm audit')
    save(folder/'generation_completed.json',{'status':'complete','rows':rows,'source_generation_sha256':sha(OUT/'generation_completed.json'),'no_original_mask_or_model_output_modified':True});original.OUT=folder;original.main();d=read(folder/'support_area_audit.json');return {'arm':name,'candidates':[{k:r[k] for k in ['key','index','raw_pixels','extra_pixels','valid','area_fits','person_conflict']} for r in d['candidates']],'radii':{c['case']:[{'radius':r['radius'],'old':r['old_matches'],'with_candidates':r['geometry_area_person_valid_matches'],'ignoring_native_validity_and_area':r['unfiltered_all_matches'],'ownership':r['new_mask_ownership'],'remaining_targets':r['remaining_targets']} for r in c['radii']] for c in d['cases']}}

def evaluation_row(r,key=None):
    folder=Path(r['folder']);full=np.load(folder/'complete_full_mask.npy',allow_pickle=False);assert hashlib.sha256(full.tobytes()).hexdigest()==r['full_mask_sha256'];evaluation=OUT/'evaluation_adapters'/(key or r.get('key') or r['request_sha256']);evaluation.mkdir(parents=True,exist_ok=True);np.savez_compressed(evaluation/'bitmap.npz',evaluation_full_masks=full[None]);save(evaluation/'metadata.json',{'evaluation_adapter_only_not_SAM_capture':True,'original_raw_array_file':r['raw_array_file'],'original_metadata_file':r['metadata_file'],'original_prediction_call':r['prediction_call'],'original_native_index':r['native_index'],'all_source_foreground_pixels_retained':int(full.sum()),'no_original_bitmap_modified':True});return r|{'key':key or r.get('key') or r['request_sha256'],'kind':'real','box':[0,0,1920,1536],'raw_array_file':str(evaluation/'bitmap.npz'),'metadata_file':str(evaluation/'metadata.json'),'prediction_call':'evaluation_full','native_index':0,'raw_pixels':int(full.sum()),'mask_sha256':r['full_mask_sha256'],'original_native_array_file':r['raw_array_file'],'original_capture_metadata_file':r['metadata_file']}

def main():
    if (OUT/'independent_readback.json').exists():raise FileExistsError('Preserve complete MF-first audit')
    p=read(OUT/'frozen_plan.json');g=read(OUT/'generation_completed.json');pool=read(OUT/'native_pool/frozen_outputs.json');assert all(sha(f)==h for f,h in p['protected'].items()) and all(sha(f)==h for f,h in pool['protected'].items());assert read(OUT/'semantic_source_selection.json')['selected_before_support_area_audit'];assert pool['all_source_native_valid_selected_before_reference_scoring'];checks=[];unmapped=[];arms={'original_source':[],'explicit_padded_view':[]};teachers=read(BASE/'labelagent_registered_fixed_grid_20261108/generation_completed.json')['rows']
    for r in g['rows']:
        payload=read(r['request_file']);assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==r['request_sha256'] and sha(r['input'])==r['input_sha256'];reply=read(Path(r['folder'])/'response.json');assert sha(Path(r['folder'])/'response.json')==r['response_sha256'];d=reply['raw_response'];meta=read(r['metadata_file']);assert meta==d['raw_capture']['metadata'] and meta['image_sha256']==r['input_sha256'] and meta['source_sha256']==SOURCE_SHA and meta['prior_mask_sha256'] is None;arrays=np.load(r['raw_array_file'],allow_pickle=False);call=r['prediction_call'];masks=arrays[f'{call}_masks'];assert hashlib.sha256(masks.tobytes()).hexdigest()==meta['predictions'][call]['masks']['sha256'];np.testing.assert_array_equal(arrays[f'{call}_point_coords'],[[q['x'],q['y']] for q in r['points']]);np.testing.assert_array_equal(arrays[f'{call}_point_labels'],[q['label'] for q in r['points']]);native=(masks[r['native_index']].squeeze()!=0).astype('uint8');np.testing.assert_array_equal(native,np.load(Path(r['folder'])/'complete_native_mask.npy',allow_pickle=False));feedback=describe_native(native,r['points']);assert feedback==r['native_feedback'];x1,y1,x2,y2=r['observed_rectangle'];outside=native.copy();outside[y1:y2,x1:x2]=0;pad=int(outside.sum());assert pad==r['foreground_on_synthetic_padding'];assert r['valid']==(meta['status']=='segment_returned' and not d.get('warnings') and feedback['native_prompt_passed'] and pad==0)
        # Compare every observed RGB pixel, including halo, with the original
        # photo. Synthetic pixels outside the observed rectangle must be127.
        image=np.asarray(Image.open(r['input']).convert('RGB'));source=np.asarray(Image.open(r['source_image']).convert('RGB'));vx,vy,_,_=r['virtual_box'];np.testing.assert_array_equal(image[y1:y2,x1:x2],source[y1+vy:y2+vy,x1+vx:x2+vx]);observed=np.zeros(native.shape,bool);observed[y1:y2,x1:x2]=True;assert (image[~observed]==127).all()
        teacher=next(t for t in teachers if t['model']==r['model'] and t['teacher_key']==r['teacher_key']);rgb=np.asarray(Image.open(Path(teacher['folder'])/'panoptic.png')).astype('uint32');ids=rgb[:,:,0]+256*rgb[:,:,1]+65536*rgb[:,:,2];mx,my=r['source_point'][0]-teacher['source_virtual_box'][0],r['source_point'][1]-teacher['source_virtual_box'][1];assert ids[my,mx]==r['stock_segment']['id'] and r['stock_segment']['category_id']==1
        if pad:
            assert not (Path(r['folder'])/'complete_full_mask.npy').exists();unmapped.append({'cue':r['index'],'arm':r['arm'],'native_FG_on_synthetic_padding':pad,'no_full_frame_mask_fabricated':True})
        else:
            full=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False);np.testing.assert_array_equal(full[y1+vy:y2+vy,x1+vx:x2+vx],native[y1:y2,x1:x2]);assert int(full.sum())==int(native.sum());arms[r['arm']].append(evaluation_row(r))
        checks.append({'cue':r['index'],'arm':r['arm'],'source_pixels_request_points_masks_and_original_selection_exact':True,'source_MF_stock_PAL_at_actual_proposed_pixel_verified':True,'native_border':feedback['touches_input_border'],'padding_FG':pad,'point_caption_not_relabelled_as_PAL':r['original_neutral_point_PAL_evidence']})
    results=[]
    for arm,rows in arms.items():results.append(score_arm('source_default_'+arm,rows))
    for index in [0,1,2]:
        for arm in ['original_source','explicit_padded_view']:
            rows=[evaluation_row(r) for r in pool['rows'] if r['native_index']==index and r['arm']==arm and r['valid']];results.append(score_arm(f'fixed_index{index}_'+arm,rows))
    save(OUT/'audit_summary.json',{'status':'complete_post_source_freeze_default_and_fixed_index_diagnostic','arms':results,'unmappable_padding_violations_not_scored_by_cutting':unmapped,'all_native_hypotheses_selected_before_scoring':True,'original_source_selected_default_not_replaced':True,'no_alternative_chosen_by_GT_purity_or_area':True,'new_model_calls':0,'new_Agent_runs':0,'new_accepted_instances':0,'adopted':False});save(OUT/'independent_readback.json',{'passed':True,'checks':checks,'unmappable_outputs_preserved_without_mask_clipping':unmapped,'all_MF_source_pixels_and_SAM_actual_inputs_points_and_native_masks_verified':True,'original_point_captions_preserved_not_fabricated_PAL':True,'pool_outputs_sha256':sha(OUT/'native_pool/frozen_outputs.json'),'source_semantic_selection_sha256':sha(OUT/'semantic_source_selection.json'),'reference_evaluator_original_and_gates_unchanged':True});print({'source_requests_checked':len(checks),'unmappable_padding_failures':len(unmapped),'diagnostic_arms':len(results),'new_model_calls':0},flush=True)

if __name__=='__main__':main()
