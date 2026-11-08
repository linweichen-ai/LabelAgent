"""Read back source-generated points/replies/masks before unchanged scoring."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from run_rgb_layers import ROOT,read,save,sha
from paired_source_cue import body
from semantic_subject_guard import compatible_pallet_subject
from native_prompt_guard import evaluate
import run_source_gap_instances as trial
import audit_soft_prior as audit
def main():
    out=trial.OUT
    if (out/'independent_readback.json').exists():raise FileExistsError('Preserve complete post-freeze evaluation')
    p=read(out/'frozen_plan.json');plan=read(out/'SAM/frozen_plan.json');g=read(out/'SAM/generation_completed.json');rv=read(out/'review_completed.json');assert all(sha(f)==h for f,h in p['protected'].items()) and all(sha(f)==h for f,h in plan['protected'].items());checks=[]
    for row in p['rows']:
        source=Image.open(row['source_image']).convert('RGB');r=next(r for r in rv['rows'] if r['partition']==row['index']);v=row['new_point_views'][0];request=body(*[b['image'] for b in v['bindings']]);assert request==read(v['request_file']) and hashlib.sha256(__import__('json').dumps(request,sort_keys=True).encode()).hexdigest()==v['request_sha256'];assert sha(r['response_file'])==r['response_sha256'];reply=read(r['response_file'])['raw_response'];assert reply['local_vlm']['model_config_sha256']==p['model_config_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256,256] and compatible_pallet_subject(r['decision'])==r['PAL_subject_evidence']
        for b in v['bindings']:
            box=b['source_canvas_box'];clip=b['clipped_box'];canvas=Image.new('RGB',(b['side'],)*2,(127,127,127));canvas.paste(source.crop(tuple(clip)),(clip[0]-box[0],clip[1]-box[1]));np.testing.assert_array_equal(np.asarray(canvas.resize((512,512),Image.Resampling.LANCZOS)),np.asarray(Image.open(b['image']).convert('RGB')))
    for r in g['rows']:
        a=np.load(r['raw_array_file'],allow_pickle=False);m=read(r['metadata_file']);i=r['prediction_call'];native=a[f'{i}_masks'][r['native_index']].squeeze();full=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False);x1,y1,x2,y2=r['box'];np.testing.assert_array_equal(native,full[y1:y2,x1:x2]);assert int(full.sum())==int((native!=0).sum()) and hashlib.sha256(full.tobytes()).hexdigest()==r['mask_sha256'];np.testing.assert_array_equal(a[f'{i}_point_coords'],[[q['x'],q['y']] for q in r['points']]);np.testing.assert_array_equal(a[f'{i}_point_labels'],[q['label'] for q in r['points']]);assert m['prior_mask_sha256']==(None if r['prior'] is None else sha(r['prior'])) and m['image_sha256']==sha(r['input']) and m['source_sha256']==p['source_sha256'] and evaluate(native,r['points'])==r['native_prompt_guard'];checks.append({'index':r['index'],'full_native_bitmap_and_all_actual_inputs_exact':True,'native_masks':sum(len(a[k]) for k in a.files if k.endswith('_masks')),'native_guard':r['native_prompt_guard']})
    arms=[]
    for arm in ['point_only','RGB_gap_prior']:
        folder=out/'audits'/arm;save(folder/'generation_completed.json',{'status':'complete','rows':[r for r in g['rows'] if r['arm']==arm],'source_generation_sha256':sha(out/'SAM/generation_completed.json')});audit.OUT=folder;audit.main();d=read(folder/'support_area_audit.json');arms.append({'arm':arm,'candidates':[{k:r[k] for k in ['index','raw_pixels','extra_pixels','valid','area_fits','person_conflict']} for r in d['candidates']],'radii':{c['case']:[{'radius':r['radius'],'old':r['old_matches'],'with_candidates':r['geometry_area_person_valid_matches'],'ignoring_area_validity':r['unfiltered_all_matches']} for r in c['radii']] for c in d['cases']}})
    save(out/'audit_summary.json',{'arms':arms,'actual_new_SAM_calls':g['actual_new_SAM_calls'],'actual_new_VLM_calls':rv['actual_new_VLM_calls'],'new_Agent_runs':0,'no_adoption':True});save(out/'independent_readback.json',{'passed':True,'rows':checks,'new_positive_source_photos_requests_replies_model_and_tokens_exact':True,'semantic_candidate_selection_all_native_valid_frozen_before_reference_audit':sha(out/'semantic/frozen_plan.json'),'no_output_mask_clipping':True});print('Source gap instances evaluated with original gates',flush=True)
if __name__=='__main__':main()
