"""Full native readback and unchanged evaluation for both frozen anchor trials."""
import hashlib,sys
from pathlib import Path
import numpy as np
from run_rgb_layers import ROOT,read,save,sha
from native_prompt_guard import evaluate
import run_anchor_bracket_prior as first
import audit_soft_prior as audit
def main(name):
    assert name in ['labelagent_anchor_bracket_prior_20261108','labelagent_anchor_image_extent_prior_20261108'];out=first.BASE/name
    if (out/'independent_readback.json').exists():raise FileExistsError('Preserve complete native audit')
    plan=read(out/'frozen_plan.json');g=read(out/'generation_completed.json');assert all(sha(f)==h for f,h in plan['protected'].items());checks=[];native=0
    weightfile=first.BASE/'labelagent_highres_context_cues_20261108/SAM_weights_identity.json';weight=read(weightfile);st=Path(weight['path']).stat();assert st.st_size==weight['bytes'] and st.st_mtime_ns==weight['mtime_ns']
    for r in g['rows']:
        a=np.load(r['raw_array_file'],allow_pickle=False);m=read(r['metadata_file']);i=r['prediction_call'];local=a[f'{i}_masks'][r['native_index']].squeeze();full=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False);x1,y1,x2,y2=r['box'];np.testing.assert_array_equal(local,full[y1:y2,x1:x2]);assert int(full.sum())==int((local!=0).sum()) and hashlib.sha256(full.tobytes()).hexdigest()==r['mask_sha256'];np.testing.assert_array_equal(a[f'{i}_point_coords'],[[q['x'],q['y']] for q in r['points']]);np.testing.assert_array_equal(a[f'{i}_point_labels'],[q['label'] for q in r['points']]);assert evaluate(local,r['points'])==r['native_prompt_guard'] and m['source_sha256']==plan['source_sha256'] and m['image_sha256']==sha(r['input']) and m['prior_mask_sha256']==sha(r['prior']);count=sum(len(a[k]) for k in a.files if k.endswith('_masks'));native+=count
        if r['exact_previous_capture']:assert sha(r['exact_previous_capture'])==r['response_sha256']
        checks.append({'index':r['index'],'all_native_pixels_prior_points_and_labels_exact':True,'native_masks':count,'reused':not r['new_SAM_request']})
    pairs=[]
    for parent in sorted({r['parent'] for r in g['rows']}):
        rows=[r for r in g['rows'] if r['parent']==parent];assert len(rows)==2;one,two=rows;assert one['arm']=='original' and two['arm']=='bracket' and one['points']==two['points'] and one['box']==two['box'] and sha(one['input'])==sha(two['input']) and sha(one['prior'])!=sha(two['prior']);pairs.append({'parent':parent,'same_source_RGB_points_box_and_original_native_selection':True})
    summaries=[]
    for arm in ['original','bracket']:
        folder=out/'audits'/arm;save(folder/'generation_completed.json',{'status':'complete','rows':[r for r in g['rows'] if r['arm']==arm],'source_generation_sha256':sha(out/'generation_completed.json')});audit.OUT=folder;audit.main();d=read(folder/'support_area_audit.json');summaries.append({'arm':arm,'candidates':[{k:r[k] for k in ['index','raw_pixels','extra_pixels','valid','area_fits','person_conflict']} for r in d['candidates']],'radii':{c['case']:[{'radius':r['radius'],'old':r['old_matches'],'with_candidates':r['geometry_area_person_valid_matches'],'ignoring_area_validity':r['unfiltered_all_matches']} for r in c['radii']] for c in d['cases']}})
    save(out/'audit_summary.json',{'arms':summaries,'actual_new_SAM_calls':g['actual_new_SAM_calls'],'exact_reused_requests':g['exact_previous_requests_reused'],'new_VLM_calls':0,'new_Agent_runs':0,'no_adoption':True});save(out/'independent_readback.json',{'passed':True,'rows':checks,'pairs':pairs,'native_masks':native,'weights_bound_to_original_streamed_identity_by_unchanged_stat':{'source_receipt_sha256':sha(weightfile),'not_a_new_full_weights_hash':True,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'expected_sha256':weight['sha256']},'no_output_mask_clipping':True,'generation_frozen_before_reference_evaluation':True});print('Complete native anchor prior audit: '+name,flush=True)
if __name__=='__main__':main(sys.argv[1])
