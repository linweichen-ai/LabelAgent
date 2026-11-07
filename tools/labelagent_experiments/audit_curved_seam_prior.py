"""Independent native readback and unchanged post-generation evaluation."""
import hashlib
from pathlib import Path
import numpy as np
from run_rgb_layers import read,save,sha
import run_curved_seam_prior as trial
import audit_soft_prior as audit
def main():
    out=trial.OUT
    if (out/'independent_readback.json').exists():raise FileExistsError('Preserve completed evaluation')
    p=read(out/'frozen_plan.json');g=read(out/'generation_completed.json');assert all(sha(f)==h for f,h in p['protected'].items());checks=[]
    for r in g['rows']:
        m=read(r['metadata_file']);a=np.load(r['raw_array_file'],allow_pickle=False);i=r['prediction_call'];local=a[f'{i}_masks'][r['native_index']].squeeze();full=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False);x1,y1,x2,y2=r['box'];np.testing.assert_array_equal(local,full[y1:y2,x1:x2]);assert int(full.sum())==int((local!=0).sum()) and hashlib.sha256(full.tobytes()).hexdigest()==r['mask_sha256'];np.testing.assert_array_equal(a[f'{i}_point_coords'],[[q['x'],q['y']] for q in r['points']]);np.testing.assert_array_equal(a[f'{i}_point_labels'],[q['label'] for q in r['points']]);assert m['prior_mask_sha256']==sha(r['prior']);checks.append({'index':r['index'],'source_and_prior_digest_exact':True,'all_full_native_pixels_and_prompt_arrays_exact':True,'native_shapes':{k:list(a[k].shape) for k in a.files if k.endswith('_masks')}})
    summaries=[]
    for arm in ['original','curved']:
        folder=out/'audits'/arm;rows=[r for r in g['rows'] if r['arm']==arm];save(folder/'generation_completed.json',{'status':'complete','rows':rows,'source_generation_sha256':sha(out/'generation_completed.json')});audit.OUT=folder;audit.main();d=read(folder/'support_area_audit.json');summaries.append({'arm':arm,'candidates':[{k:r[k] for k in ['index','raw_pixels','extra_pixels','valid','area_fits','person_conflict']} for r in d['candidates']],'radii':{c['case']:[{'radius':r['radius'],'old':r['old_matches'],'with_candidates':r['geometry_area_person_valid_matches'],'ignoring_area_validity':r['unfiltered_all_matches']} for r in c['radii']] for c in d['cases']}})
    save(out/'audit_summary.json',{'arms':summaries,'actual_new_SAM_calls':g['actual_new_SAM_calls'],'new_VLM_calls':0,'positive_semantics_exact_previous_review_not_new_whole_base_review':True,'new_Agent_runs':0,'no_adoption':True});save(out/'independent_readback.json',{'passed':True,'rows':checks,'no_output_mask_clipping':True,'generation_frozen_before_reference_audit':True});print('Complete curved-seam native outputs independently evaluated',flush=True)
if __name__=='__main__':main()
