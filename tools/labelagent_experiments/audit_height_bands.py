"""Original acceptance on complete selected masks, after strict native guard."""
from pathlib import Path
import hashlib,numpy as np
from run_rgb_layers import ROOT,read,save,sha
from native_prompt_guard import evaluate
import audit_soft_prior as audit
BASE=ROOT/'outputs/labelagent_height_review_20261108';OUT=BASE/'post_freeze_audit'

def main():
    if OUT.exists():raise FileExistsError('Preserve completed sensor comparison')
    p=read(BASE/'SAM/frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());g=read(BASE/'SAM/generation_completed.json');rows=[]
    for r in g['rows']:
        a=np.load(r['raw_array_file'],allow_pickle=False);meta=read(r['metadata_file']);rr={**r,'key':r['sam_request_sha256'],'kind':'real'}
        if 'mask_sha256' in r:
            local=a[f"{r['prediction_call']}_masks"][r['native_index']].squeeze();guard=evaluate(local,r['points']);rr.update(service_reported_valid=r['valid'],native_prompt_guard=guard,valid=r['valid'] and guard['valid']);assert meta['selected_index']==r['native_index'];x1,y1,x2,y2=r['box'];full=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False);assert hashlib.sha256(full.tobytes()).hexdigest()==r['mask_sha256'];np.testing.assert_array_equal(full[y1:y2,x1:x2],local);assert int(full.sum())==int((local!=0).sum())
        rows.append(rr)
    save(OUT/'generation_completed.json',{'status':'complete','rows':rows,'source_generation_sha256':sha(BASE/'SAM/generation_completed.json'),'no_reference_for_generation':True,'original_native_guard_applied_no_mask_changes':True,'extra_unlabelled_sensor_input_distinct_from_RGB_baseline':True});audit.OUT=OUT;audit.main();d=read(OUT/'support_area_audit.json');summary={'candidates':[{k:r[k] for k in ['index','raw_pixels','extra_pixels','valid','area_fits','person_conflict']} for r in d['candidates']],'radii':{c['case']:[{'radius':r['radius'],'old':r['old_matches'],'with_candidates':r['geometry_area_person_valid_matches']} for r in c['radii']] for c in d['cases']},'actual_new_SAM_calls':g['actual_new_SAM_calls'],'new_Agent_runs':0,'no_adoption':True};save(BASE/'audit_summary.json',summary)

if __name__=='__main__':main()
