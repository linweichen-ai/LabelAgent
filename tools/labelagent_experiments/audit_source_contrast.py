"""Separate fixed photometric arms, preserving original selected full bitmaps."""
from pathlib import Path
import numpy as np
from run_rgb_layers import ROOT,read,save,sha
from native_prompt_guard import evaluate
import audit_soft_prior as audit
OUT=ROOT/'outputs/labelagent_source_contrast_20261108'

def main():
    if (OUT/'audit_summary.json').exists():raise FileExistsError('Preserve completed photometric comparison')
    p=read(OUT/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());g=read(OUT/'generation_completed.json');summary=[]
    for policy in ['source_LAB_CLAHE_clip2_grid8','source_unsharp_sigma1_amount1']:
        rows=[];folder=OUT/'audits'/policy
        for r in g['rows']:
            if r['photometric_policy']!=policy:continue
            a=np.load(r['raw_array_file'],allow_pickle=False);m=a[f"{r['prediction_call']}_masks"][r['native_index']].squeeze();guard=evaluate(m,r['points']);rows.append({**r,'key':r['sam_request_sha256'],'kind':'real','service_reported_valid':r['valid'],'native_prompt_guard':guard,'valid':r['valid'] and guard['valid']})
        assert len(rows)==3;save(folder/'generation_completed.json',{'status':'complete','rows':rows,'source_generation_sha256':sha(OUT/'generation_completed.json'),'original_native_guard_no_mask_change':True,'fixed_photometric_arm_no_reference_selection':True});audit.OUT=folder;audit.main();d=read(folder/'support_area_audit.json');summary.append({'policy':policy,'candidates':[{k:r[k] for k in ['index','raw_pixels','extra_pixels','valid','area_fits','person_conflict']} for r in d['candidates']],'radii':{c['case']:[{'radius':r['radius'],'old':r['old_matches'],'with_candidates':r['geometry_area_person_valid_matches']} for r in c['radii']] for c in d['cases']}})
    save(OUT/'audit_summary.json',{'arms':summary,'actual_new_SAM_calls':g['actual_new_SAM_calls'],'new_VLM_calls':0,'new_Agent_runs':0,'not_adopted':True});print('Fixed source contrast arms audited',flush=True)

if __name__=='__main__':main()
