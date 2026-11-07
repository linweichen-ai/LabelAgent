"""Native-constraint convergence is distinct from strict instance acceptance."""
from pathlib import Path
import hashlib,numpy as np
from run_rgb_layers import read,save
from native_prompt_guard import evaluate
import audit_soft_prior as audit
OUT=Path('D:/AI/Codex/Home/visualizations/2026/11/04/01a1a56c-0d6f-7d61-8326-d3bbd344d2df/labelagent_native_constraint_repair_20261108')

def main():
    if (OUT/'post_freeze_audit').exists():raise FileExistsError('Preserve repair acceptance audit')
    done=read(OUT/'completed.json');rows=[];checks=[]
    for case in done['cases']:
        r=case['rounds'][-1]['row'];a=np.load(r['raw_array_file'],allow_pickle=False);local=a[f"{r['prediction_call']}_masks"][r['native_index']].squeeze();full=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False);x1,y1,x2,y2=r['box'];np.testing.assert_array_equal(full[y1:y2,x1:x2],local);assert int(full.sum())==int((local!=0).sum()) and hashlib.sha256(full.tobytes()).hexdigest()==r['mask_sha256'];guard=evaluate(local,r['points']);assert guard['valid'];rows.append({**r,'key':r['sam_request_sha256'],'kind':'real','service_reported_valid':r['valid'],'native_prompt_guard':guard,'valid':r['valid'] and guard['valid']});initial=case['rounds'][0]['row'];old=np.load(Path(initial['folder'])/'complete_full_mask.npy',allow_pickle=False);checks.append({'source_case':case['source_case'],'requests_total':len(case['rounds']),'cumulative_points':case['per_case_point_annotations'],'initial_positive_and_existing_points_preserved':r['points'][:len(initial['points'])]==initial['points'],'full_new_raw_output_exact':True,'native_pixels_added_by_model':int(((full!=0)&(old==0)).sum()),'native_pixels_removed_by_model':int(((full==0)&(old!=0)).sum()),'no_post_mask_pixel_edit':True})
    folder=OUT/'post_freeze_audit';save(folder/'generation_completed.json',{'status':'complete','rows':rows,'no_reference_coordinates_for_repair':True,'not_Agent_or_instance_convergence':True});audit.OUT=folder;audit.main();d=read(folder/'support_area_audit.json');save(OUT/'independent_readback.json',{'passed':True,'checks':checks,'source_native_constraints_passed':len(rows),'total_point_annotations':done['total_point_annotations'],'maximum_requests_per_case':max(len(c['rounds']) for c in done['cases']),'geometry_counts':{c['case']:[{'radius':r['radius'],'old':r['old_matches'],'final':r['geometry_area_person_valid_matches']} for r in c['radii']] for c in d['cases']},'new_Agent_runs':0,'not_adoption':True});print('Bounded model repair raw pixels and original instance gates independently checked',flush=True)

if __name__=='__main__':main()
