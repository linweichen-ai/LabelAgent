"""Unchanged reference evaluator after original native point-neighborhood guard."""
from pathlib import Path
import hashlib,numpy as np
from run_rgb_layers import read,save,sha
from native_prompt_guard import evaluate
import audit_soft_prior as audit
OUT=Path('D:/AI/Codex/Home/visualizations/2026/11/04/01a1a56c-0d6f-7d61-8326-d3bbd344d2df/labelagent_mask_graph_collision_20261108')

def main():
    if (OUT/'audit_summary.json').exists():raise FileExistsError('Preserve complete mask-conflict comparison')
    p=read(OUT/'SAM/frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());g=read(OUT/'SAM/generation_completed.json');summaries=[];native_records=[]
    for arm in ['neighbor_negatives','exclusive_surface_positives']:
        rows=[];folder=OUT/'audits'/arm
        for r in g['rows']:
            if r['arm']!=arm:continue
            a=np.load(r['raw_array_file'],allow_pickle=False);meta=read(r['metadata_file']);row={**r,'key':r['sam_request_sha256'],'kind':'real'}
            if 'mask_sha256' in r:
                local=a[f"{r['prediction_call']}_masks"][r['native_index']].squeeze();full=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False);x1,y1,x2,y2=r['box'];np.testing.assert_array_equal(full[y1:y2,x1:x2],local);assert int(full.sum())==int((local!=0).sum()) and hashlib.sha256(full.tobytes()).hexdigest()==r['mask_sha256'];guard=evaluate(local,r['points']);row.update(service_reported_valid=r['valid'],native_prompt_guard=guard,valid=r['valid'] and guard['valid']);assert meta['points']==r['points'] and meta['selected_index']==r['native_index'];np.testing.assert_array_equal(a[f"{r['prediction_call']}_point_coords"],[[q['x'],q['y']] for q in r['points']]);np.testing.assert_array_equal(a[f"{r['prediction_call']}_point_labels"],[q['label'] for q in r['points']]);native_records.append({'cue':r['index'],'native_shapes':{name:list(a[name].shape) for name in a.files if name.endswith('_masks')},'metadata_multimask':[x.get('multimask_output') for x in meta['predictions']],'all_source_points_labels_and_full_selected_bitmap_exact':True,'native_guard':guard})
            rows.append(row)
        save(folder/'generation_completed.json',{'status':'complete','rows':rows,'source_generation_sha256':sha(OUT/'SAM/generation_completed.json'),'all_complete_selected_masks_preserved_no_reference_for_generation':True});audit.OUT=folder;audit.main();d=read(folder/'support_area_audit.json');summaries.append({'arm':arm,'candidates':[{k:r[k] for k in ['index','raw_pixels','extra_pixels','valid','area_fits','person_conflict']} for r in d['candidates']],'radii':{c['case']:[{'radius':r['radius'],'old':r['old_matches'],'with_candidates':r['geometry_area_person_valid_matches'],'ignoring_area_validity':r['unfiltered_all_matches']} for r in c['radii']] for c in d['cases']}})
    save(OUT/'native_readback.json',{'passed':True,'rows':native_records,'original_multimask_policy_not_forced':True,'source_sampling_region_not_returned_as_mask':True});save(OUT/'audit_summary.json',{'arms':summaries,'actual_new_SAM_calls':g['actual_new_SAM_calls'],'new_VLM_calls':read(OUT/'review_completed.json')['actual_new_VLM_calls'],'new_Agent_runs':0,'no_adoption':True});print('Source mask-conflict arms evaluated with original gates',flush=True)

if __name__=='__main__':main()
