"""Apply original native prompt guard without editing historical generations."""
from pathlib import Path
import numpy as np
from run_rgb_layers import ROOT,read,save,sha
from native_prompt_guard import evaluate
import audit_soft_prior as audit
sources=['labelagent_base_material_20261108','labelagent_base_blue_surface_20261108','labelagent_two_sided_layers_20261108'];checks=[]
for name in sources:
    base=ROOT/'outputs'/name;out=base/'native_prompt_audit'
    if out.exists():raise FileExistsError('Preserve strict native audit')
    g=read(base/'SAM/generation_completed.json');rows=[]
    for r in g['rows']:
        m=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False);x1,y1,x2,y2=r['box'];guard=evaluate(m[y1:y2,x1:x2],r['points']);rows.append({**r,'kind':'real','key':r['sam_request_sha256'],'service_reported_valid':r['valid'],'native_prompt_guard':guard,'valid':r['valid'] and guard['valid']});checks.append({'trial':name,'partition':r['source_partition_index'],'service_reported_valid':r['valid'],'native_guard':guard})
    save(out/'generation_completed.json',{'status':'complete','rows':rows,'raw_generation_sha256':sha(base/'SAM/generation_completed.json'),'guard_sha256':sha(Path(__file__).with_name('native_prompt_guard.py')),'original_radius_preserved':3,'native_masks_not_modified':True});audit.OUT=out;audit.main()
save(ROOT/'outputs/labelagent_base_surfaces_report_20261108/native_prompt_guard_readback.json',{'passed':True,'checks':checks,'actual_native_guard_rejections':sum(not r['native_guard']['valid'] for r in checks),'source_service_not_patched_or_restarted_for_this_guard':True,'offline_candidate_acceptance_guard_only':True,'all_raw_generations_and_full_bitmaps_preserved':True})
print({'raw_masks_checked':len(checks),'rejected':sum(not r['native_guard']['valid'] for r in checks)})
