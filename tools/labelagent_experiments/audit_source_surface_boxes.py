"""Same original evaluator after material-width box generation is frozen."""
from run_rgb_layers import ROOT,read,save,sha
import audit_soft_prior as audit
BASE=ROOT/'outputs/labelagent_source_surface_boxes_20261108';OUT=BASE/'post_freeze_audit'
if OUT.exists():raise FileExistsError('Preserve material-width audit')
p=read(BASE/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());g=read(BASE/'generation_completed.json');save(OUT/'generation_completed.json',{'status':'complete','rows':[{**r,'kind':'real','key':r['request_sha256']} for r in g['rows']],'actual_generation_sha256':sha(BASE/'generation_completed.json'),'native_prompt_guard_before_reference':True,'no_mask_changes':True});audit.OUT=OUT;audit.main()
