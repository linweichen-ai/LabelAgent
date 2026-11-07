"""Run unchanged original gates only after source-only generation completes."""
import sys
from run_rgb_layers import ROOT,read,save,sha
import audit_soft_prior as audit
BASE=ROOT/'outputs'/('labelagent_base_blue_surface_20261108' if sys.argv[1]=='blue' else 'labelagent_base_material_20261108')
OUT=BASE/'post_freeze_audit'
if OUT.exists():raise FileExistsError('Preserve surface audit')
p=read(BASE/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());g=read(BASE/'SAM/generation_completed.json');save(OUT/'generation_completed.json',{'status':'complete','rows':[{**r,'kind':'real','key':r['sam_request_sha256']} for r in g['rows']],'original_generation_sha256':sha(BASE/'SAM/generation_completed.json'),'no_mask_changes':True});audit.OUT=OUT;audit.main()
