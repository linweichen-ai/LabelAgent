"""Original evaluator, post-freeze, for complete five-point layer hypotheses."""
from run_rgb_layers import ROOT,read,save,sha
import audit_soft_prior as audit
BASE=ROOT/'outputs/labelagent_two_sided_layers_20261108';OUT=BASE/'post_freeze_audit'
if OUT.exists():raise FileExistsError('Preserve two-sided audit')
p=read(BASE/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());g=read(BASE/'SAM/generation_completed.json');save(OUT/'generation_completed.json',{'status':'complete','rows':[{**r,'kind':'real','key':r['sam_request_sha256']} for r in g['rows']],'raw_generation_sha256':sha(BASE/'SAM/generation_completed.json'),'no_mask_changes':True});audit.OUT=OUT;audit.main()
