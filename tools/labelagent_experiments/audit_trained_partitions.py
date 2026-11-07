"""Original full-mask evaluator after frozen cross-teacher generation ends."""
from run_rgb_layers import ROOT,read,save,sha
import audit_soft_prior as audit
BASE=ROOT/'outputs/labelagent_trained_partition_20261108'
OUT=BASE/'post_freeze_audit'
if OUT.exists():raise FileExistsError('Preserve evaluated partition cohort')
plan=read(BASE/'partition_plan.json');assert all(sha(f)==h for f,h in plan['protected'].items());g=read(BASE/'SAM/generation_completed.json')
save(OUT/'generation_completed.json',{'status':'complete','rows':[{**r,'kind':'real','key':r['sam_request_sha256']} for r in g['rows']],'immutable_raw_generation_sha256':sha(BASE/'SAM/generation_completed.json'),'no_mask_changes':True})
audit.OUT=OUT;audit.main()
