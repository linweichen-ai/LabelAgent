"""Adapt immutable new captures to the unchanged, post-freeze evaluator."""
import os
from pathlib import Path
import audit_soft_prior as audit
from run_rgb_layers import read,save,sha,ROOT
SAM=ROOT/'outputs/labelagent_trained_point_sam_20261108'
OUT=SAM/'post_freeze_audit'
if OUT.exists():raise FileExistsError('Preserve post-freeze audit')
p=read(SAM/'frozen_plan.json');g=read(SAM/'generation_completed.json');assert all(sha(f)==h for f,h in p['protected'].items())
rows=[{**r,'kind':'real','key':r['sam_request_sha256']} for r in g['rows']]
save(OUT/'generation_completed.json',{'status':'complete','rows':rows,'immutable_source_generation_sha256':sha(SAM/'generation_completed.json'),'wrapper_only_no_mask_changes':True})
audit.OUT=OUT;audit.main()
binding=read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'];assert all(sha(ROOT/f)==h for f,h in binding.items())
save(OUT/'original_readback.json',{'passed':True,'all_six_original_results_exact':True,'adopted':False,'new_Agent_runs':0})
