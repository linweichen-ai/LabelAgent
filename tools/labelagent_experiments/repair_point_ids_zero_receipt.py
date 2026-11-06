"""Document a missing empty progress list without inventing a model request."""
from point_ids_common import *
folder=OUT/'seam_generation';r=read(folder/'completed.json');plans=read(OUT/'planning/completed.json')
assert r['status']=='complete' and r['real_scene_SAM_requests']==r['synthetic_control_requests']==r['valid_complete_outputs']==r['model_trace_records']==plans['valid_prompts']==0
assert read(folder/'all_hypotheses.json')==[] and read(folder/'blank_controls.json')['rows']==[] and not (folder/'progress.json').exists()
save(folder/'progress.json',[])
save(OUT/'engineering_failures/zero_generation_receipt/repair.json',{'issue':'zero-plan generator completed but omitted empty progress.json','evidence':'planning valid_prompts=0; SAM scene/control/trace counts all0; hypotheses and blank rows empty','repair':'write the derived empty progress list only','model_requests_added':0,'predictions_modified':False,'executed_pre_fix_generator_preserved':True})
print('Recorded explicit empty generation progress; no model call',flush=True)
