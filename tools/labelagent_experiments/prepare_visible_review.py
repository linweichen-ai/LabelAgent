"""Aggregate complete responses only; reuse exact original semantic checks."""
import json,shutil
from pathlib import Path
from plan_visible_bases import ROOT,work,out,read,save
helpers=Path(__file__).parent;old=ROOT/'outputs/labelagent_target_identity_20261106';previous=ROOT/'outputs/labelagent_instance_separation_20261105';aggregate=out/'seam_generation_all';assert not aggregate.exists();aggregate.mkdir();new=[];progress=[];blank=[];requests=0
for name in ['seam_generation','seam_generation_native','text_generation']:
 folder=out/name;assert read(folder/'completed.json')['status'] in ['complete','skipped_shared_model_busy'];new+=read(folder/'all_hypotheses.json');requests+=read(folder/'completed.json')['real_scene_SAM_requests'];progress+=read(folder/'progress.json')
 if (folder/'blank_controls.json').exists():blank+=read(folder/'blank_controls.json')['rows']
new+=read(out/'aspect_model_generation/all_hypotheses.json');save(aggregate/'all_hypotheses.json',new);save(aggregate/'progress.json',progress);save(aggregate/'blank_controls.json',{'rows':blank,'not_accuracy':True});save(aggregate/'completed.json',{'status':'complete','real_scene_SAM_requests':requests,'synthetic_control_requests':len(blank),'valid_complete_outputs':len(new),'fresh_CPU_mask2former_forwards':8,'reference_inputs':False,'no_output_clipping':True})
oldSAM=read(old/'seam_generation/old_and_new_hypotheses.json');save(aggregate/'old_and_new_hypotheses.json',oldSAM+new);query=out/'query_generation';query.mkdir();shutil.copy2(old/'query_generation/all_hypotheses.json',query/'all_hypotheses.json');shutil.copy2(old/'query_generation/completed.json',query/'completed.json')
dest=out/'instance_generation';dest.mkdir();shutil.copytree(old/'instance_generation/structure_cache',dest/'structure_cache');shutil.copytree(old/'instance_generation/neutral_cache',dest/'neutral_cache')
src=(helpers/'review_target_identity.py').read_text(encoding='utf-8').replace('labelagent_target_identity_20261106','labelagent_visible_base_20261107').replace("out/'seam_generation/","out/'seam_generation_all/")
(helpers/'review_visible_base.py').write_text(src,encoding='utf-8')
