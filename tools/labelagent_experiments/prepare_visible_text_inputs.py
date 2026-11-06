"""Prepare native square photos without pretending a busy VLM returned points."""
from pathlib import Path
import numpy as np
from PIL import Image
from plan_visible_bases import ROOT,work,out,read,save
from visible_base_native_policy import planning_views
dest=out/'native_inputs';dest.mkdir(exist_ok=True);rows=[]
for seed in read(out/'planning/cohort.json'):
 source=Image.open(ROOT/'outputs/labelagent_context_comparison_20261105/inputs'/(seed['case']+'.png')).convert('RGB');mask=(np.asarray(Image.open(Path(seed['folder'])/'seed_mask.png'))>0).astype('uint8');views,details,binding=planning_views(source,mask);folder=dest/f"seed{seed['index']:02d}";folder.mkdir(exist_ok=True);source.crop(tuple(binding['source_box'])).save(folder/'SAM_source_input.png');Image.fromarray(mask*255).save(folder/'seed_mask.png')
 for name,im in views.items():im.save(folder/(name+'.png'))
 for i,im in enumerate(details):im.save(folder/f'detail{i}.png')
 rows.append({**seed,'folder':str(folder),'binding':binding})
save(dest/'cohort.json',rows);save(out/'planning_native/failure.json',{'status':'not_executed_shared_model_busy','inference_responses':0,'retained_failed_client_trace':True,'native_inputs_are_preparation_only':True,'no_other_client_or_service_stopped':True})
save(out/'seam_generation_native/completed.json',{'status':'skipped_shared_model_busy','real_scene_SAM_requests':0,'synthetic_control_requests':0,'valid_complete_outputs':0,'not_a_model_result':True});save(out/'seam_generation_native/all_hypotheses.json',[]);save(out/'seam_generation_native/progress.json',[])
