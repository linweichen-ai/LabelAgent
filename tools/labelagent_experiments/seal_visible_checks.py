"""Full original byte preservation and every complete mapped raster proof."""
import json,sys,hashlib
from pathlib import Path
from plan_visible_bases import ROOT,work,out,read,save
cfg=read(work/'config_comparison.json');sys.path.insert(0,cfg['source_snapshot']);import numpy as np
from PIL import Image
from source.agent.edit_agent import _rasterize
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();comparison=Path(cfg['output']);plan=read(comparison/'frozen_plan.json');p=read(out/'original_preservation.json');p.update(reference_annotations_unchanged=sha(cfg['reference_annotations'])==plan['reference_annotation_sha256'],reference_description_unchanged=sha(cfg['reference_description'])==plan['reference_description_sha256'],all_incumbent_result_files_byte_unchanged=all(sha(ROOT/f)==h for f,h in read(out/'incumbent_binding.json')['files'].items()));assert p['passed'] and p['reference_annotations_unchanged'] and p['reference_description_unchanged'] and p['all_incumbent_result_files_byte_unchanged'];save(out/'original_preservation.json',p)
rows=[]
for r in read(out/'query_generation/all_hypotheses.json')+read(out/'seam_generation_all/old_and_new_hypotheses.json'):
 x1,y1,x2,y2=r['source_box'];m=np.zeros((1536,1920),np.uint8)
 if r['source']=='raw_trained_instance_query':m[y1:y2,x1:x2]=(np.asarray(Image.open(r['raw_mask_file']))>0).astype('uint8')
 elif r['source']=='aspect_preserving_trained_query':m[y1:y2,x1:x2]=(np.asarray(Image.open(r['source_mask_file']))>0).astype('uint8')
 else:
  response=read(Path(r['source_prediction']));response=response['results'][r['source_result_index']] if 'source_result_index' in r else response;m[y1:y2,x1:x2]=_rasterize(response,x2-x1,y2-y1)
 assert np.array_equal(m,_rasterize(r['shape'],1920,1536));rows.append({'key':r['key'],'case':r['case'],'source':r['source'],'raster_sha256':hashlib.sha256(m.tobytes()).hexdigest(),'removed_pixels':0,'added_pixels':0})
save(out/'complete_generated_mask_proof.json',{'passed':True,'masks':len(rows),'rows':rows,'budget_clipping':False})
old=ROOT/'outputs/labelagent_context_comparison_20261105';seed_proofs=[]
for cam in [0,1]:
 case=f'camera{cam}_frame76';a=read(comparison/'seed_stages'/case/'all_arms.json');b=read(old/'seed_stages'/case/'all_arms.json');assert a==b;seed_proofs.append({'case':case,'all3arm_initial_geometry_identical':True,'sha256':sha(comparison/'seed_stages'/case/'all_arms.json')})
save(out/'initial_seed_reuse_proof.json',{'passed':True,'all6_prior_real_Agent_runs_reused':True,'fresh_Agent_runs':0,'rows':seed_proofs,'not_a_new_measurement_of_convergence':True})
save(out/'planning/protocol_deviation.json',{'declared_output_limit':512,'actual_max_tokens_field_not_sent_without_compact_profile':True,'server_default_not_independently_bound':True,'all_first_variant_point_plans_rejected':True,'not_a_controlled_output_budget_comparison':True})
c=ROOT/'task_workspaces/labelagent_handoff_control_20261104';(c/'verify_visible_base.cjs').write_text((c/'verify_instance_separation.cjs').read_text(encoding='utf-8').replace('instance_separation','visible_base'),encoding='utf-8');print(json.dumps({'complete_masks_verified':len(rows),'all6_initial_seeds_equal':True,'originals_preserved':True}),flush=True)
