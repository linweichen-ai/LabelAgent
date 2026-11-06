"""Verify original byte identity and every reused/new raw candidate raster."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'outputs/labelagent_target_identity_20261106';work=ROOT/'task_workspaces/labelagent_target_identity_20261106';control=ROOT/'task_workspaces/labelagent_handoff_control_20261104'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=read(work/'config_comparison.json');sys.path.insert(0,cfg['source_snapshot']);import numpy as np
from PIL import Image
from source.agent.edit_agent import _rasterize
plan=read(Path(cfg['output'])/'frozen_plan.json');p=read(out/'original_preservation.json');p.update(reference_annotations_unchanged=sha(cfg['reference_annotations'])==plan['reference_annotation_sha256'],reference_description_unchanged=sha(cfg['reference_description'])==plan['reference_description_sha256'],all_incumbent_result_files_byte_unchanged=all(sha(ROOT/f)==v for f,v in read(out/'incumbent_binding.json')['files'].items()));assert p['passed'] and p['reference_annotations_unchanged'] and p['reference_description_unchanged'] and p['all_incumbent_result_files_byte_unchanged'];save(out/'original_preservation.json',p)
rows=[]
for r in read(out/'query_generation/all_hypotheses.json')+read(out/'seam_generation/old_and_new_hypotheses.json'):
 x1,y1,x2,y2=r['source_box'];m=np.zeros((1536,1920),np.uint8)
 if r['source']=='raw_trained_instance_query':m[y1:y2,x1:x2]=(np.asarray(Image.open(r['raw_mask_file']))>0).astype('uint8')
 else:m[y1:y2,x1:x2]=_rasterize(read(Path(r['source_prediction'])),x2-x1,y2-y1)
 actual=_rasterize(r['shape'],1920,1536);assert np.array_equal(actual,m);rows.append({'key':r['key'],'case':r['case'],'source':r['source'],'removed_pixels':0,'added_pixels':0,'raster_sha256':hashlib.sha256(m.tobytes()).hexdigest()})
save(out/'complete_generated_mask_proof.json',{'passed':True,'masks':len(rows),'rows':rows,'budget_clipping':False})
source=control/'verify_instance_separation.cjs';(control/'verify_target_identity.cjs').write_text(source.read_text(encoding='utf-8').replace('instance_separation','target_identity'),encoding='utf-8')
print(json.dumps({'all_originals_preserved':True,'complete_masks_verified':len(rows)}),flush=True)
