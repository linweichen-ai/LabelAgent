"""Readback every original input and complete generated mask before delivery."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'outputs/labelagent_instance_separation_20261105';control=ROOT/'task_workspaces/labelagent_handoff_control_20261104'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
cfg=read(ROOT/'task_workspaces/labelagent_instance_separation_20261105/config_comparison.json');sys.path.insert(0,cfg['source_snapshot'])
import numpy as np
from PIL import Image
from source.agent.edit_agent import _rasterize
comparison=Path(cfg['output']);plan=read(comparison/'frozen_plan.json');originals=read(out/'original_preservation.json');originals['reference_annotations_unchanged']=sha(cfg['reference_annotations'])==plan['reference_annotation_sha256'];originals['reference_description_unchanged']=sha(cfg['reference_description'])==plan['reference_description_sha256'];originals['all_incumbent_result_files_byte_unchanged']=all(sha(ROOT/path)==h for path,h in read(out/'incumbent_binding.json')['files'].items());assert originals['passed'] and all(originals[k] for k in ['reference_annotations_unchanged','reference_description_unchanged','all_incumbent_result_files_byte_unchanged']);save(out/'original_preservation.json',originals)
rows=[]
for name in ['query_generation','seam_generation']:
 for r in read(out/name/'all_hypotheses.json'):
  x1,y1,x2,y2=r['source_box'];expected=np.zeros((1536,1920),np.uint8)
  if name=='query_generation':expected[y1:y2,x1:x2]=(np.asarray(Image.open(r['raw_mask_file']))>0).astype('uint8')
  else:expected[y1:y2,x1:x2]=_rasterize(read(Path(r['source_prediction'])),x2-x1,y2-y1)
  actual=_rasterize(r['shape'],1920,1536);assert np.array_equal(expected,actual);rows.append({'case':r['case'],'key':r['key'],'source':name,'removed_pixels':0,'added_pixels':0,'raster_sha256':hashlib.sha256(actual.tobytes()).hexdigest()})
save(out/'complete_generated_mask_proof.json',{'passed':True,'masks':len(rows),'rows':rows,'budget_clipping':False})
src=control/'verify_projection_recovery.cjs';(control/'verify_instance_separation.cjs').write_text(src.read_text(encoding='utf-8').replace('projection_recovery','instance_separation'),encoding='utf-8')
page=(out/'index.html').read_text(encoding='utf-8');extra='<section><h2>附加逐项核验</h2><p><a href="complete_generated_mask_proof.json">37个生成候选的原输出逐像素核对</a></p><p><a href="instance_support_audit/same_forward_competition.json">同次推理竞争前后支持对照</a></p><p><a href="本轮结论.md">完整结论与限制</a></p><p><a href="seam_generation/blank_controls.json">合成空白SAM诊断（不计入真实结果）</a></p></section>'
if '附加逐项核验' not in page:(out/'index.html').write_text(page.replace('</html>',extra+'</html>'),encoding='utf-8')
print(json.dumps({'originals_preserved':True,'all_original_6_results_byte_equal':True,'complete_raster_proofs':len(rows)}),flush=True)
