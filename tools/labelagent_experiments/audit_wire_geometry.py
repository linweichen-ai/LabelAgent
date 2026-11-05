import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'outputs/labelagent_projection_recovery_20261105'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
cfg=read(ROOT/'task_workspaces/labelagent_projection_recovery_20261105/config_comparison.json');sys.path.insert(0,cfg['source_snapshot']);import numpy as np
from source.agent.edit_agent import _rasterize
rows=[]
for case,selected in read(out/'instance_generation_blue/selection.json')['selected'].items():
 for row in selected:
  if row['source']!='new_true_SAM_instance':continue
  file=Path(row['source_prediction']);r=read(file);p=read(file.parent/'prompt.json');b=p['source_box'];original=np.zeros((1536,1920),np.uint8);original[b[1]:b[3],b[0]:b[2]]=_rasterize(r,b[2]-b[0],b[3]-b[1]);converted=_rasterize(row['shape'],1920,1536);rows.append({'case':case,'key':row['key'],'source':str(file),'removed_pixels':int(((original!=0)&(converted==0)).sum()),'added_pixels':int(((converted!=0)&(original==0)).sum()),'original_pixels':int(original.sum()),'converted_pixels':int(converted.sum())})
receipt={'passed':all(r['removed_pixels']==0 and r['added_pixels']==0 for r in rows),'rows':rows,'scope':'first append variant failed full response-raster preservation; incumbent original masks unaffected','next_action':'direct offset of every original polygon part/hole, no raster contour reconstruction'}
(out/'wire_geometry_failed_variant.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,ensure_ascii=False),flush=True)
