"""Identify exact prior inputs without treating sparse support as visible truth."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'outputs/labelagent_target_identity_20261106';old=ROOT/'outputs/labelagent_instance_separation_20261105';cfg=json.loads((ROOT/'task_workspaces/labelagent_target_identity_20261106/config.json').read_text(encoding='utf-8'));sys.path.insert(0,cfg['source_snapshot'])
import numpy as np,cv2
from PIL import Image
from source.agent.edit_agent import _rasterize
from target_identity_policy import centered_photos,extent_views,png
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
key='fb1434b8b2962bb9057f9db39e89f281dd0c646af623af50f74ff2556efb8b6f';row=next(r for r in read(old/'seam_generation/all_hypotheses.json') if r['key']==key);mask=_rasterize(row['shape'],1920,1536);photo=Image.open(ROOT/'outputs/labelagent_context_comparison_20261105/inputs'/f"{row['case']}.png").convert('RGB');ys,xs=np.where(mask);margin=max(32,int(max(xs.max()-xs.min(),ys.max()-ys.min())*.5));boxes=[[max(0,int(xs.min())-8),max(0,int(ys.min())-8),min(1920,int(xs.max())+9),min(1536,int(ys.max())+9)],[max(0,int(xs.min())-margin),max(0,int(ys.min())-margin),min(1920,int(xs.max())+margin+1),min(1536,int(ys.max())+margin+1)]];folder=out/'target_12of16_diagnostic';folder.mkdir(exist_ok=True);rows=[]
for j,b in enumerate(boxes):
 crop=photo.crop(tuple(b));scale=max(1,512/max(crop.size));crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.LANCZOS);raw=png(crop);h=hashlib.sha256(raw).hexdigest();cache=next(read(p) for p in (old/'instance_generation_consistency/neutral_cache').glob('*.json') if read(p)['image_sha256']==h);center=[(b[0]+b[2]-1)//2,(b[1]+b[3]-1)//2];(folder/f'original_neutral_{j}.png').write_bytes(raw);rows.append({'source_box':b,'input_sha256':h,'original_cached_decision':cache['decision'],'center_source_pixel':center,'center_in_candidate':bool(mask[center[1],center[0]]),'candidate_fraction_of_source_box':float(mask[b[1]:b[3],b[0]:b[2]].mean())})
views,binding=centered_photos(photo,mask)
for name,im in views.items():im.save(folder/(name+'.png'))
im,binary,eb=extent_views(photo,mask);im.save(folder/'extent_photo.png');binary.save(folder/'extent_binary.png');save(folder/'geometry.json',{'candidate_key':key,'candidate_case':row['case'],'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),'mask_pixels':int(mask.sum()),'bbox_width_height':[int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)],'old_neutral_inputs':rows,'new_centered_binding':binding,'extent_binding':eb,'no_reference_used_in_view_generation':True,'diagnostic_only_no_adoption':True});print(json.dumps({'old_inputs':rows,'mask_pixels':int(mask.sum()),'bbox_width_height':[int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)]},ensure_ascii=False),flush=True)
