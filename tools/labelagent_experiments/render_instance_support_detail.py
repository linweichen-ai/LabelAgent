"""Readable zooms from the frozen diagnosis; no prediction or metric updates."""
import sys,json
from pathlib import Path
from projection_audit_core import ROOT,read,frame_data,cameras
cfg=read(ROOT/'task_workspaces/labelagent_instance_separation_20261105/config_comparison.json');sys.path[:0]=[cfg['source_snapshot'],str(Path(cfg['source_snapshot'])/'tools'),cfg['dependency_dir']]
import numpy as np,cv2,evaluate_projected_support as ev
from source.agent.edit_agent import _rasterize
from PIL import Image,ImageOps,ImageDraw,ImageFont
out=ROOT/'outputs/labelagent_instance_separation_20261105';dest=out/'instance_support_audit';audit=read(dest/'completed.json');raw=read(out/'query_generation/all_hypotheses.json')+read(out/'seam_generation/all_hypotheses.json');inc=ROOT/'outputs/labelagent_context_comparison_20261105';data=frame_data(Path(cfg['dataset']),read(Path(cfg['dataset'])/'import_manifest.json'),76,ev);cams=cameras(Path(cfg['dataset']),read(Path(cfg['dataset'])/'import_manifest.json'),ev);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
for case in audit['cases']:
 cam=int(case['case'][6]);source,pixels=ev.project_visible(data['xyz'],cams[cam]);image=Image.open(inc/'inputs'/(case['case']+'.png')).convert('RGB');old=[s for s in read(inc/'C_hybrid'/case['case']/'result.json')['shapes'] if s['label']=='pallet']
 for t in case['targets']:
  oid=t['object_id'];owner=next(i+1 for i,b in enumerate(data['boxes']) if b['obj_id']==oid);xy=pixels[data['owner'][source]==owner];x1,y1=np.maximum(0,xy.min(0)-100);x2,y2=np.minimum([1919,1535],xy.max(0)+100);box=(int(x1),int(y1),int(x2)+1,int(y2)+1);best=t['best_candidates'][0];r=next(r for r in raw if r['key']==best['key'] and r['case']==case['case']);canvas=Image.new('RGB',(1920,1080),'#142436');d=ImageDraw.Draw(canvas);d.text((25,20),case['case']+f' / 对象{oid} / 仅在冻结后核查支持，非二维真值',font=font,fill='white')
  for col,(title,shapes) in enumerate([('原保留前景',[*old]),('原始候选最高支持者（未通过全部验收）',[r['shape']])]):
   rgb=np.array(image).copy();m=np.maximum.reduce([_rasterize(s,1920,1536) for s in shapes]);rgb[m!=0]=(.55*rgb[m!=0]+.45*np.array([0,220,180])).astype('uint8')
   for x,y in xy:cv2.circle(rgb,(int(x),int(y)),3,(255,80,70),-1)
   im=ImageOps.contain(Image.fromarray(rgb).crop(box),(900,730));canvas.paste(im,(30+col*950+(900-im.width)//2,130+(730-im.height)//2));d.text((30+col*950,85),title,font=small,fill='white')
  d.text((28,900),f"原点数{t['visible_reference_points']}；旧有效支持{t['incumbent_best_support']:.3f}；全部原候选最高{t['raw_pool_best_qualified']:.3f}；最终{t['final_best_support']:.3f}",font=font,fill='white')
  d.text((28,948),f"图示候选：{best['source']}；点覆盖{best['raw_point_fraction']:.3f}；纯度{best['purity']:.3f}；新增并集{best['added_union_pixels']}像素",font=small,fill='#d2e3ed');d.text((28,995),'不按参考点裁掩码；原0.67纯度、0.5一对一、面积上限122511.25与其余门槛不变。',font=small,fill='#d2e3ed');canvas.save(dest/(case['case']+f'_obj{oid}.jpg'),quality=94)
print(json.dumps([{'case':c['case'],'object':t['object_id'],'best':t['best_candidates'][0]} for c in audit['cases'] for t in c['targets']],ensure_ascii=False))
