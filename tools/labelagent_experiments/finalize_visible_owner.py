"""Explain complete-base identity competition using frozen original point owners."""
import sys,json,hashlib,html
from pathlib import Path
from plan_visible_bases import ROOT,work,out,read,save
cfg=read(work/'config_comparison.json');sys.path[:0]=[cfg['source_snapshot'],str(Path(cfg['source_snapshot'])/'tools'),cfg['dependency_dir']]
import numpy as np,cv2,evaluate_projected_support as ev
from projection_audit_core import frame_data,cameras,matrix_audit
from source.agent.edit_agent import _rasterize
from PIL import Image,ImageDraw,ImageOps,ImageFont
data=frame_data(Path(cfg['dataset']),read(Path(cfg['dataset'])/'import_manifest.json'),76,ev);cams=cameras(Path(cfg['dataset']),read(Path(cfg['dataset'])/'import_manifest.json'),ev);source,pix=ev.project_visible(data['xyz'],cams[0]);inc=ROOT/'outputs/labelagent_context_comparison_20261105';old=matrix_audit(read(inc/'C_hybrid/camera0_frame76/result.json')['shapes'],data,source,pix,ev);matched={data['boxes'][old['eligible'][i]]['obj_id'] for i,j,v in old['pairs']};best=next(r for r in read(out/'aspect_model_generation/all_hypotheses.json') if r['key'].startswith('608ca00f'));new=matrix_audit([best['shape']],data,source,pix,ev);owners=[]
for oid in [42,43]:
 index=next(i+1 for i,b in enumerate(data['boxes']) if b['obj_id']==oid);hits=new['hits'][0]&(data['owner'][source]==index);owners.append({'object_id':oid,'covered_reference_points':int(hits.sum()),'already_matched_by_incumbent':oid in matched})
assert owners[0]['covered_reference_points']==13 and owners[1]['covered_reference_points']==50
receipt={'status':'complete','scope':'post-freeze sparse point ownership, not independent 2D object identity','owners':owners,'target42_purity':13/63,'original_purity_threshold':.67,'all_original_points_retained':True,'no_reference_positions_used_for_generation':True};save(out/'area_diagnostic/complete_base_owner.json',receipt)
photo=Image.open(inc/'inputs/camera0_frame76.png').convert('RGB');mask=_rasterize(best['shape'],1920,1536);ys,xs=np.where(mask);box=(max(0,int(xs.min())-70),max(0,int(ys.min())-70),min(1920,int(xs.max())+71),min(1536,int(ys.max())+71));canvas=Image.new('RGB',(1920,1080),'#142436');d=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21);d.text((25,20),'完整外形为何仍不能恢复独立实例：13个目标点与50个邻实例点',font=font,fill='white')
for j,label in enumerate(['原图','新完整模型查询，未经裁剪','同一完整掩码内的原参考归属']):
 rgb=np.array(photo).copy()
 if j>0:rgb[mask!=0]=(.55*rgb[mask!=0]+.45*np.array([0,220,180])).astype('uint8')
 if j==2:
  for oid,color in [(42,(255,70,70)),(43,(255,220,30))]:
   index=next(i+1 for i,b in enumerate(data['boxes']) if b['obj_id']==oid);hits=new['hits'][0]&(data['owner'][source]==index)
   for x,y in pix[hits]:cv2.circle(rgb,(int(x),int(y)),3,color,-1)
 im=ImageOps.contain(Image.fromarray(rgb).crop(box),(595,720));canvas.paste(im,(25+j*635+(595-im.width)//2,135+(720-im.height)//2));d.text((25+j*635,85),label,font=small,fill='white')
d.text((25,910),f"红：目标42共13点；黄：实例43共50点，旧方案已匹配：{43 in matched}。目标纯度13/63=0.206<0.67。",font=font,fill='white');d.text((25,962),'完整外形可能对应邻近物体；本图只解释原稀疏投影一致性，不当作二维物体真值。',font=small,fill='#d4e6ef');d.text((25,1005),'没有据此删参考点、修改时间/外参、改变原掩码或重新选择提示位置。',font=small,fill='#d4e6ef');file=out/'assets/完整基座的13与50点归属.jpg';canvas.save(file,quality=94)
page=(out/'index.html').read_text(encoding='utf-8');section='<section><h2>完整基座的实例归属核对</h2><p><a href="area_diagnostic/complete_base_owner.json">原参考点归属与旧匹配核对</a></p><img loading="lazy" src="assets/完整基座的13与50点归属.jpg"></section>'
if '完整基座的实例归属核对' not in page:
 (out/'index.html').write_text(page.replace('</html>',section+'</html>'),encoding='utf-8');s=read(out/'public_summary.json');s['pictures']+=1;save(out/'public_summary.json',s);m=read(out/'media_verification.json');m['pictures']+=1;save(out/'media_verification.json',m)
print(json.dumps(receipt,ensure_ascii=False),flush=True)
