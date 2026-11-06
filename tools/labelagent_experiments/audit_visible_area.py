"""Read-only area/connectedness evidence, never used to trim a candidate."""
import json,sys,hashlib
from pathlib import Path
from plan_visible_bases import ROOT,work,out,read,save
cfg=read(work/'config.json');sys.path.insert(0,cfg['source_snapshot'])
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont,ImageOps
from source.agent.edit_agent import _rasterize
old=ROOT/'outputs/labelagent_target_identity_20261106';rows=read(old/'seam_generation/all_hypotheses.json');dest=out/'area_diagnostic';dest.mkdir(exist_ok=True);records=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
for row in rows:
 case=row['case'];source=Image.open(ROOT/'outputs/labelagent_context_comparison_20261105/inputs'/(case+'.png')).convert('RGB');inc=read(ROOT/'outputs/labelagent_context_comparison_20261105/C_hybrid'/case/'result.json')['shapes'];union=np.maximum.reduce([_rasterize(s,1920,1536) for s in inc if s['label']=='pallet']);mask=_rasterize(row['shape'],1920,1536);extra=mask&~union;n,cc,stats,_=cv2.connectedComponentsWithStats(mask,8);components=[{'component':i,'pixels':int(stats[i,cv2.CC_STAT_AREA]),'extra_union_pixels':int((extra&(cc==i)).sum())} for i in range(1,n)];components.sort(key=lambda r:-r['pixels']);yy,xx=np.where(mask);box=[max(0,int(xx.min())-50),max(0,int(yy.min())-75),min(1920,int(xx.max())+51),min(1536,int(yy.max())+76)];record={'case':case,'key':row['key'],'mask_pixels':int(mask.sum()),'extra_union_pixels':int(extra.sum()),'remaining_original_area_budget':209.25,'fits_complete_mask':int(extra.sum())<=209.25,'components':components,'largest_component_extra_pixels':components[0]['extra_union_pixels'],'raster_unchanged':True,'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),'reference_points_not_read':True};records.append(record);canvas=Image.new('RGB',(1920,1080),'#142436');d=ImageDraw.Draw(canvas);d.text((25,20),case+' / 完整旧候选面积核查 '+row['key'][:8],font=font,fill='white')
 for j,(label,flag,color) in enumerate([('原始照片',None,None),('完整原始SAM输出',mask,np.array([0,220,170])),('新增并集区域（仅显示，不删除）',extra,np.array([255,90,80]))]):
  rgb=np.array(source).copy()
  if flag is not None:rgb[flag!=0]=(.5*rgb[flag!=0]+.5*color).astype('uint8')
  im=ImageOps.contain(Image.fromarray(rgb).crop(tuple(box)),(595,720));canvas.paste(im,(25+j*635+(595-im.width)//2,135+(720-im.height)//2));d.text((25+j*635,85),label,font=small,fill='white')
 d.text((25,910),f"完整掩码{int(mask.sum())}像素；新增{int(extra.sum())}；原剩余预算209.25；最大连通部分新增{components[0]['extra_union_pixels']}。",font=font,fill='white');d.text((25,960),'所有原parts/holes和像素保持；红色仅用于解释面积，不是下一轮SAM负点或裁剪模板。',font=small,fill='#d5e7ef');d.text((25,1002),'新提示从实际可见结构产生，不能按预算删像素或把参考点位置作为目标范围。',font=small,fill='#d5e7ef');canvas.save(dest/(row['key'][:12]+'.jpg'),quality=94)
save(dest/'completed.json',{'status':'complete','rows':records,'no_prediction_changes':True,'no_area_based_prompt_generation':True});print(json.dumps([r for r in records if r['key'].startswith('2d854b6e')],ensure_ascii=False),flush=True)
