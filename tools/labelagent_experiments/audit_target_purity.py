"""Post-freeze purity lineage; every source point and whole mask is retained."""
import json,sys,hashlib
from pathlib import Path
from projection_audit_core import ROOT,read,frame_data,cameras,matrix_audit
out=ROOT/'outputs/labelagent_target_identity_20261106';cfg=read(ROOT/'task_workspaces/labelagent_target_identity_20261106/config_comparison.json');sys.path[:0]=[cfg['source_snapshot'],str(Path(cfg['source_snapshot'])/'tools'),cfg['dependency_dir']]
import numpy as np,cv2,evaluate_projected_support as ev
from source.agent.edit_agent import _rasterize
from PIL import Image,ImageDraw,ImageFont,ImageOps
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert read(Path(cfg['output'])/'completed.json')['status']=='complete';dest=out/'instance_support_audit';dest.mkdir(exist_ok=True)
if (dest/'completed.json').exists():raise FileExistsError('Frozen purity audit')
data=frame_data(Path(cfg['dataset']),read(Path(cfg['dataset'])/'import_manifest.json'),76,ev);cams=cameras(Path(cfg['dataset']),read(Path(cfg['dataset'])/'import_manifest.json'),ev);allrows=read(out/'query_generation/all_hypotheses.json')+read(out/'seam_generation/old_and_new_hypotheses.json');cases=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21);inc=ROOT/'outputs/labelagent_context_comparison_20261105'
for cam in [0,1]:
 case=f'camera{cam}_frame76';source,pix=ev.project_visible(data['xyz'],cams[cam]);sh=read(inc/'C_hybrid'/case/'result.json')['shapes'];old=matrix_audit(sh,data,source,pix,ev);raw=[r for r in allrows if r['case']==case];trial=matrix_audit(sh+[r['shape'] for r in raw],data,source,pix,ev);targets=[]
 for oid in ([42,48] if cam==0 else [47,49]):
  k=next(k for k,i in enumerate(old['eligible']) if data['boxes'][i]['obj_id']==oid);rs=[]
  for j,r in enumerate(raw,len(old['shapes'])):
   extra=int((trial['masks'][j]&~np.maximum.reduce(old['masks'])).sum());known=trial['hits'][j]&trial['known'];ownership=[]
   for v in np.unique(data['owner'][source][known]):
    ownership.append({'object_id':data['boxes'][int(v)-1]['obj_id'],'points':int((known&(data['owner'][source]==v)).sum())})
   rs.append({'key':r['key'],'source':r['source'],'new_SAM':str(out/'seam_generation') in r['source_prediction'],'raw_point_fraction':float(trial['raw'][k,j]),'purity':float(trial['purity'][k,j]),'qualified_support':float(trial['support'][k,j]),'added_union_pixels':extra,'fits_remaining_area_alone':extra<=209.25,'known_PAL_point_ownership':ownership})
  rs.sort(key=lambda r:(-r['qualified_support'],-r['raw_point_fraction'],-r['purity'],r['key']));best=rs[0];new=[r for r in rs if r['new_SAM']];oldSAM=[r for r in rs if r['source']=='seam_guided_SAM' and not r['new_SAM']];t={'object_id':oid,'visible_reference_points':int((data['owner'][source]==old['eligible'][k]+1).sum()),'incumbent_best_support':float(old['support'][k].max()),'raw_pool_best_raw':max(r['raw_point_fraction'] for r in rs),'raw_pool_best_qualified':best['qualified_support'],'new_SAM_best_raw':max([r['raw_point_fraction'] for r in new]+[0]),'new_SAM_best_support':max([r['qualified_support'] for r in new]+[0]),'old_SAM_best_support':max([r['qualified_support'] for r in oldSAM]+[0]),'best_candidates':rs};targets.append(t)
  image=Image.open(inc/'inputs'/(case+'.png')).convert('RGB');xy=pix[data['owner'][source]==old['eligible'][k]+1];xymin=np.maximum(0,xy.min(0)-100);xymax=np.minimum([1919,1535],xy.max(0)+100);box=tuple(map(int,[*xymin,xymax[0]+1,xymax[1]+1]));selected=max(new,key=lambda r:(r['qualified_support'],r['raw_point_fraction'],r['purity'])) if new else best;row=next(r for r in raw if r['key']==selected['key']);canvas=Image.new('RGB',(1920,1080),'#142436');d=ImageDraw.Draw(canvas);d.text((30,25),case+f' / 对象{oid} / 原参考冻结后的纯度诊断，非二维真值',font=font,fill='white')
  for col,(title,shapes) in enumerate([('保留13实例前景',sh),('本轮新SAM最佳者（未通过最终验收）',[row['shape']])]):
   masks=[_rasterize(s,1920,1536) for s in shapes if s['label']=='pallet'];m=np.maximum.reduce(masks);rgb=np.array(image).copy();rgb[m!=0]=(.55*rgb[m!=0]+.45*np.array([0,220,180])).astype('uint8')
   for x,y in xy:cv2.circle(rgb,(int(x),int(y)),3,(255,80,70),-1)
   im=ImageOps.contain(Image.fromarray(rgb).crop(box),(900,730));canvas.paste(im,(30+col*950+(900-im.width)//2,130+(730-im.height)//2));d.text((30+col*950,85),title,font=small,fill='white')
  d.text((30,905),f"点数{t['visible_reference_points']}；新SAM最高点覆盖{t['new_SAM_best_raw']:.3f}；新SAM最高有效支持{t['new_SAM_best_support']:.3f}",font=font,fill='white');d.text((30,950),f"图示新候选：覆盖{selected['raw_point_fraction']:.3f}，纯度{selected['purity']:.3f}，新增并集{selected['added_union_pixels']}像素。",font=small,fill='#d5e7ef');d.text((30,995),'不删冲突参考点、不裁候选；原纯度0.67、支持0.5、人员保护与面积预算不变。',font=small,fill='#d5e7ef');canvas.save(dest/(case+f'_obj{oid}.jpg'),quality=94)
 cases.append({'case':case,'targets':targets,'incumbent_instances':len(old['pairs']),'unfiltered_pool_diagnostic_matches':len(trial['pairs'])})
save(dest/'completed.json',{'status':'complete','reference_read_only_after_freeze':True,'cases':cases,'raw_hypotheses':len(allrows),'original_scorer_sha256':hashlib.sha256((Path(cfg['source_snapshot'])/'tools/evaluate_projected_support.py').read_bytes()).hexdigest(),'no_reference_or_mask_modification':True});print(json.dumps([{k:v for k,v in t.items() if k!='best_candidates'} for c in cases for t in c['targets']],ensure_ascii=False),flush=True)
