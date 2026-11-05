"""Post-freeze diagnostic only; never fed back into candidate selection."""
import json,sys,hashlib,collections
from pathlib import Path
from projection_audit_core import ROOT,read,frame_data,cameras,matrix_audit
out=ROOT/'outputs/labelagent_instance_separation_20261105';comparison=ROOT/'outputs/labelagent_instance_separation_comparison_20261105'
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 assert read(comparison/'completed.json')['status']=='complete'
 cfg=read(ROOT/'task_workspaces/labelagent_instance_separation_20261105/config_comparison.json');sys.path[:0]=[cfg['source_snapshot'],str(Path(cfg['source_snapshot'])/'tools'),cfg['dependency_dir']]
 import numpy as np,cv2,evaluate_projected_support as ev
 from source.agent.edit_agent import _rasterize
 from PIL import Image,ImageDraw,ImageFont
 dataset=Path(cfg['dataset']);manifest=read(dataset/'import_manifest.json');data=frame_data(dataset,manifest,76,ev);cams=cameras(dataset,manifest,ev);inc=ROOT/'outputs/labelagent_context_comparison_20261105';dest=out/'instance_support_audit';dest.mkdir(exist_ok=True)
 if (dest/'completed.json').exists():raise FileExistsError('Frozen diagnostic')
 allrows=read(out/'query_generation/all_hypotheses.json')+read(out/'seam_generation/all_hypotheses.json');cases=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
 for cam in [0,1]:
  case=f'camera{cam}_frame76';source,pixels=ev.project_visible(data['xyz'],cams[cam]);oldsh=read(inc/'C_hybrid'/case/'result.json')['shapes'];newsh=read(comparison/'C_hybrid'/case/'result.json')['shapes'];rows=[r for r in allrows if r['case']==case];old=matrix_audit(oldsh,data,source,pixels,ev);trial=matrix_audit(newsh,data,source,pixels,ev);pool=matrix_audit(oldsh+[r['shape'] for r in rows],data,source,pixels,ev);targets=[]
  for oid in ([42,48] if cam==0 else [47,49]):
   k=next(k for k,i in enumerate(old['eligible']) if data['boxes'][i]['obj_id']==oid);n=int((data['owner'][source]==old['eligible'][k]+1).sum());candidate_rows=[]
   for j,r in enumerate(rows,len(old['shapes'])):
    mask=pool['masks'][j];extra=int((mask&~np.maximum.reduce(old['masks'])).sum());candidate_rows.append({'key':r['key'],'source':r['source'],'raw_point_fraction':float(pool['raw'][k,j]),'purity':float(pool['purity'][k,j]),'qualified_support':float(pool['support'][k,j]),'added_union_pixels':extra,'fits_remaining_area_alone':extra<=209.25,'source_prediction':r['source_prediction']})
   ranked=sorted(candidate_rows,key=lambda r:(-r['qualified_support'],-r['raw_point_fraction'],-r['purity'],r['key']));targets.append({'object_id':oid,'visible_reference_points':n,'incumbent_best_raw':float(old['raw'][k].max()),'incumbent_best_support':float(old['support'][k].max()),'final_best_raw':float(trial['raw'][k].max()),'final_best_support':float(trial['support'][k].max()),'raw_pool_best_raw':max([r['raw_point_fraction'] for r in ranked]+[0]),'raw_pool_best_qualified':max([r['qualified_support'] for r in ranked]+[0]),'best_candidates':ranked,'matching_unchanged':True})
   # The diagnostic is explicitly a reference projection, never dense truth.
   image=Image.open(inc/'inputs'/(case+'.png')).convert('RGB');selected=(data['owner'][source]==old['eligible'][k]+1);xy=pixels[selected];x1,y1=np.maximum(0,xy.min(0)-100);x2,y2=np.minimum([1919,1535],xy.max(0)+100);b=[int(x1),int(y1),int(x2)+1,int(y2)+1]
   panel=Image.new('RGB',(1920,1080),'#152638');d=ImageDraw.Draw(panel);d.text((28,22),f'{case} / obj {oid} / 冻结后的原投影诊断，非二维真值',font=font,fill='white')
   for col,(label,shapes) in enumerate([('保留13实例方案',oldsh),('原始候选中支持最高者（未通过最终验收）',[next(r['shape'] for r in rows if r['key']==ranked[0]['key'])] if ranked else [])]):
    rgb=np.array(image).copy();masks=[_rasterize(s,1920,1536) for s in shapes if s['label']=='pallet'];u=np.maximum.reduce(masks) if masks else np.zeros((1536,1920),np.uint8);rgb[u!=0]=(rgb[u!=0]*.55+np.array([0,220,180])*.45).astype('uint8')
    for x,y in xy:cv2.circle(rgb,(int(x),int(y)),3,(255,80,70),-1)
    im=Image.fromarray(rgb).crop(tuple(b));im.thumbnail((900,800));panel.paste(im,(35+col*950,130));d.text((35+col*950,85),label,font=font,fill='white')
   t=targets[-1];d.text((30,955),f"点数{n}；旧有效支持{t['incumbent_best_support']:.3f}；原始候选最高{t['raw_pool_best_qualified']:.3f}；最终{t['final_best_support']:.3f}",font=font,fill='white');d.text((30,1000),'不以投影位置生成候选；0.67纯度、0.5一对一、原面积和数量门槛均不变。',font=font,fill='#ccdce6');panel.save(dest/(case+f'_obj{oid}.jpg'),quality=94)
  cases.append({'case':case,'incumbent_matched':len(old['pairs']),'trial_matched':len(trial['pairs']),'diagnostic_unfiltered_pool_matches':len(pool['pairs']),'targets':targets})
 rejection=collections.Counter(r['reason'] for r in read(out/'instance_generation_consistency/selection.json')['rejected_whole_masks']);neutral=[{'key':r['row']['key'],'source':r['row']['source'],'neutral':r['neutral'],'neutral_passed':r['neutral_passed']} for r in read(out/'instance_generation_consistency/verification_progress.json')['rows']]
 result={'status':'complete','cases':cases,'raw_hypotheses':len(allrows),'whole_rejection_counts':dict(rejection),'neutral_review':neutral,'reference_used_only_after_all_predictions_frozen':True,'original_scorer_sha256':hashlib.sha256((Path(cfg['source_snapshot'])/'tools/evaluate_projected_support.py').read_bytes()).hexdigest(),'no_metric_or_visibility_change':True};save(dest/'completed.json',result);print(json.dumps({'targets':[{k:t[k] for k in ['object_id','visible_reference_points','incumbent_best_support','raw_pool_best_raw','raw_pool_best_qualified','final_best_support']} for c in cases for t in c['targets']],'rejection_counts':dict(rejection),'neutral':neutral},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
