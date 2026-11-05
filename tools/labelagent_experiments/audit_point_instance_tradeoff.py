"""Exact point ownership, purity and matching decomposition of old vs rejected."""
import sys,json,hashlib,io
from pathlib import Path
from projection_audit_core import ROOT,read,environment,frame_data,cameras
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 cfg,ev=environment();import numpy as np,cv2
 from PIL import Image,ImageDraw,ImageFont,ImageOps
 from projection_audit_core import matrix_audit
 out=ROOT/'outputs/labelagent_projection_recovery_20261105/point_instance_audit';out.mkdir(parents=True,exist_ok=True)
 if (out/'completed.json').exists():raise FileExistsError('Preserve exact diagnostic')
 dataset=Path(cfg['dataset']);manifest=read(dataset/'import_manifest.json');data=frame_data(dataset,manifest,76,ev);cams=cameras(dataset,manifest,ev);old=ROOT/'outputs/labelagent_context_comparison_20261105';new=ROOT/'outputs/labelagent_semantic_comparison_20261105';font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',26);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);cases=[];allpoints=[]
 for cam in [0,1]:
  case=f'camera{cam}_frame76';source,pixels=ev.project_visible(data['xyz'],cams[cam]);a=matrix_audit(read(old/'C_hybrid'/case/'result.json')['shapes'],data,source,pixels,ev);b=matrix_audit(read(new/'C_hybrid'/case/'result.json')['shapes'],data,source,pixels,ev);eligible=a['eligible'];semantic=data['bits'][source];owner=data['owner'][source];pal=semantic==1;gained=pal&b['union_hits']&~a['union_hits'];lost=pal&a['union_hits']&~b['union_hits'];rows=[]
  oldmatched={r for r,j,v in a['pairs']};newmatched={r for r,j,v in b['pairs']}
  for i in sorted(set(owner[pal])):
   selected=pal&(owner==i);ref=data['boxes'][i-1]['obj_id'] if i>0 else 'same_class_ambiguous_owner';where=eligible.index(i-1) if i>0 and i-1 in eligible else None
   r={'object_id':ref,'visible_points':int(selected.sum()),'old_covered':int((selected&a['union_hits']).sum()),'new_covered':int((selected&b['union_hits']).sum()),'gained':int((selected&gained).sum()),'lost':int((selected&lost).sum()),'eligible_for_instance_support':where is not None}
   if where is not None:
    av=float(a['support'][where].max());bv=float(b['support'][where].max());r.update(old_best_supported=av,new_best_supported=bv,old_matched=where in oldmatched,new_matched=where in newmatched,old_best_raw=float(a['raw'][where].max()),new_best_raw=float(b['raw'][where].max()))
    if r['old_matched'] and not r['new_matched']:r['loss_mechanism']='lost purity-qualified support before matching' if bv<.5 else 'one-to-one matching competition'
   rows.append(r)
  candidates=[]
  for name,run in [('retained13',a),('rejected12',b)]:
   matched={r:j for r,j,v in run['pairs']};payload={'object_ids':[data['boxes'][i]['obj_id'] for i in eligible],'shape_ids':[s['id'] for s in run['shapes']],'support':run['support'].tolist(),'raw_coverage':run['raw'].tolist(),'purity':run['purity'].tolist(),'matching':[{'object_id':data['boxes'][eligible[r]]['obj_id'],'shape_id':run['shapes'][j]['id'],'support':v} for r,j,v in run['pairs']]};save(out/(case+'_'+name+'_matrix.json'),payload)
   for k,i in enumerate(eligible):
    if data['boxes'][i]['obj_id'] not in ([42,48] if cam==0 else [47,48,49]):continue
    for j,s in enumerate(run['shapes']):
     if run['raw'][k,j]<=0:continue
     candidates.append({'variant':name,'object_id':data['boxes'][i]['obj_id'],'shape_id':s['id'],'mask_sha256':hashlib.sha256(run['masks'][j].tobytes()).hexdigest(),'point_coverage':float(run['raw'][k,j]),'purity':float(run['purity'][k,j]),'qualified_support':float(run['support'][k,j]),'matched_to_this_object':matched.get(k)==j})
   cw,ch=58,48;canvas=Image.new('RGB',(max(1200,180+len(run['shapes'])*cw),180+len(eligible)*ch),'#142436');d=ImageDraw.Draw(canvas);d.text((20,20),case+' / '+name+' / 原0.67纯度过滤后支持矩阵',font=font,fill='white')
   for j,s in enumerate(run['shapes']):d.text((180+j*cw,85),str(s['id']),font=small,fill='white')
   for k,i in enumerate(eligible):
    d.text((20,135+k*ch),str(data['boxes'][i]['obj_id']),font=small,fill='white')
    for j in range(len(run['shapes'])):
     v=run['support'][k,j];color=(25,int(45+155*v),int(70+100*v));x,y=175+j*cw,130+k*ch;d.rectangle((x,y,x+cw-3,y+ch-3),fill=color,outline='#ffffff' if matched.get(k)==j else '#364b5e');d.text((x+2,y+8),f'{v:.2f}',font=small,fill='white')
   canvas.save(out/(case+'_'+name+'_匹配矩阵.jpg'),quality=93)
  for k in np.flatnonzero(pal):
   idx=source[k];oid=int(owner[k]);allpoints.append({'camera':cam,'filtered_source_index':int(idx),'raw_pcd_row':int(data['raw_pcd_rows'][idx]),'object_id':data['boxes'][oid-1]['obj_id'] if oid>0 else 'ambiguous','pixel':pixels[k].tolist(),'old_covered':bool(a['union_hits'][k]),'new_covered':bool(b['union_hits'][k]),'change':'gained' if gained[k] else 'lost' if lost[k] else 'held' if a['union_hits'][k] else 'uncovered'})
  image=Image.open(old/'inputs'/(case+'.png')).convert('RGB');render=np.array(image).copy()
  for flag,color in [(pal&a['union_hits']&b['union_hits'],(40,200,210)),(gained,(50,250,90)),(lost,(255,60,60))]:
   for x,y in pixels[flag]:cv2.circle(render,(int(x),int(y)),4,color,-1)
  canvas=Image.new('RGB',(1920,1080),'#142436');d=ImageDraw.Draw(canvas);d.text((30,20),case+' / 相同原始点索引的覆盖增减',font=font,fill='white')
  for i,im in enumerate([image,Image.fromarray(render)]):im=ImageOps.contain(im,(930,800));canvas.paste(im,(20+i*950,100))
  d.text((30,940),f"绿：新增{int(gained.sum())}；红：丢失{int(lost.sum())}；青：保持覆盖。原半径3、0.67纯度、0.5一对一不变。",font=font,fill='white');d.text((30,995),'参考投影不是二维真值；本图只解释旧结果，不参与新候选生成。',font=small,fill='#c6dce8');canvas.save(out/(case+'_参考点增减.jpg'),quality=94)
  row={'case':case,'old_points':int((pal&a['union_hits']).sum()),'new_points':int((pal&b['union_hits']).sum()),'gained':int(gained.sum()),'lost':int(lost.sum()),'old_instances':len(a['pairs']),'new_instances':len(b['pairs']),'objects':rows,'target_candidate_lineage':candidates};cases.append(row)
 assert sum(r['old_points'] for r in cases)==490 and sum(r['new_points'] for r in cases)==493
 assert sum(r['old_instances'] for r in cases)==13 and sum(r['new_instances'] for r in cases)==12
 save(out/'point_lineage.json',allpoints);save(out/'completed.json',{'status':'complete','scope':'post-freeze old-result diagnostic, not generation','cases':cases,'matching_algorithm_unchanged':True,'scorer_unchanged':True,'explanation':'union point coverage and purity-qualified one-to-one support measure different properties'})
 print(json.dumps({'gained':sum(r['gained'] for r in cases),'lost':sum(r['lost'] for r in cases),'per_case':[{k:r[k] for k in ['case','gained','lost','old_instances','new_instances']} for r in cases]},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
