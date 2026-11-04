"""Retrospective diagnosis only; reference data must never enter proposal selection."""
import sys,json,io,hashlib,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'task_workspaces/labelagent_efficiency_20261104'
OUT=ROOT/'outputs/labelagent_efficiency_20261104'
OLD=ROOT/'outputs/labelagent_hybrid_verified_20261104'
sys.path[:0]=[str(WORK/'source_snapshot'),str(WORK/'source_snapshot/tools'),str(ROOT/'task_workspaces/cvat_mask2former_20261103/deps')]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 import cv2,numpy as np,yaml
 from PIL import Image,ImageDraw,ImageFont
 import evaluate_projected_support as ev
 from source.agent.edit_agent import _rasterize,apply_operations
 from source.agent.workflow import candidate_holes
 out=OUT/'diagnosis';out.mkdir(exist_ok=True)
 if (out/'completed.json').exists():raise FileExistsError('Retain completed diagnosis')
 cfg=read(ROOT/'task_workspaces/labelagent_hybrid_repair_20261104/config_verified.json');dataset=Path(cfg['dataset']);manifest=read(dataset/'import_manifest.json')
 stamp=str(manifest['timestamps_ns'][76]);annotation=json.loads(ev.checked_source(dataset,manifest,f'labels/{stamp}.json'))
 path=f'lidar/{stamp}.pcd';ev.checked_source(dataset,manifest,path);payload,fields,stride,count,_=ev.read_pcd(dataset/'source'/path);fields={f['name']:f for f in fields};types={7:'<f4',8:'<f8'}
 xyz=np.column_stack([np.ndarray((count,),dtype=types[fields[n]['type']],buffer=payload,offset=fields[n]['offset'],strides=(stride,)) for n in ['x','y','z']]);xyz=xyz[np.isfinite(xyz).all(axis=1)&(np.linalg.norm(xyz,axis=1)>.001)]
 bits,owner,boxes=ev.class_membership(xyz,annotation['objects']);extrinsics=yaml.safe_load(ev.checked_source(dataset,manifest,'calib/lidar2camera.yaml'))
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
 rows=[];rounds=[];images=[];component_rows=[]
 for cam in [0,1]:
  case=f'camera{cam}_frame76';image=Image.open(OLD/'inputs'/(case+'.png')).convert('RGB');w,h=image.size
  text=ev.checked_source(dataset,manifest,f'calib/camera{cam}_calib.yaml').decode('utf-8');intrinsic=yaml.safe_load('\n'.join(s for s in text.splitlines() if not s.startswith('%YAML:')))
  camera=ev.camera_parameters(intrinsic,extrinsics[f'lidar_to_camera_{cam}']);source,pixels=ev.project_visible(xyz,camera)
  semantics=bits[source];owners=owner[source];known=(semantics==1)&(owners>0)
  eligible={i:np.flatnonzero(owners==i+1) for i,b in enumerate(boxes) if b['bit']==1 and np.count_nonzero(owners==i+1)>=5}
  a=[s for s in read(OLD/'A_current'/case/'result.json')['shapes'] if s['label']=='pallet'];c=[s for s in read(OLD/'C_hybrid'/case/'result.json')['shapes'] if s['label']=='pallet']
  amasks=[_rasterize(s,w,h) for s in a];cmasks=[_rasterize(s,w,h) for s in c];baseline=np.maximum.reduce(amasks);hashes={hashlib.sha256(m.tobytes()).hexdigest() for m in amasks}
  existing_supported={r['obj_id'] for r in read(OLD/'A_current/evaluation/per_image.json')[cam]['objects'] if r['label']=='pallet' and r['best_candidate_supported_coverage']>=.5}
  for origin,shapes,masks in [('baseline',a,amasks),('hybrid',c,cmasks)]:
   for shape,mask in zip(shapes,masks):
    n,cc,stats,_=cv2.connectedComponentsWithStats(mask,8);components=[int(stats[k,cv2.CC_STAT_AREA]) for k in range(1,n) if stats[k,cv2.CC_STAT_AREA]>=9]
    component_rows.append({'case':case,'source':origin,'shape_id':shape['id'],'components':components})
  for shape,mask in zip(c,cmasks):
   if hashlib.sha256(mask.tobytes()).hexdigest() in hashes:continue
   dilated=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)));hits=dilated[pixels[:,1],pixels[:,0]]!=0;known_hits=int((hits&known).sum())
   object_rows=[]
   for i,indices in eligible.items():
    hit=int(hits[indices].sum());coverage=hit/len(indices);purity=hit/known_hits if known_hits else 0
    object_rows.append({'obj_id':boxes[i]['obj_id'],'hits':hit,'points':len(indices),'coverage':coverage,'purity':purity,'qualifies':coverage>=.5 and purity>=.67,'already_supported_by_baseline':boxes[i]['obj_id'] in existing_supported})
   intersection=[int((mask&m).sum()) for m in amasks];area=int(mask.sum());marginal=int(((mask!=0)&(baseline==0)).sum())
   row={'case':case,'shape_id':shape['id'],'area':area,'marginal_pixels_outside_baseline_union':marginal,
        'maximum_IoU_with_baseline':max(inter/max(1,area+int(m.sum())-inter) for inter,m in zip(intersection,amasks)),
        'maximum_overlap_coefficient_with_baseline':max(inter/max(1,min(area,int(m.sum()))) for inter,m in zip(intersection,amasks)),
        'object_support':object_rows,'reference_use':'retrospective diagnosis only; never used to select predictions'}
   rows.append(row)
   render=np.array(image).copy();render[mask!=0]=(.5*render[mask!=0]+.5*np.array([245,80,60])).astype('uint8');cv2.drawContours(render,cv2.findContours(baseline,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(30,210,210),2)
   yy,xx=np.where(mask);rect=(max(0,int(xx.min())-32),max(0,int(yy.min())-32),min(w,int(xx.max())+33),min(h,int(yy.max())+33))
   canvas=Image.new('RGB',(1920,930),'#f1f5fa');d=ImageDraw.Draw(canvas);d.text((20,15),case+f" · 新增候选{shape['id']}：同物体重包/范围增加诊断",font=font,fill='#142536')
   canvas.paste(image.crop(rect).resize((960,768)),(0,80));canvas.paste(Image.fromarray(render).crop(rect).resize((960,768)),(960,80))
   d.text((20,866),f"红：新增；青线：原候选。新增面积 {marginal} / 总面积 {area}；图示不是二维真值。",font=font,fill='#142536')
   file=out/(case+f'_addition_{shape["id"]}.jpg');canvas.save(file,quality=94);images.append(file.name)
  for arm in ['B_sam7','C_hybrid']:
   result=read(OLD/arm/case/'result.json');current=copy.deepcopy(read(OLD/'seed_stages'/case/'all_arms.json')[arm])
   for audit in result['agent']['audit']:
    holes=candidate_holes(current,['floor'],w,h);lookup={v['id']:v for v in holes};details=[]
    ground=sum(v['semantic']=='ground' for v in audit['decision']['hole_assessment'])
    for operation in audit['decision']['operations']:
     if operation['operation']!='fill_gap':raise ValueError('Unexpected old operation')
     hole=lookup[operation['hole_id']];region=np.zeros((h,w),np.uint8);cv2.fillPoly(region,[np.array(operation['polygon'],np.int32)],1)
     requested=int(hole['mask'].sum());covered=int((region&hole['mask']).sum())
     current,applied,queue=apply_operations(current,[operation],['pallet','person','floor','slope','ego','other_robot','cargo_floor'],w,h,gap_regions=holes)
     new_mask=_rasterize(next(s for s in current if s['id']==hole['shape_id']),w,h)
     remaining=int(((hole['mask']!=0)&(new_mask==0)).sum())
     details.append({'hole_id':hole['id'],'kind':hole['kind'],'bbox_inclusive':hole['bbox'],'target_pixels':requested,'pixels_inside_old_polygon':covered,'target_pixels_still_missing_after_edit':remaining,'actual_tool_results':applied,'queue':queue})
    rounds.append({'case':case,'arm':arm,'round':audit['round'],'approved_ground_tasks':ground,'selected_operations':details})
   assert all(np.array_equal(_rasterize(x,w,h),_rasterize(y,w,h)) for x,y in zip(sorted(current,key=lambda s:s['id']),sorted(result['shapes'],key=lambda s:s['id'])))
 save(out/'candidate_diagnosis.json',rows);save(out/'round_replay.json',rounds);save(out/'component_counts.json',component_rows)
 conclusion={'status':'complete','new_candidates':len(rows),'new_qualifying_objects_from_additions':sorted({(r['case'],o['obj_id']) for r in rows for o in r['object_support'] if o['qualifies'] and not o['already_supported_by_baseline']}),
  'operations_with_incomplete_target_polygon':sum(o['pixels_inside_old_polygon']<o['target_pixels'] for r in rounds for o in r['selected_operations']),
  'operations_leaving_reviewed_target_pixels_missing':sum(o['target_pixels_still_missing_after_edit']>0 for r in rounds for o in r['selected_operations']),
  'first_round_ground_tasks':[{k:r[k] for k in ['case','arm','approved_ground_tasks']} for r in rounds if r['round']==1],
  'baseline_multi_component_masks':sum(len(r['components'])>1 for r in component_rows if r['source']=='baseline'),
  'annotation_writes':0,'prediction_changes':0,'images':images}
 save(out/'completed.json',conclusion);print(json.dumps(conclusion,ensure_ascii=False),flush=True)
 html=['<!doctype html><meta charset="utf-8"><title>实例支持/面积/轮数诊断</title><style>body{max-width:1400px;margin:30px auto;font:18px/1.7 sans-serif}img{width:100%}</style><h1>新增候选与三轮回放诊断</h1><p>只读回放，3D参考仅用于解释旧结果，不进入新预测筛选。原始数据和标注不改。</p><pre>'+json.dumps(conclusion,ensure_ascii=False,indent=2)+'</pre>']
 for file in images:html.append(f'<p>{file}</p><img src="{file}">')
 (out/'index.html').write_text('\n'.join(html),encoding='utf-8')
if __name__=='__main__':main()
