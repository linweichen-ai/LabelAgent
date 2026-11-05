"""Compare pre/post competition of the same saved forwards, post-freeze only."""
import sys,json
from pathlib import Path
from projection_audit_core import ROOT,read,frame_data,cameras
cfg=read(ROOT/'task_workspaces/labelagent_instance_separation_20261105/config_comparison.json');sys.path[:0]=[cfg['source_snapshot'],str(Path(cfg['source_snapshot'])/'tools'),cfg['dependency_dir']]
import numpy as np,cv2,evaluate_projected_support as ev
from PIL import Image
from source.agent.edit_agent import _rasterize
out=ROOT/'outputs/labelagent_instance_separation_20261105';assert read(ROOT/'outputs/labelagent_instance_separation_comparison_20261105/completed.json')['status']=='complete'
target=out/'instance_support_audit/same_forward_competition.json'
if target.exists():raise FileExistsError('Preserve frozen diagnosis')
data=frame_data(Path(cfg['dataset']),read(Path(cfg['dataset'])/'import_manifest.json'),76,ev);cams=cameras(Path(cfg['dataset']),read(Path(cfg['dataset'])/'import_manifest.json'),ev);rows=[]
for cam in [0,1]:
 case=f'camera{cam}_frame76';source,pix=ev.project_visible(data['xyz'],cams[cam]);owner=data['owner'][source];known=(data['bits'][source]==1)&(owner>0);pool={k:[] for k in ['standard_same_forward','raw_query','seam_SAM']}
 for f in sorted((out/'query_generation'/case).glob('*/prediction.json')):
  p=read(f);encoded=np.asarray(Image.open(f.parent/'panoptic.png')).astype('int32');ids=encoded[:,:,0]+256*encoded[:,:,1]+65536*encoded[:,:,2];x1,y1,x2,y2=p['source_box']
  for seg in p['segments_info']:
   if seg['category_id']!=1:continue
   m=np.zeros((1536,1920),np.uint8);m[y1:y2,x1:x2]=ids==seg['id'];pool['standard_same_forward'].append(m)
 for key,file in [('raw_query',out/'query_generation/all_hypotheses.json'),('seam_SAM',out/'seam_generation/all_hypotheses.json')]:pool[key]=[_rasterize(r['shape'],1920,1536) for r in read(file) if r['case']==case]
 for oid in ([42,48] if cam==0 else [47,49]):
  obj=next(i+1 for i,b in enumerate(data['boxes']) if b['obj_id']==oid);which=owner==obj;n=int(which.sum());stages={}
  for kind,masks in pool.items():
   rr=[]
   for m in masks:
    hits=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))[pix[:,1],pix[:,0]]!=0;covered=int((hits&which).sum());purity=covered/max(1,int((hits&known).sum()));rr.append((covered/n,purity,covered/n if purity>=.67 else 0))
   stages[kind]={'count':len(masks),'best_raw_coverage':max([r[0] for r in rr]+[0]),'best_qualified_support':max([r[2] for r in rr]+[0])}
  rows.append({'case':case,'object_id':oid,'stages':stages})
target.write_text(json.dumps({'status':'complete','same_forwards':16,'new_model_forwards':0,'reference_only_after_final_freeze':True,'original_radius_purity_support_unchanged':True,'rows':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(rows,ensure_ascii=False))
