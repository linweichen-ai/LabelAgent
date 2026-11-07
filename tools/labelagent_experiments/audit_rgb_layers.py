"""Score frozen complete native bitmaps with original point and area gates."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,cv2
from PIL import Image
from run_rgb_layers import ROOT,OUT,INC,read,save,sha
from projection_audit_core import frame_data,cameras

def main():
 generated=read(OUT/'generation_completed.json');assert generated['status']=='complete'
 cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');sys.path[:0]=[cfg['source_snapshot'],str(Path(cfg['source_snapshot'])/'tools'),cfg['dependency_dir']]
 import evaluate_projected_support as ev
 from source.agent.edit_agent import _rasterize
 assert sha(Path(cfg['source_snapshot'])/'tools/evaluate_projected_support.py')=='93c5152169d04d43b25dbf4a4139b12676daf78750a7f059837121656bb644c2'
 dataset=Path(cfg['dataset']);manifest=read(dataset/'import_manifest.json');data=frame_data(dataset,manifest,76,ev);cams=cameras(dataset,manifest,ev)
 originals={};old={};people={};protected={}
 for cam in (0,1):
  case=f'camera{cam}_frame76';path=INC/'C_hybrid'/case/'result.json';protected[str(path)]=sha(path);sh=read(path)['shapes']
  old[case]=[_rasterize(s,1920,1536) for s in sh if s['label']=='pallet'];people[case]=[_rasterize(s,1920,1536) for s in sh if s['label']=='person'];originals[case]=[_rasterize(s,1920,1536) for s in read(INC/'A_current'/case/'result.json')['shapes'] if s['label']=='pallet']
 basearea=sum(int(np.maximum.reduce(ms).sum()) for ms in old.values());originalarea=sum(int(np.maximum.reduce(ms).sum()) for ms in originals.values());assert basearea==122302 and originalarea==98009
 candidates=[];seen=set()
 for row in generated['rows']:
  if row['key'] in seen or 'mask_sha256' not in row:continue
  seen.add(row['key']);folder=Path(row['folder']);m=np.load(folder/'complete_full_mask.npy',allow_pickle=False);assert hashlib.sha256(m.tobytes()).hexdigest()==row['mask_sha256']
  raw=np.load(folder/'all_arrays.npz',allow_pickle=False);meta=read(folder/'metadata.json');local=(raw[f"{meta['selected_prediction_call']}_masks"][meta['selected_index']].squeeze()!=0).astype('uint8');x1,y1,x2,y2=row['box'];assert np.array_equal(m[y1:y2,x1:x2],local) and int(m.sum())==int(local.sum())
  extra=int(((m!=0)&(np.maximum.reduce(old[row['case']])==0)).sum());person_guard=cv2.dilate(np.maximum.reduce(people[row['case']]),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
  candidates.append({**row,'extra_pixels':extra,'area_fits':basearea+extra<=1.25*originalarea,'person_conflict':bool((m&person_guard).any()),'complete_internal_bitmap_preserved':True})
 cases=[]
 for cam in (0,1):
  case=f'camera{cam}_frame76';source,pix=ev.project_visible(data['xyz'],cams[cam]);owned=data['owner'][source];known=(data['bits'][source]==1)&(owned>0);eligible=[i for i,b in enumerate(data['boxes']) if b['bit']==1 and (owned==i+1).sum()>=5]
  new=[r for r in candidates if r['case']==case];masks=old[case]+[np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False) for r in new];radii=[]
  for radius in (0,3,6):
   kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1));hits=np.array([(cv2.dilate(m,kernel) if radius else m)[pix[:,1],pix[:,0]]!=0 for m in masks]);support=np.zeros((len(eligible),len(masks)));details=[]
   for j,h in enumerate(hits):
    total=int((h&known).sum());ownership=[]
    for k,i in enumerate(eligible):
     n=int((h&(owned==i+1)).sum());fraction=n/int((owned==i+1).sum());purity=n/total if total else 0;support[k,j]=fraction if purity>=.67 else 0
     if n:ownership.append({'object_id':data['boxes'][i]['obj_id'],'points':n,'coverage':fraction,'purity':purity,'qualified_support':float(support[k,j])})
    if j>=len(old[case]):details.append({'key':new[j-len(old[case])]['key'],'ownership':ownership})
   oldpairs=ev.match(support[:,:len(old[case])],.5);pairs=ev.match(support,.5);targets=[]
   for oid in ([42,48] if cam==0 else [47,49]):
    k=next(k for k,i in enumerate(eligible) if data['boxes'][i]['obj_id']==oid);targets.append({'object_id':oid,'old_best_support':float(support[k,:len(old[case])].max()),'new_best_support':float(support[k,len(old[case]):].max()) if new else 0})
   radii.append({'radius':radius,'old_matches':len(oldpairs),'unfiltered_all_matches':len(pairs),'new_mask_ownership':details,'remaining_targets':targets})
  cases.append({'case':case,'radii':radii})
 assert all(sha(p)==h for p,h in protected.items())
 result={'status':'complete_post_freeze_raw_bitmap_audit','base_union':basearea,'cap':1.25*originalarea,'candidates':candidates,'cases':cases,'old_masks_preserved':27,'old_person_masks_preserved':3,'all_old_result_bytes_unchanged':True,'not_adoption_or_dense_truth':True,'new_candidate_adopted':False}
 save(OUT/'support_area_audit.json',result);print(json.dumps({'candidates':[{'key':r['key'],'area':r['raw_pixels'],'extra':r['extra_pixels'],'area_fits':r['area_fits'],'person':r['person_conflict']} for r in candidates],'cases':cases},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
