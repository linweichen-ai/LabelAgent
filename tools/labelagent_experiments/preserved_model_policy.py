"""Reference-free admission of complete model masks with per-camera reserves."""
import hashlib
import cv2,numpy as np
def select_complete_masks(pool,baselines,people):
 selected={};rejected=[];summary={}
 for case,base in baselines.items():
  union=np.maximum.reduce(base).copy();initial=int(union.sum());limit_count=int(1.5*len(base))-len(base);taken=[];taken_masks=[]
  human=np.maximum.reduce(people[case]) if people[case] else np.zeros_like(union)
  human=cv2.dilate(human,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
  def iou(a,b):
   intersection=int((a&b).sum());return intersection/max(1,int(a.sum())+int(b.sum())-intersection)
  candidates=[]
  for row in pool.get(case,[]):
   mask=row['mask'];novelty=1-max([iou(mask,b) for b in base]+[0]);candidates.append((float(row['trained_score'])*novelty,row))
  for rank,row in sorted(candidates,key=lambda pair:(-pair[0],pair[1]['key'])):
   mask=row['mask'];why=None;novel=int(((mask!=0)&(union==0)).sum())
   if any(np.array_equal(mask,b) for b in base):why='identical original mask'
   elif int((mask&human).sum()):why='whole mask conflicts with original predicted person margin'
   elif len(taken)>=limit_count:why='per-camera original count reserve used'
   elif int(union.sum())+novel>1.25*initial:why='whole mask over per-camera original area reserve'
   elif any(iou(mask,b)>=.5 or int((mask&b).sum())/max(1,min(int(mask.sum()),int(b.sum())))>=.8 for b in taken_masks):why='duplicate newly admitted hypothesis'
   if why:rejected.append({'case':case,'key':row['key'],'reason':why,'rank':rank,'new_pixels':novel});continue
   taken.append({k:v for k,v in row.items() if k!='mask'});taken_masks.append(mask);union|=mask
  selected[case]=taken;summary[case]={'original_count':len(base),'new_count':len(taken),'original_union_pixels':initial,'new_union_pixels':int(union.sum()),'whole_masks_preserved':True}
 return selected,rejected,summary
