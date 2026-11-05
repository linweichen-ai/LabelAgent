"""Preserve the entire incumbent; append complete hypotheses or reject them."""
import cv2,numpy as np
def append_complete_masks(pool,incumbent,original,people):
 unions={c:np.maximum.reduce(ms).copy() for c,ms in incumbent.items()};original_area=sum(int(np.maximum.reduce(ms).sum()) for ms in original.values());original_count=sum(len(ms) for ms in original.values());count=sum(len(ms) for ms in incumbent.values());selected={c:[] for c in incumbent};rejected=[];added={c:[] for c in incumbent}
 guards={c:cv2.dilate(np.maximum.reduce(ms),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))) if ms else np.zeros_like(unions[c]) for c,ms in people.items()}
 def iou(a,b):
  n=int((a&b).sum());return n/max(1,int(a.sum())+int(b.sum())-n)
 ranked=[]
 for case,rows in pool.items():
  for row in rows:
   rank=float(row.get('score',row.get('trained_score',0)))*(1-max([iou(row['mask'],b) for b in incumbent[case]]+[0]));ranked.append((rank,case,row))
 for rank,case,row in sorted(ranked,key=lambda r:(-r[0],r[1],r[2]['key'])):
  m=row['mask'];extra=int(((m!=0)&(unions[case]==0)).sum());why=None
  if any(np.array_equal(m,b) for b in incumbent[case]):why='identical incumbent mask'
  elif np.any(m&guards[case]):why='conflicts with incumbent predicted-person radius3 guard'
  elif count>=int(1.5*original_count):why='original global count limit'
  elif sum(int(u.sum()) for u in unions.values())+extra>1.25*original_area:why='complete mask exceeds remaining original global area budget'
  elif any(iou(m,b)>=.5 or int((m&b).sum())/max(1,min(int(m.sum()),int(b.sum())))>=.8 for b in added[case]):why='duplicate new hypothesis'
  if why:rejected.append({'case':case,'key':row['key'],'reason':why,'new_pixels':extra,'rank':rank});continue
  selected[case].append({k:v for k,v in row.items() if k!='mask'});added[case].append(m);unions[case]|=m;count+=1
 return selected,rejected,{'PAL_count':count,'PAL_union_pixels':sum(int(u.sum()) for u in unions.values()),'original_A_count':original_count,'original_A_area':original_area,'all_incumbent_masks_preserved':True,'budget_pixel_clipping':False}
