"""Image-only group context, visible seams and exact mask representation."""
import hashlib,copy
import cv2,numpy as np
def exact_binary_shape(mask,label='pallet',shape_id=1):
 from source.agent.edit_agent import _shape_from_mask,_rasterize,valid_shape
 h,w=mask.shape;shape=_shape_from_mask(mask,label,shape_id)
 if shape and valid_shape(shape,w,h) and np.array_equal(mask,_rasterize(shape,w,h)):return shape
 # Lossless fallback merges identical horizontal runs vertically. It never
 # removes pixels to fit a part/vertex limit; an over-complex mask is rejected.
 active={};rectangles=[]
 for y in range(h+1):
  row=mask[y] if y<h else np.zeros(w,np.uint8);changes=np.diff(np.pad((row!=0).astype('int8'),(1,1)));runs=list(zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)));present=set(runs)
  for key in list(active):
   if key not in present:rectangles.append((*key,active.pop(key),y))
  for key in runs:active.setdefault(key,y)
 if not rectangles or len(rectangles)>200:return None
 parts=[{'points':[[float(x1+.05),float(y1+.05)],[float(x2-.05),float(y1+.05)],[float(x2-.05),float(y2-.05)],[float(x1+.05),float(y2-.05)]],'holes':[]} for x1,x2,y1,y2 in rectangles]
 shape={'id':shape_id,'label':label,'parts':parts,'points':parts[0]['points'],'holes':[]}
 return shape if valid_shape(shape,w,h) and np.array_equal(mask,_rasterize(shape,w,h)) else None
def predicted_groups(shapes,image):
 from source.agent.edit_agent import _rasterize
 h,w=np.asarray(image).shape[:2];candidates={};rgb=np.asarray(image).astype('int16')
 for s in shapes:
  if s['label']!='pallet':continue
  mask=_rasterize(s,w,h);n,cc,stats,_=cv2.connectedComponentsWithStats(mask,8)
  for i in range(1,n):
   x,y,bw,bh,area=map(int,stats[i]);m=(cc==i).astype('uint8')
   if area<300 or bw<24 or bh<18:continue
   key=hashlib.sha256(m.tobytes()).hexdigest();candidates[key]={'key':key,'mask':m,'box':[x,y,x+bw,y+bh],'source_shape_ids':[s['id']]}
 groups=[]
 for row in sorted(candidates.values(),key=lambda r:r['key']):
  merge=[]
  for i,g in enumerate(groups):
   a,b=row['box'],g['box'];inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));iou=inter/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
   if iou>=.25:merge.append(i)
  for i in reversed(merge):
   g=groups.pop(i);row={'key':row['key'],'mask':row['mask']|g['mask'],'box':[min(row['box'][0],g['box'][0]),min(row['box'][1],g['box'][1]),max(row['box'][2],g['box'][2]),max(row['box'][3],g['box'][3])],'source_shape_ids':sorted(set(row['source_shape_ids']+g['source_shape_ids']))}
  groups.append(row)
 for g in groups:
  p=rgb[g['mask']!=0];g['blue_fraction']=float(((p[:,2]-p[:,0]>=12)&(p[:,2]>=p[:,1]-3)&(p[:,2]>=45)).mean());g['area']=int(g['mask'].sum());g['key']=hashlib.sha256(g['mask'].tobytes()).hexdigest()
 blue=sorted((g for g in groups if g['blue_fraction']>=.1),key=lambda r:(-r['area']*r['blue_fraction'],r['key']))[:2];taken={g['key'] for g in blue};rest=sorted((g for g in groups if g['key'] not in taken),key=lambda r:(-r['area'],r['key']))[:4-len(blue)]
 return blue+rest
def visible_seam_prompts(image,parent,max_intervals=2):
 from scipy.signal import find_peaks
 x1,y1,x2,y2=parent['box'];m=parent['mask'];gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY).astype('float32');dy=np.abs(cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3));region=m[y1:y2,x1:x2];weighted=dy[y1:y2,x1:x2]*(region!=0);den=np.maximum(1,region.sum(1));profile=weighted.sum(1)/den
 if len(profile)<12:return [],{'peaks':[],'reason':'insufficient visible height'}
 profile=cv2.GaussianBlur(profile[:,None],(1,3),0).ravel();peaks,_=find_peaks(profile,distance=max(4,len(profile)//12),prominence=max(1,float(profile.std())*.5));peaks=sorted(int(p+y1) for p in peaks)
 intervals=[]
 for lo,hi in zip(peaks,peaks[1:]):
  if hi-lo<6:continue
  band=m.copy();band[:lo]=0;band[hi+1:]=0;area=int(band.sum())
  if area<60 or area>.8*parent['area']:continue
  intervals.append((area,lo,hi,band))
 prompts=[]
 def pick(region):
  if not region.any():return None
  d=cv2.distanceTransform(region.astype('uint8'),cv2.DIST_L2,5);y,x=np.unravel_index(np.argmax(d),d.shape);return {'x':int(x),'y':int(y),'label':1}
 for area,lo,hi,band in sorted(intervals,key=lambda r:(-r[0],r[1]))[:max_intervals]:
  points=[]
  for j in range(3):
   r=band.copy();left=x1+(x2-x1)*j//3;right=x1+(x2-x1)*(j+1)//3;r[:,:left]=0;r[:,right:]=0;p=pick(r)
   if p and p not in points:points.append(p)
  for upper in [True,False]:
   r=m.copy()
   if upper:r[lo:]=0;r[:max(y1,lo-max(8,hi-lo))]=0
   else:r[:hi+1]=0;r[min(y2,hi+1+max(8,hi-lo)):]=0
   for j in range(2):
    rr=r.copy();left=x1+(x2-x1)*j//2;right=x1+(x2-x1)*(j+1)//2;rr[:,:left]=0;rr[:,right:]=0;p=pick(rr)
    if p and all((p['x']-q['x'])**2+(p['y']-q['y'])**2>=64 for q in points):points.append(dict(p,label=0))
  if len([p for p in points if p['label']==1])>=2 and any(p['label']==0 for p in points):prompts.append({'points':points,'seam_interval':[lo,hi],'visible_parent_pixels_in_interval':area,'no_mask_prior':True})
 return prompts,{'peaks':peaks,'row_gradient_profile':profile.tolist(),'image_evidence_only':True}
