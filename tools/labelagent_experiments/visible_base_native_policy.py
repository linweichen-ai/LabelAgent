"""Separate input repair: real square source photo and explicit 0..1000 points."""
import copy,math
import numpy as np,cv2
from PIL import Image,ImageDraw
from visible_base_policy import planning_views as padded_views,bind_points as bind_pixel_points,PLAN_PROMPT as ORIGINAL_PROMPT,PLAN_SCHEMA as ORIGINAL_SCHEMA,png
PLAN_PROMPT=ORIGINAL_PROMPT.replace('FIRST 512x512 image','FIRST image using normalized integer coordinates 0..1000 (0 is its top/left edge and 1000 its bottom/right edge)')+' The first image is entirely a real square photo without gray padding. Never use coordinates from a detail photo. Put the first positive on the marked seed if you can bind a single physical object; abstain otherwise.'
PLAN_SCHEMA=copy.deepcopy(ORIGINAL_SCHEMA)
for key in ['positive_points','negative_points']:
 for axis in ['x','y']:PLAN_SCHEMA['properties'][key]['items']['properties'][axis]['maximum']=1000
def planning_views(source,mask):
 _,details,_=padded_views(source,mask);m=np.asarray(mask,dtype='uint8');ys,xs=np.where(m);w,h=source.size;bw,bh=int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1);side=min(w,h,max(bw,bh)+2*max(48,round(.35*max(bw,bh))))
 if side<max(bw,bh):raise ValueError('Whole seed cannot fit a real square crop; reject instead of clipping')
 left=max(0,min(w-side,math.floor((int(xs.min())+int(xs.max())+1-side)/2)));top=max(0,min(h-side,math.floor((int(ys.min())+int(ys.max())+1-side)/2)));box=[left,top,left+side,top+side];assert int(m[top:top+side,left:left+side].sum())==int(m.sum());base=source.crop(tuple(box)).resize((512,512),Image.Resampling.LANCZOS);scale=512/side;distance=cv2.distanceTransform(np.pad(m,1),cv2.DIST_L2,5)[1:-1,1:-1];ay,ax=map(int,np.unravel_index(distance.argmax(),distance.shape));display=lambda x,y:(round((x-left)*scale),round((y-top)*scale));tip=display(ax,ay);a=np.array(base).copy();tail=(max(3,min(508,tip[0]+(-90 if tip[0]>256 else 90))),max(3,min(508,tip[1]-90)));cv2.arrowedLine(a,tail,tip,(255,150,0),3,tipLength=.12);corner=base.copy();d=ImageDraw.Draw(corner);l,t=display(int(xs.min()),int(ys.min()));r,b=display(int(xs.max())+1,int(ys.max())+1)
 for x,y,dx,dy in [(l,t,1,1),(r,t,-1,1),(l,b,1,-1),(r,b,-1,-1)]:d.line([(x+dx*10,y),(x,y),(x,y+dy*10)],fill=(255,0,255),width=3)
 return {'pointer':Image.fromarray(a),'corners':corner},details,{'source_box':box,'display_size':[512,512],'offset':[0,0],'scale_xy':[scale,scale],'anchor_source':[ax,ay],'retained_seed_pixels':int(m.sum()),'seed_pixels':int(m.sum()),'complete_seed_in_overview':True,'no_padding_in_overview':True,'coordinate_system':'normalized_0_1000','expected_class_rendered':False}
def bind_points(decision,binding,seed):
 scaled=copy.deepcopy(decision)
 for key in ['positive_points','negative_points']:
  for p in scaled.get(key,[]):
   if not 0<=p['x']<=1000 or not 0<=p['y']<=1000:return None,'point outside declared normalized coordinates'
   p['x']=p['x']*511/1000;p['y']=p['y']*511/1000
 return bind_pixel_points(scaled,binding,seed)
