"""Input geometry never changes the predicted mask or acceptance thresholds."""
import math,io,copy
import cv2,numpy as np
from PIL import Image,ImageDraw
def png(im):
 b=io.BytesIO();im.save(b,'PNG');return b.getvalue()
def centered_photos(source,mask):
 m=np.asarray(mask,dtype='uint8');ys,xs=np.where(m);assert len(xs)>0 and m.shape==(source.height,source.width)
 distance=cv2.distanceTransform(np.pad(m,1),cv2.DIST_L2,5)[1:-1,1:-1];ay,ax=map(int,np.unravel_index(distance.argmax(),distance.shape));radius=max(abs(xs-ax).max(),abs(ys-ay).max())+9;views={};binding=[]
 for name,factor in [('near',1),('context',1.5)]:
  half=int(math.ceil(radius*factor));box=[ax-half,ay-half,ax+half+1,ay+half+1];clipped=[max(0,box[0]),max(0,box[1]),min(source.width,box[2]),min(source.height,box[3])];canvas=Image.new('RGB',(2*half+1,2*half+1),(127,127,127));canvas.paste(source.crop(tuple(clipped)),(clipped[0]-box[0],clipped[1]-box[1]));assert int(m[clipped[1]:clipped[3],clipped[0]:clipped[2]].sum())==int(m.sum());canvas=canvas.resize((512,512),Image.Resampling.LANCZOS);views[name]=canvas;binding.append({'view':name,'source_box':clipped,'canvas_box':box,'foreground_anchor':[ax,ay],'center_is_original_foreground':True,'all_target_pixels_retained':True,'target_pixels':int(m.sum()),'expected_class_in_view':False})
 return views,binding
def deck_foot_prompts(image,parent):
 from instance_separation_geometry import visible_seam_prompts
 m=parent['mask'];old,evidence=visible_seam_prompts(image,parent);x1,y1,x2,y2=parent['box'];rows=[]
 def deepest(region):
  if not region.any():return None
  d=cv2.distanceTransform(region.astype('uint8'),cv2.DIST_L2,5);y,x=np.unravel_index(d.argmax(),d.shape);return {'x':int(x),'y':int(y),'label':1}
 for item in old[:2]:
  lo,hi=item['seam_interval'];span=hi-lo;top=max(y1,lo-max(3,round(.75*span)));bottom=min(y2,hi+max(3,round(.5*span))+1);region=m.copy();region[:top]=0;region[bottom:]=0;points=[]
  # Two visible upper deck anchors, plus one lower central connection/foot.
  for left,right,upper in [(x1,(x1+x2)//2,True),((x1+x2)//2,x2,True),(x1,x2,False)]:
   rr=region.copy();rr[:,:left]=0;rr[:,right:]=0;mid=(top+bottom)//2
   if upper:rr[mid:]=0
   else:rr[:mid]=0
   q=deepest(rr)
   if q and q not in points:points.append(q)
  if len(points)<3:continue
  for upper in [True,False]:
   rr=m.copy()
   if upper:rr[top:]=0;rr[:max(y1,top-max(8,span))]=0
   else:rr[:bottom]=0;rr[min(y2,bottom+max(8,span)):]=0
   for half in [0,1]:
    r=rr.copy();a=x1+(x2-x1)*half//2;b=x1+(x2-x1)*(half+1)//2;r[:,:a]=0;r[:,b:]=0;q=deepest(r)
    if q and all((q['x']-p['x'])**2+(q['y']-p['y'])**2>=64 for p in points):points.append(dict(q,label=0))
  if any(q['label']==0 for q in points):rows.append({'points':points,'seam_interval':[top,bottom-1],'old_seam_interval':[lo,hi],'deck_foot_envelope':[top,bottom],'mask_prior':None,'no_reference_inputs':True})
 return rows,evidence
EXTENT_SCHEMA={'type':'object','properties':{'object':{'type':'string','minLength':1,'maxLength':180},'target_extent':{'type':'string','enum':['whole_visible_object','identifiable_occluded_object','edge_fragment','multiple_objects','uncertain']},'evidence':{'type':'string','minLength':1,'maxLength':180}},'required':['object','target_extent','evidence'],'additionalProperties':False}
EXTENT_PROMPT=('Describe the actual physical object at the indicated source pixels. The second image is the exact candidate mask, not an object reference. Judge whether those pixels identify one whole visible object, an identifiable occluded object, only an edge fragment, multiple objects, or uncertainty. Do not assign the identity of a larger nearby object merely because it is near the mask. State concrete visible material and structural evidence. Do not invent hidden parts or infer a category from this task. Return object, target_extent, evidence.')
def extent_views(source,mask):
 photos,bind=centered_photos(source,mask);b=bind[0]['canvas_box'];ax,ay=bind[0]['foreground_anchor'];im=photos['near'].copy();d=ImageDraw.Draw(im);d.line([(195,190),(256,256)],fill=(255,150,0),width=3);local=np.zeros((b[3]-b[1],b[2]-b[0]),np.uint8);x1,y1,x2,y2=bind[0]['source_box'];local[y1-b[1]:y2-b[1],x1-b[0]:x2-b[0]]=mask[y1:y2,x1:x2];binary=Image.fromarray(local*255).resize((512,512),Image.Resampling.NEAREST);return im,binary,bind
