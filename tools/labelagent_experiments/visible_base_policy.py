"""Image-only complete-object proposals; a point planner never grants acceptance."""
import io,math,copy
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageOps
PLAN_SCHEMA={'type':'object','properties':{'object':{'type':'string','minLength':1,'maxLength':160},'instance_scope':{'type':'string','enum':['single_visible_object','multiple_objects','edge_only','uncertain','empty']},'positive_points':{'type':'array','maxItems':3,'items':{'type':'object','properties':{'x':{'type':'integer','minimum':0,'maximum':511},'y':{'type':'integer','minimum':0,'maximum':511}},'required':['x','y'],'additionalProperties':False}},'negative_points':{'type':'array','maxItems':4,'items':{'type':'object','properties':{'x':{'type':'integer','minimum':0,'maximum':511},'y':{'type':'integer','minimum':0,'maximum':511}},'required':['x','y'],'additionalProperties':False}},'evidence':{'type':'string','minLength':1,'maxLength':180}},'required':['object','instance_scope','positive_points','negative_points','evidence'],'additionalProperties':False}
PLAN_PROMPT=('Name the actual visible physical object containing the marked source pixels. The marker identifies a seed region and may cover only an edge; it is not evidence for any category. The next two photos show original left and right local detail without category labels. Resolve ONE identifiable physical unit, not a whole stack or an adjacent occluder. If its complete visible structure can be identified, propose two or three positive points on distinct visible structural parts of that same unit and up to four negative points on visible adjacent or occluding objects. All point coordinates must refer to the FIRST 512x512 image, not the detail photos. At least one positive must remain on the seed region. Do not invent hidden parts, copy the task as a category, or use the gray padding as evidence. If only an edge, multiple units or uncertainty can be identified, give empty point lists. Return object, instance_scope, positive_points, negative_points, evidence.')
def png(im):
 b=io.BytesIO();im.save(b,'PNG');return b.getvalue()
def planning_views(source,mask):
 m=np.asarray(mask,dtype='uint8');ys,xs=np.where(m);assert len(xs) and m.shape==(source.height,source.width);w,h=source.size;pad=max(48,int(.35*max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)));box=[max(0,int(xs.min())-pad),max(0,int(ys.min())-pad),min(w,int(xs.max())+pad+1),min(h,int(ys.max())+pad+1)];crop=source.crop(tuple(box));scale=min(496/crop.width,496/crop.height);dw,dh=round(crop.width*scale),round(crop.height*scale);ox,oy=(512-dw)//2,(512-dh)//2;base=Image.new('RGB',(512,512),(127,127,127));base.paste(crop.resize((dw,dh),Image.Resampling.LANCZOS),(ox,oy));distance=cv2.distanceTransform(np.pad(m,1),cv2.DIST_L2,5)[1:-1,1:-1];ay,ax=map(int,np.unravel_index(distance.argmax(),distance.shape));sx,sy=dw/crop.width,dh/crop.height;display=lambda x,y:(round(ox+(x-box[0])*sx),round(oy+(y-box[1])*sy));tip=display(ax,ay);a=np.array(base).copy();tail=(max(3,min(508,tip[0]+(-90 if tip[0]>256 else 90))),max(3,min(508,tip[1]-90)));cv2.arrowedLine(a,tail,tip,(255,150,0),3,tipLength=.12);pointed=Image.fromarray(a);corner=base.copy();d=ImageDraw.Draw(corner);l,t=display(int(xs.min()),int(ys.min()));r,b=display(int(xs.max())+1,int(ys.max())+1)
 for x,y,dx,dy in [(l,t,1,1),(r,t,-1,1),(l,b,1,-1),(r,b,-1,-1)]:d.line([(x+dx*10,y),(x,y),(x,y+dy*10)],fill=(255,0,255),width=3)
 details=[];detailboxes=[]
 for fraction in [.25,.75]:
  order=np.argsort(xs);k=int(order[min(len(order)-1,round((len(order)-1)*fraction))]);x,y=int(xs[k]),int(ys[k]);half=max(48,min(128,int(.35*(xs.max()-xs.min()+1))));bb=[max(0,x-half),max(0,y-half),min(w,x+half+1),min(h,y+half+1)];details.append(ImageOps.pad(source.crop(tuple(bb)),(512,512),color=(127,127,127)));detailboxes.append(bb)
 return {'pointer':pointed,'corners':corner},details,{'source_box':box,'display_size':[dw,dh],'offset':[ox,oy],'scale_xy':[sx,sy],'anchor_source':[ax,ay],'retained_seed_pixels':int(m[box[1]:box[3],box[0]:box[2]].sum()),'seed_pixels':int(m.sum()),'detail_source_boxes':detailboxes,'complete_seed_in_overview':True,'expected_class_rendered':False}
def bind_points(decision,binding,seed):
 if decision.get('instance_scope')!='single_visible_object':return None,'planner did not identify a single visible object'
 positive=decision.get('positive_points',[]);negative=decision.get('negative_points',[])
 if not 2<=len(positive)<=3 or len(negative)>4:return None,'insufficient positive or excessive prompt points'
 box=binding['source_box'];ox,oy=binding['offset'];sx,sy=binding['scale_xy'];rows=[]
 for label,points in [(1,positive),(0,negative)]:
  for p in points:
   x=(p['x']-ox)/sx+box[0];y=(p['y']-oy)/sy+box[1]
   if not box[0]<=x<box[2] or not box[1]<=y<box[3]:return None,'planner point in padding/outside bound source photo'
   x,y=round(x),round(y)
   if not box[0]<=x<box[2] or not box[1]<=y<box[3]:return None,'rounding escapes bound source photo'
   if any((x-r['x'])**2+(y-r['y'])**2<9 for r in rows):return None,'duplicate/conflicting prompt points'
   rows.append({'x':x,'y':y,'label':label})
 dilated=cv2.dilate(seed.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
 if not any(r['label']==1 and dilated[r['y'],r['x']] for r in rows):return None,'no positive bound to original seed'
 if any(r['label']==0 and seed[r['y'],r['x']] for r in rows):return None,'negative conflicts with original seed'
 return rows,None
def touches_crop_border(mask):return bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
