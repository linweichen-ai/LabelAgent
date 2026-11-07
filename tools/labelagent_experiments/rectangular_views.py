"""Anchor-centered rectangular crops retain every unconfirmed mask pixel."""
import math
import numpy as np,cv2
from PIL import Image

def render(source,binding,size=1024):
 box=binding['canvas_box'];clip=binding['source_box'];native=Image.new('RGB',(box[2]-box[0],box[3]-box[1]),(127,127,127));native.paste(source.crop(tuple(clip)),(clip[0]-box[0],clip[1]-box[1]));side=max(native.size);square=Image.new('RGB',(side,side),(127,127,127));offset=((side-native.width)//2,(side-native.height)//2);square.paste(native,offset)
 return square.resize((size,size),Image.Resampling.LANCZOS),native,offset

def rectangular_views(source,mask,size=1024):
 m=(np.asarray(mask)!=0).astype('uint8');assert m.shape==(source.height,source.width)
 ys,xs=np.where(m)
 if not len(xs):raise ValueError('Empty candidate')
 distance=cv2.distanceTransform(np.pad(m,1),cv2.DIST_L2,5)[1:-1,1:-1];ay,ax=map(int,np.unravel_index(distance.argmax(),distance.shape));rx=int(abs(xs-ax).max())+9;ry=int(abs(ys-ay).max())+9;views={};bindings=[]
 for name,factor in [('near',1),('context',1.5)]:
  hx,hy=int(math.ceil(rx*factor)),int(math.ceil(ry*factor));box=[ax-hx,ay-hy,ax+hx+1,ay+hy+1];clip=[max(0,box[0]),max(0,box[1]),min(source.width,box[2]),min(source.height,box[3])]
  assert int(m[clip[1]:clip[3],clip[0]:clip[2]].sum())==int(m.sum())
  binding={'view':name,'source_box':clip,'canvas_box':box,'foreground_anchor':[ax,ay],'center_on_mask':bool(m[ay,ax]),'all_candidate_pixels_retained':True,'candidate_pixels':int(m.sum()),'candidate_not_confirmed_physical_extent':True,'padding_RGB':[127,127,127],'context_factor':factor}
  image,native,offset=render(source,binding,size);binding.update(native_canvas_size=list(native.size),square_padding_offset=list(offset),output_size=[size,size]);views[name]=image;bindings.append(binding)
 return views,bindings
