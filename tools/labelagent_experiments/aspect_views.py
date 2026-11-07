"""Retain native rectangular content within a fixed 256-token input canvas."""
import math
from PIL import Image,ImageOps
from rectangular_views import render
GRIDS=((4,64),(8,32),(16,16),(32,8),(64,4))
def aspect_view(source,binding):
 _,native,_=render(source,binding)
 gx,gy=min(GRIDS,key=lambda g:(abs(math.log((native.width/native.height)/(g[0]/g[1]))),g))
 size=(gx*32,gy*32);fitted=ImageOps.contain(native,size,Image.Resampling.LANCZOS);canvas=Image.new('RGB',size,(127,127,127));offset=((size[0]-fitted.width)//2,(size[1]-fitted.height)//2);canvas.paste(fitted,offset)
 scale_xy=[fitted.width/native.width,fitted.height/native.height]
 meta={'output_size':list(size),'grid_width_height':[gx,gy],'expected_actual_tokens':gx*gy,'fitted_size':list(fitted.size),'offset':list(offset),'native_size':list(native.size),'content_fraction':fitted.width*fitted.height/(size[0]*size[1]),'candidate_area_fraction_estimate':binding['candidate_pixels']*scale_xy[0]*scale_xy[1]/(size[0]*size[1]),'square_candidate_fraction':binding['candidate_pixels']/max(native.size)**2,'no_crop_of_native_rectangle':True,'not_additional_native_detail':True}
 assert gx*gy==256 and fitted.width<=size[0] and fitted.height<=size[1]
 return canvas,meta
