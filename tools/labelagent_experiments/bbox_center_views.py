"""Use complete mask extents to center review, never remove an output pixel."""
import math,numpy as np
from aspect_views import aspect_view


def views(source,mask):
    yy,xx=np.where(np.asarray(mask)!=0)
    if len(xx)==0:raise ValueError('Empty candidate')
    left,right=int(xx.min()),int(xx.max());top,bottom=int(yy.min()),int(yy.max());cx=(left+right)//2;cy=(top+bottom)//2;rx=max(cx-left,right-cx)+9;ry=max(cy-top,bottom-cy)+9;result={};bindings=[]
    for name,factor in [('near',1),('context',1.5)]:
        hx=math.ceil(rx*factor);hy=math.ceil(ry*factor);box=[cx-hx,cy-hy,cx+hx+1,cy+hy+1];clip=[max(0,box[0]),max(0,box[1]),min(source.width,box[2]),min(source.height,box[3])]
        assert int((mask[clip[1]:clip[3],clip[0]:clip[2]]!=0).sum())==len(xx)
        b={'view':name,'source_box':clip,'canvas_box':box,'foreground_anchor':[cx,cy],'center_definition':'complete_candidate_bbox_center','center_is_foreground':bool(mask[cy,cx]),'candidate_pixels':len(xx),'all_candidate_pixels_retained':True,'not_confirmed_physical_object_extent':True,'context_factor':factor};im,geometry=aspect_view(source,b);b['geometry']=geometry;result[name]=im;bindings.append(b)
    return result,bindings
