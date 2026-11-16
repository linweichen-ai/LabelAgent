"""Three- and four-span local context, retaining the checked target and region."""
import math
import cv2
import numpy as np
from PIL import Image,ImageDraw
from tools.centered_target_views import centered_target_views
from tools.evaluate_target_cards import fit_panel
from tools.grounded_target_views import target_anchor,validate_target

def context_target_views(source,target,verified_box):
    old,oldmeta=centered_target_views(source,target,verified_box)
    x1,y1,x2,y2=verified_box;local=target[y1:y2,x1:x2];photo=source.crop(verified_box).convert('RGB')
    yy,xx=np.nonzero(local);bounds=(int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1)
    cx,cy=(bounds[0]+bounds[2])/2,(bounds[1]+bounds[3])/2;half=2*max(bounds[2]-bounds[0],bounds[3]-bounds[1])
    a,b,c,d=max(0,math.floor(cx-half)),max(0,math.floor(cy-half)),min(photo.width,math.ceil(cx+half)),min(photo.height,math.ceil(cy+half))
    assert int(local[b:d,a:c].sum())==int(target.sum())
    picture,fit=fit_panel(photo.crop((a,b,c,d)),512,512);sx,sy=fit['display_size'][0]/(c-a),fit['display_size'][1]/(d-b);ox,oy=fit['offset']
    display=lambda p:(round(ox+(p[0]-a)*sx),round(oy+(p[1]-b)*sy));tip=target_anchor(local)
    tail=(max(a,min(c-1,tip[0]+(-.25*(c-a) if tip[0]>cx else .25*(c-a)))),max(b,min(d-1,tip[1]-.3*(d-b))))
    array=np.array(picture);cv2.arrowedLine(array,display(tail),display(tip),(255,150,0),thickness=3,tipLength=.12)
    picture=Image.fromarray(array);draw=ImageDraw.Draw(picture);left,top=display(bounds[:2]);right,bottom=display(bounds[2:])
    for x,y,dx,dy in ((left,top,1,1),(right,top,-1,1),(left,bottom,1,-1),(right,bottom,-1,-1)):
        draw.line([(x+dx*8,y),(x,y),(x,y+dy*8)],fill=(255,0,255),width=2)
    return {'overview':old['linked'],'linked':picture},{'verified_source_box':list(verified_box),'original_target_pixels':int(target.sum()),
        'all_target_retained':True,'outside_verified_region_unavailable':True,'geometry_changed':False,
        'views':{'overview':oldmeta['views']['linked'],'linked':{'factor':4,'source_box':[x1+a,y1+b,x1+c,y1+d],'source_size':[c-a,d-b],'retained_target_pixels':int(local[b:d,a:c].sum()),'fit':fit}}}
