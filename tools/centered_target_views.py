"""Mask-centered nested crops inside an already verified source region."""
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tools.evaluate_target_cards import fit_panel
from tools.grounded_target_views import target_anchor, validate_target


def centered_target_views(source, target, verified_box):
    validate_target(target)
    if target.shape != (source.height, source.width):
        raise ValueError('Source and target dimensions differ')
    x1,y1,x2,y2=verified_box
    if not (0<=x1<x2<=source.width and 0<=y1<y2<=source.height):
        raise ValueError('Verified region is outside the source')
    region=target[y1:y2,x1:x2]
    if int(region.sum())!=int(target.sum()):
        raise ValueError('Original target may not be clipped')
    # Source pixels outside this verified region are never available below.
    photo=source.crop(verified_box).convert('RGB')
    yy,xx=np.nonzero(region)
    bounds=(int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1)
    cx,cy=(bounds[0]+bounds[2])/2,(bounds[1]+bounds[3])/2
    side=max(bounds[2]-bounds[0],bounds[3]-bounds[1])
    tip=target_anchor(region)
    views={};metadata={}
    for view,factor in (('overview',2),('linked',3)):
        half=factor*side/2
        box=(max(0,math.floor(cx-half)),max(0,math.floor(cy-half)),min(photo.width,math.ceil(cx+half)),min(photo.height,math.ceil(cy+half)))
        a,b,c,d=box;local=region[b:d,a:c]
        assert int(local.sum())==int(target.sum())
        picture,fit=fit_panel(photo.crop(box),512,512)
        sx,sy=fit['display_size'][0]/(c-a),fit['display_size'][1]/(d-b)
        ox,oy=fit['offset'];display=lambda p:(round(ox+(p[0]-a)*sx),round(oy+(p[1]-b)*sy))
        tail=(max(a,min(c-1,tip[0]+(-.25*(c-a) if tip[0]>cx else .25*(c-a)))),max(b,min(d-1,tip[1]-.3*(d-b))))
        cv=np.array(picture);cv2.arrowedLine(cv,display(tail),display(tip),(255,150,0),thickness=3,tipLength=.12)
        picture=Image.fromarray(cv);draw=ImageDraw.Draw(picture)
        left,top=display(bounds[:2]);right,bottom=display(bounds[2:])
        for x,y,dx,dy in ((left,top,1,1),(right,top,-1,1),(left,bottom,1,-1),(right,bottom,-1,-1)):
            draw.line([(x+dx*8,y),(x,y),(x,y+dy*8)],fill=(255,0,255),width=2)
        views[view]=picture
        metadata[view]={'factor':factor,'source_box':[x1+a,y1+b,x1+c,y1+d],
            'retained_target_pixels':int(local.sum()),'source_size':[c-a,d-b],'fit':fit}
    return views,{'verified_source_box':list(verified_box),'original_target_pixels':int(target.sum()),
                  'target_anchor_source':[x1+tip[0],y1+tip[1]],'all_target_retained':True,
                  'outside_verified_region_unavailable':True,'geometry_changed':False,'views':metadata}
