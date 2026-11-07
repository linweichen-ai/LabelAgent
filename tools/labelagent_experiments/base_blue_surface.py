"""Avoid dark apertures when proposing visible blue-surface points, not masks."""
import cv2,numpy as np


def propose(rgb,parent,anchor):
    rgb=np.asarray(rgb);parent=np.asarray(parent)!=0;h,w=parent.shape;ax,ay=map(int,anchor)
    if rgb.shape!=(h,w,3) or not(0<=ax<w and 0<=ay<h):raise ValueError('Blue surface point geometry mismatch')
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);yy,xx=np.indices((h,w));half=max(4,round(.10*h));blue=parent&(np.abs(yy-ay)<=half)&(hsv[:,:,0]>=90)&(hsv[:,:,0]<=130)&(hsv[:,:,1]>=50)&(hsv[:,:,2]>=30);n,labels,stats,_=cv2.connectedComponentsWithStats(blue.astype('uint8'),8);choices=[]
    for label in range(1,n):
        ys,xs=np.where(labels==label)
        if len(xs)<16:continue
        distance=(xs-ax)**2+(ys-ay)**2;near=int(distance.min())
        if near>(.15*w)**2:continue
        choices.append((near,-len(xs),label))
    base={'half_band_height':half,'blue_hue_range':[90,130],'minimum_saturation':50,'minimum_value':30,'maximum_source_distance_fraction':.15,'no_reference_or_area_inputs':True,'not_complete_base_evidence':True,'parent_and_output_masks_unchanged':True}
    if not choices:return None,{**base,'reason':'No nearby visible blue interior component'}
    label=min(choices)[2];ys,xs=np.where(labels==label);lo,hi=int(xs.min()),int(xs.max());span=hi-lo;diag={**base,'material_component_pixels':len(xs),'component_x_extent':[lo,hi],'span':span,'anchor_value':int(hsv[ay,ax,2]),'candidate_components':len(choices)}
    if span<max(16,.20*w):return None,{**diag,'reason':'Nearest visible blue component lacks distributed span'}
    distance=cv2.distanceTransform(np.pad((labels==label).astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1];inside=distance[ys,xs]>=1.4
    if inside.sum()<2:return None,{**diag,'reason':'No blue component interior'}
    ys,xs=ys[inside],xs[inside];points=[]
    for frac in [.2,.8]:
        target=lo+frac*span;k=int(np.lexsort((ys,xs,(xs-target)**2+(ys-ay)**2))[0]);x,y=int(xs[k]),int(ys[k])
        if (x-ax)**2+(y-ay)**2<9:return None,{**diag,'reason':'Distributed blue cue coincides with old anchor'}
        points.append({'x':x,'y':y,'label':1})
    if len({(q['x'],q['y']) for q in points})!=2:return None,{**diag,'reason':'Duplicate blue cues'}
    return {'points':points,'component_label':label},diag
