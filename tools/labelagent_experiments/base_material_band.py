"""Anchored source material bands select prompts only; never edit masks."""
import cv2,numpy as np


def propose(rgb,parent,anchor):
    rgb=np.asarray(rgb);parent=np.asarray(parent)!=0;h,w=parent.shape;ax,ay=map(int,anchor)
    if rgb.shape!=(h,w,3) or not(0<=ax<w and 0<=ay<h):raise ValueError('Material point geometry mismatch')
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype('float32');patch=lab[max(0,ay-2):min(h,ay+3),max(0,ax-2):min(w,ax+3)];center=np.median(patch.reshape(-1,3),axis=0);distance=np.linalg.norm(lab-center,axis=2);yy,xx=np.indices((h,w));half=max(4,round(.10*h));eligible=parent&(np.abs(yy-ay)<=half)&(distance<=20)
    count,labels,stats,_=cv2.connectedComponentsWithStats(eligible.astype('uint8'),8);near=[]
    for y in range(max(0,ay-2),min(h,ay+3)):
        for x in range(max(0,ax-2),min(w,ax+3)):
            label=int(labels[y,x])
            if label:near.append(((x-ax)**2+(y-ay)**2,-int(stats[label,cv2.CC_STAT_AREA]),label))
    if not near:return None,{'reason':'No parent/material component at anchor','eligible_pixels':int(eligible.sum())}
    label=min(near)[2];ys,xs=np.where(labels==label);lo,hi=int(xs.min()),int(xs.max());span=hi-lo;diag={'material_component_pixels':len(xs),'component_x_extent':[lo,hi],'span':span,'half_band_height':half,'lab_radius':20,'anchor_material_patch':center.tolist(),'no_reference_or_area_inputs':True,'parent_and_output_masks_unchanged':True,'not_complete_base_evidence':True}
    if span<max(16,.20*w):return None,{**diag,'reason':'Source material does not visibly extend across enough input width'}
    points=[]
    for frac in [.2,.8]:
        target=lo+frac*span;order=np.lexsort((ys,xs,(xs-target)**2+(ys-ay)**2));k=int(order[0]);x,y=int(xs[k]),int(ys[k])
        if (x-ax)**2+(y-ay)**2<9:return None,{**diag,'reason':'Distributed cue coincides with original anchor'}
        points.append({'x':x,'y':y,'label':1})
    if len({(p['x'],p['y']) for p in points})!=2:return None,{**diag,'reason':'Duplicate distributed cues'}
    return {'points':points,'component_label':label},diag
