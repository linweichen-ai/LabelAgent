"""RGB material cues select prompts only; never cut or constrain a SAM mask."""
import cv2,numpy as np


def propose(rgb,anchor):
    rgb=np.asarray(rgb);h,w=rgb.shape[:2];ax,ay=anchor
    if not(0<=ax<w and 0<=ay<h):raise ValueError('Original anchor outside source ROI')
    r,g,b=rgb.astype('int16').transpose(2,0,1)
    cue=(((r-b>=8)&(g-b>=4)&(r>=80))|((b-r>=8)&(b>=60))).astype('uint8')
    count,labels,stats,_=cv2.connectedComponentsWithStats(cue,8);regions=[]
    for i in range(1,count):
        if stats[i,cv2.CC_STAT_AREA]<16:continue
        m=(labels==i).astype('uint8');d=cv2.distanceTransform(np.pad(m,1),cv2.DIST_L2,5)[1:-1,1:-1];ys,xs=np.where(d>=1.4)
        if not len(xs):continue
        dist=(xs-ax)**2+(ys-ay)**2;k=int(np.argmin(dist));nearest=(int(xs[k]),int(ys[k]));dy,dx=map(int,np.unravel_index(d.argmax(),d.shape));regions.append({'label':i,'nearest':nearest,'deepest':(dx,dy),'nearest_squared_distance':float(dist[k]),'cue_pixels':int(m.sum()),'interior_radius':float(d[dy,dx])})
    if not regions:return [],cue,{'reason':'No nonneutral material interior','no_semantic_or_instance_claim':True}
    chosen=min(regions,key=lambda z:(z['nearest_squared_distance'],-z['cue_pixels'],z['label']));rows=[]
    for name,xy in [('nearest_material_interior',chosen['nearest']),('deep_material_interior',chosen['deepest'])]:
        if any((xy[0]-p['points'][0]['x'])**2+(xy[1]-p['points'][0]['y'])**2<9 for p in rows):continue
        rows.append({'cue':name,'points':[{'x':xy[0],'y':xy[1],'label':1}],'source_is_RGB_only':True,'mask_prior':None})
    return rows,cue,{'chosen_region':chosen,'eligible_components':len(regions),'hypothesis_limit':2,'cue_area_not_output_mask_or_acceptance_budget':True,'no_reference_inputs':True,'no_output_mask_clipping':True}
