"""Deterministic visible-color components propose parts, never edit output masks."""
import cv2,numpy as np

def propose(rgb,stock,observed,max_components=2):
    rgb=np.asarray(rgb,dtype='uint8');stock=np.asarray(stock)!=0;observed=np.asarray(observed)!=0
    if rgb.shape!=(*stock.shape,3) or observed.shape!=stock.shape:raise ValueError('Exact source geometry required')
    inside=stock&observed;yy,xx=np.where(inside)
    if len(xx)<24:return [],{'reason':'too_few_real_stock_pixels','component_hypotheses':[]}
    # Learn source colors on a bounded deterministic lattice; query RGB and
    # stock output remain immutable. Labels are proposals, never final masks.
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype('float32');values=lab[inside];sample=values[::max(1,len(values)//4096)]
    if np.max(np.ptp(sample,axis=0))<4:return [],{'reason':'uniform_source_region','component_hypotheses':[]}
    cv2.setRNGSeed(0);_,_,centers=cv2.kmeans(sample,3,None,(cv2.TERM_CRITERIA_MAX_ITER|cv2.TERM_CRITERIA_EPS,30,.05),1,cv2.KMEANS_PP_CENTERS)
    color_order=sorted(range(3),key=lambda i:tuple(float(x) for x in centers[i]));centers=centers[color_order];labels=np.sum((values[:,None]-centers[None])**2,axis=2).argmin(1);records=[]
    for color in range(3):
        region=np.zeros(stock.shape,'uint8');region[yy[labels==color],xx[labels==color]]=1;count,components,stats,_=cv2.connectedComponentsWithStats(region,8);chroma=float(np.linalg.norm(centers[color,1:]-128))
        for component in range(1,count):
            x,y,w,h,area=map(int,stats[component]);fraction=area/len(xx)
            if area<9 or w<3 or h<2:continue
            local=components[y:y+h,x:x+w]==component;distance=cv2.distanceTransform(np.pad(local.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1];dy,dx=map(int,np.unravel_index(distance.argmax(),distance.shape));point=[x+dx,y+dy]
            # Prefer actual chromatic structure, broad support, and a source
            # interior. There is no expected pallet color or reference point.
            score=(1+chroma)*np.sqrt(area)*min(4,w/max(1,h));records.append({'color':color,'component':component,'point':point,'area':area,'bbox':[x,y,w,h],'chroma':chroma,'area_fraction':fraction,'score':float(score),'distance_to_component_edge':float(distance[dy,dx])})
    selected=[]
    for row in sorted(records,key=lambda r:(-r['score'],r['point'][1],r['point'][0],r['color'])):
        x,y=row['point']
        if any((x-r['point'][0])**2+(y-r['point'][1])**2<12**2 for r in selected):continue
        selected.append(row)
        if len(selected)>=max_components:break
    output=[]
    for row in selected:
        x,y=row['point'];near=[]
        for other in records:
            nx,ny=other['point'];delta=ny-y
            if other is row or abs(delta)<8 or (nx-x)**2+(ny-y)**2<16**2:continue
            left=max(row['bbox'][0],other['bbox'][0]);right=min(row['bbox'][0]+row['bbox'][2],other['bbox'][0]+other['bbox'][2])
            if right-left<min(row['bbox'][2],other['bbox'][2])*.25:continue
            near.append((abs(delta),-other['area'],ny,nx,other))
        negatives=[]
        for sign in [-1,1]:
            options=[r for r in near if (r[4]['point'][1]-y)*sign>0]
            if options:negatives.append(min(options,key=lambda r:r[:4])[4])
        output.append(row|{'points':[{'x':x,'y':y,'label':1}],'negative_hypotheses':[{'x':r['point'][0],'y':r['point'][1],'label':0} for r in negatives],'negative_component_records':negatives,'negative_physical_identity_unconfirmed':True})
    return output,{'lab_centers':centers.tolist(),'component_hypotheses':records,'real_stock_pixels':len(xx),'selected_components':len(output),'no_reference_or_area_inputs':True,'not_an_instance_identity_or_mask_acceptance_claim':True,'no_output_mask_was_edited':True}
