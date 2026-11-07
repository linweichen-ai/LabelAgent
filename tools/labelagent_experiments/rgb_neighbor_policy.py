"""Image-only layer prompts. Regions select points, never crop any output."""
import cv2
import numpy as np


def propose(rgb, parent):
    rgb=np.asarray(rgb);m=(np.asarray(parent)!=0).astype('uint8')
    if rgb.shape[:2]!=m.shape:raise ValueError('Source geometry differs')
    yy,xx=np.where(m)
    if len(xx)<20:return [],{'reason':'Insufficient parent pixels'}
    x1,x2,y1,y2=map(int,[xx.min(),xx.max(),yy.min(),yy.max()]);h=y2-y1+1;w=x2-x1+1
    if h<15 or w<20:return [],{'reason':'Parent too small for layer hypotheses'}
    gray=cv2.GaussianBlur(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),(3,3),.7)
    edges=cv2.Canny(gray,50,150);lines=cv2.HoughLinesP(edges[y1:y2+1,x1:x2+1],1,np.pi/180,15,minLineLength=max(12,round(.35*w)),maxLineGap=max(3,round(.05*w)))
    slopes=[]
    for a,b,c,d in ([] if lines is None else lines[:,0].tolist()):
        if a==c:continue
        slope=(d-b)/(c-a)
        if abs(slope)<=np.tan(np.pi/9):slopes.append((slope,abs(c-a)))
    slopes.sort();slope=0.
    if slopes:
        halfway=sum(n for _,n in slopes)/2;acc=0
        for value,n in slopes:
            acc+=n
            if acc>=halfway:slope=float(value);break
    cy,cx=np.indices(m.shape);center=(x1+x2)/2
    coordinate=cy-slope*(cx-center);lo=float(coordinate[m!=0].min());hi=float(coordinate[m!=0].max());height=hi-lo
    # Only RGB vertical gradients determine seams, before any SAM output/GT.
    gradient=np.abs(cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3));profile=[]
    horizontal=(cx>=x1+.15*w)&(cx<=x2-.15*w)
    for value in range(int(np.ceil(lo+.18*height)),int(np.floor(hi-.18*height))+1):
        selected=horizontal&(np.abs(coordinate-value)<.5)
        profile.append((float(gradient[selected].mean()) if selected.any() else 0.,value))
    seams=[]
    for score,value in sorted(profile,key=lambda x:(-x[0],x[1])):
        if score<=0:break
        if all(abs(value-old)>=max(4,.23*height) for old in seams):seams.append(value)
        if len(seams)==2:break
    if not seams:return [],{'reason':'No RGB seam','slope':slope}
    seams.sort();bounds=[lo-1]+seams+[hi+1];bands=[]
    for band,(top,bottom) in enumerate(zip(bounds,bounds[1:])):
        region=(m!=0)&(coordinate>top)&(coordinate<=bottom)
        distance=cv2.distanceTransform(np.pad(region.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1]
        if distance.max()<1.5:continue
        py,px=map(int,np.unravel_index(distance.argmax(),m.shape));bands.append({'band':band,'anchor':{'x':px,'y':py,'label':1},'interior_distance':float(distance[py,px]),'parent_pixels':int(region.sum())})
    output=[]
    # Same two hypotheses per teacher as the preceding upper/lower policy.
    for rank,b in enumerate(sorted(bands,key=lambda b:(-b['interior_distance'],b['band']))[:2]):
        points=[b['anchor']]+[dict(o['anchor'],label=0) for o in bands if o['band']!=b['band']]
        if any((a['x']-b['x'])**2+(a['y']-b['y'])**2<9 for i,a in enumerate(points) for b in points[:i]):continue
        output.append({'rank':rank,'band':b['band'],'points':points,'no_reference_inputs':True,'all_neighbor_centers_excluded':True})
    return output,{'parent_bbox':[x1,y1,x2,y2],'RGB_slope':slope,'RGB_seams':seams,'bands':bands,'chosen_hypothesis_limit':2,'seams_are_hypotheses_not_object_truth':True,'no_output_mask_changes':True}
