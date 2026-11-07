"""RGB seam box hypotheses from frozen model masks, never reference labels."""
import cv2,numpy as np


def propose(rgb,mask):
    m=(np.asarray(mask)!=0).astype('uint8');ys,xs=np.where(m)
    if len(xs)<10:return [],{'reason':'Too few parent pixels'}
    x1,x2=int(xs.min()),int(xs.max());y1,y2=int(ys.min()),int(ys.max());h,w=m.shape
    if x2-x1<12 or y2-y1<8:return [],{'reason':'Parent too small for two independent box hypotheses'}
    gray=cv2.cvtColor(np.asarray(rgb),cv2.COLOR_RGB2GRAY);gradient=np.abs(cv2.Sobel(cv2.GaussianBlur(gray,(3,3),.7),cv2.CV_32F,0,1,ksize=3))
    lo=y1+max(2,round(.25*(y2-y1)));hi=y1+min(y2-y1-2,round(.75*(y2-y1)))
    score=np.mean(gradient[lo:hi+1,x1:x2+1],axis=1);seam=lo+int(np.argmax(score));rows=[]
    for side,top,bottom in [('upper',y1,seam),('lower',seam+1,y2)]:
        region=m.copy();region[:top]=0;region[bottom+1:]=0
        if region.sum()<5:continue
        distance=cv2.distanceTransform(np.pad(region,1),cv2.DIST_L2,5)[1:-1,1:-1];ay,ax=map(int,np.unravel_index(distance.argmax(),m.shape))
        box=[max(0,x1-2),max(0,top-2),min(w-1,x2+2),min(h-1,bottom+2)]
        if not(box[0]<box[2] and box[1]<box[3]):continue
        rows.append({'side':side,'diagnostic_box':box,'points':[{'x':ax,'y':ay,'label':1}],'seam':seam,'parent_pixels':int(m.sum()),'point_from_parent_interior':True,'no_reference_inputs':True})
    return rows,{'parent_bbox':[x1,y1,x2,y2],'RGB_seam':seam,'seam_mean_gradient':float(score.max()),'RGB_only_seam':True}
