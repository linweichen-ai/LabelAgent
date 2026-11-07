"""Visible parallel source edges propose distributed points; no mask editing."""
import math
import cv2,numpy as np


def propose(rgb,parent,anchor):
    rgb=np.asarray(rgb);parent=np.asarray(parent)!=0;h,w=parent.shape;ax,ay=map(int,anchor)
    if rgb.shape!=(h,w,3) or not(0<=ax<w and 0<=ay<h):raise ValueError('Source point geometry mismatch')
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);gray=cv2.GaussianBlur(gray,(5,5),1);edges=cv2.Canny(gray,50,150)
    raw=cv2.HoughLinesP(edges,1,np.pi/180,20,minLineLength=max(16,round(.25*w)),maxLineGap=max(3,round(.04*w)));lines=[]
    for x1,y1,x2,y2 in ([] if raw is None else raw[:,0].tolist()):
        if x1>x2:x1,y1,x2,y2=x2,y2,x1,y1
        if x1==x2:continue
        angle=math.degrees(math.atan2(y2-y1,x2-x1))
        if abs(angle)<=30:lines.append((x1,y1,x2,y2,angle))
    lines=sorted(set(lines));pairs=[]
    def y(line,x):return line[1]+(x-line[0])*(line[3]-line[1])/(line[2]-line[0])
    for i,a in enumerate(lines):
        for b in lines[i+1:]:
            if abs(a[4]-b[4])>5:continue
            lo=max(a[0],b[0]);hi=min(a[2],b[2]);span=hi-lo
            if span<max(16,.25*w) or not(lo<=ax<=hi):continue
            gaps=[y(b,x)-y(a,x) for x in [lo,ax,hi]]
            if min(gaps)*max(gaps)<=0 or min(map(abs,gaps))<3 or max(map(abs,gaps))>.35*h:continue
            ya,yb=sorted([y(a,ax),y(b,ax)])
            if not(ya-2<=ay<=yb+2):continue
            points=[];valid=True
            for frac in [.2,.8]:
                x=int(round(lo+frac*span));ym=int(round((y(a,x)+y(b,x))/2));near=[]
                for yy in range(max(0,ym-3),min(h,ym+4)):
                    for xx in range(max(0,x-3),min(w,x+4)):
                        ya,yb=sorted([y(a,xx),y(b,xx)])
                        if parent[yy,xx] and ya<yy<yb:near.append(((xx-x)**2+(yy-ym)**2,yy,xx))
                if not near:valid=False;break
                _,yy,xx=min(near);points.append({'x':xx,'y':yy,'label':1})
            if not valid or len({(q['x'],q['y']) for q in points})!=2:continue
            if any((q['x']-ax)**2+(q['y']-ay)**2<9 for q in points):continue
            pairs.append({'lines':[list(a),list(b)],'overlap':[lo,hi],'span':span,'anchor_vertical_distance':abs(ay-(y(a,ax)+y(b,ax))/2),'points':points})
    if not pairs:return None,{'lines':len(lines),'eligible_pairs':0,'reason':'No source-edge pair spans anchor with two parent-interior cues','not_complete_base_evidence':True}
    chosen=min(pairs,key=lambda r:(-r['span'],r['anchor_vertical_distance'],r['lines']))
    return chosen,{'lines':len(lines),'eligible_pairs':len(pairs),'chosen_hypotheses':1,'rule':'widest visible near-parallel source-edge overlap crossing original positive point; two interior points at 20/80 percent','no_reference_or_area_inputs':True,'not_complete_base_evidence':True,'parent_mask_not_modified_or_applied_to_output':True}
