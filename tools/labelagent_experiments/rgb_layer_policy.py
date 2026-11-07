"""Frozen RGB edge pairs propose points, never alter a returned mask."""
import math
import cv2
import numpy as np

def propose(rgb,seed,anchor):
 h,w=seed.shape;ax,ay=anchor;half=max(24,min(96,round(.25*w)))
 gray=cv2.cvtColor(np.asarray(rgb),cv2.COLOR_RGB2GRAY)
 smooth=cv2.GaussianBlur(gray,(5,5),1.0);edges=cv2.Canny(smooth,50,150,apertureSize=3)
 found=cv2.HoughLinesP(edges,1,np.pi/180,20,minLineLength=max(12,round(.20*w)),maxLineGap=max(3,round(.04*w)))
 lines=[]
 for x1,y1,x2,y2 in ([] if found is None else found[:,0].tolist()):
  if x1>x2:x1,y1,x2,y2=x2,y2,x1,y1
  if x2==x1:continue
  angle=math.degrees(math.atan2(y2-y1,x2-x1))
  if abs(angle)<=15:lines.append((x1,y1,x2,y2,angle))
 lines=sorted(set(lines));pairs=[];gradient=np.abs(cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3))
 for i,a in enumerate(lines):
  for b in lines[i+1:]:
   if abs(a[4]-b[4])>5:continue
   lo=max(a[0],b[0],ax-half,0);hi=min(a[2],b[2],ax+half,w-1)
   if hi-lo<max(12,round(.30*(2*half))):continue
   y=lambda line,x:line[1]+(x-line[0])*(line[3]-line[1])/(line[2]-line[0])
   gaps=[y(b,x)-y(a,x) for x in (lo,(lo+hi)/2,hi)]
   if min(gaps)*max(gaps)<=0 or min(map(abs,gaps))<3 or max(map(abs,gaps))>min(60,round(.20*h)):continue
   midx=(lo+hi)/2;midy=(y(a,midx)+y(b,midx))/2
   contrast=float(np.mean([gradient[min(h-1,max(0,round(y(line,x)))),round(x)] for line in (a,b) for x in np.linspace(lo,hi,10)]))
   pairs.append(((midx-ax)**2+(midy-ay)**2,-contrast,-(hi-lo),a,b,lo,hi))
 dilated=cv2.dilate(seed.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
 output=[];rejected=[]
 # Choose two geometric hypotheses first; rejected binding cannot trigger a search for a third.
 for rank,pair in enumerate(sorted(pairs)[:2]):
  _,_,_,a,b,lo,hi=pair;y=lambda line,x:line[1]+(x-line[0])*(line[3]-line[1])/(line[2]-line[0])
  xs=[lo+.25*(hi-lo),lo+.75*(hi-lo)]+([ax] if lo<=ax<=hi else [])
  points=[{'x':round(x),'y':round((y(a,x)+y(b,x))/2),'label':1} for x in xs]
  valid=all(0<=p['x']<w and 0<=p['y']<h for p in points)
  valid=valid and all((p['x']-q['x'])**2+(p['y']-q['y'])**2>=9 for k,p in enumerate(points) for q in points[:k])
  valid=valid and any(dilated[p['y'],p['x']] for p in points)
  row={'rank':rank,'lines':[a,b],'points':points,'valid':bool(valid),'no_reference_inputs':True}
  (output if valid else rejected).append(row)
 return output,{'lines':lines,'pairs':len(pairs),'rejected':rejected,'chosen_pair_limit':2}
