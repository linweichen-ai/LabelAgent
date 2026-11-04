"""Global extra-candidate allocation while keeping every existing anchor mask."""
import math
import cv2
import numpy as np
from tools.tiled_candidate_utils import _iou


def point_hits(record,pixels,radius,width,height):
    if type(radius) is not int or radius<0:raise ValueError('Nonnegative integer radius required')
    pixels=np.asarray(pixels)
    if pixels.ndim!=2 or pixels.shape[1]!=2:raise ValueError('Aligned pixel pairs required')
    l,t,r,b=record['bbox'];left=max(0,l-radius);top=max(0,t-radius)
    right=min(width,r+radius);bottom=min(height,b+radius)
    mask=np.pad(record['mask'],((t-top,bottom-b),(l-left,right-r)))
    if radius:mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    selected=(pixels[:,0]>=left)&(pixels[:,0]<right)&(pixels[:,1]>=top)&(pixels[:,1]<bottom)
    result=np.zeros(len(pixels),bool);rows=np.flatnonzero(selected)
    result[rows]=mask[pixels[rows,1]-top,pixels[rows,0]-left]!=0
    return result


def union_pixels(records):
    if not records:return 0
    left=min(r['bbox'][0] for r in records);top=min(r['bbox'][1] for r in records)
    right=max(r['bbox'][2] for r in records);bottom=max(r['bbox'][3] for r in records)
    value=np.zeros((bottom-top,right-left),bool)
    for record in records:
        l,t,r,b=record['bbox'];value[t-top:b-top,l-left:r-left]|=record['mask']!=0
    return int(value.sum())


def marginal_pixels(record,selected):
    left,top,right,bottom=record['bbox'];covered=np.zeros(record['mask'].shape,bool)
    for other in selected:
        l,t,r,b=other['bbox'];x0=max(left,l);y0=max(top,t);x1=min(right,r);y1=min(bottom,b)
        if x0<x1 and y0<y1:
            covered[y0-top:y1-top,x0-left:x1-left]|=other['mask'][y0-t:y1-t,x0-l:x1-l]!=0
    return int(((record['mask']!=0)&~covered).sum())


def allocate(anchors,candidates,maximum_count,maximum_area,mode,nms_iou=.8):
    if mode not in ('r3','all'):raise ValueError('Unknown raw-point guard')
    if any(type(v) is not int or v<0 for v in (maximum_count,maximum_area)):raise ValueError('Integer global caps required')
    records={key:list(values) for key,values in anchors.items()}
    count=sum(len(v) for v in records.values());area=sum(union_pixels(v) for v in records.values())
    if count>maximum_count or area>maximum_area:raise ValueError('Existing anchors already exceed original global caps')
    if any(not math.isfinite(c['priority']) or not 0<=c['priority']<=1 for c in candidates):raise ValueError('Finite bounded priority required')
    ordered=sorted(candidates,key=lambda c:(-c['priority'],int(c['key'][1]),c['key'][0],str(c['shape']['id'])))
    accepted=[];reasons={'raw_guard':0,'nms':0,'area':0}
    for candidate in ordered:
        if count>=maximum_count:break
        if not candidate['allowed'][mode]:reasons['raw_guard']+=1;continue
        key=candidate['key'];record=candidate['record']
        if any(_iou(record,other)>=nms_iou for other in records[key]):reasons['nms']+=1;continue
        added=marginal_pixels(record,records[key])
        if area+added>maximum_area:reasons['area']+=1;continue
        records[key].append(record);accepted.append(candidate);count+=1;area+=added
    if area!=sum(union_pixels(v) for v in records.values()):raise ValueError('Union-area independent readback differs')
    return accepted,{'pallet_candidates':count,'pallet_union_pixels':area,'extra_candidates':len(accepted),
        'maximum_candidates':maximum_count,'maximum_union_pixels':maximum_area,'rejected':reasons}
