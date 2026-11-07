"""Independently enforce original positive/negative native-bitmap constraints."""
import numpy as np


def evaluate(mask,points,negative_radius=3):
    m=np.asarray(mask)
    if m.ndim!=2 or not np.isin(m,[0,1]).all():raise ValueError('Expected complete binary native bitmap')
    if negative_radius!=3:raise ValueError('Keep original three-pixel negative neighborhood')
    h,w=m.shape;rows=[]
    for i,p in enumerate(points):
        x,y=int(p['x']),int(p['y']);label=p['label']
        if label not in [0,1] or not(0<=p['x']<w and 0<=p['y']<h):raise ValueError('Native prompt geometry mismatch')
        covered=int(m[max(0,y-3):min(h,y+4),max(0,x-3):min(w,x+4)].sum()) if label==0 else int(m[y,x])
        rows.append({'index':i,'label':label,'center_covered':bool(m[y,x]),'pixels_in_required_region':covered,'valid':bool(covered>0) if label else covered==0})
    return {'valid':all(r['valid'] for r in rows),'negative_radius':3,'points':rows,'failed_indices':[r['index'] for r in rows if not r['valid']],'complete_bitmap_not_modified':True}
