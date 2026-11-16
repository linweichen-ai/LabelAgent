"""Image-mask-only sampling for isolated instance-consistency experiments."""
import cv2
import numpy as np

from source.agent.edit_agent import _rasterize


def distributed_union_points(candidate,first_point,width,height,count=3,margin=2.,separation=8.):
    """Keep the original overlap click, then use deterministic farthest sampling.

    Points are separate single-positive SAM requests, never combined prompts.
    Inadequate interior area fails sampling rather than inventing repeat clicks.
    """
    if type(count) is not int or not 1<=count<=5 or margin<=0 or separation<=0:
        raise ValueError('Invalid point count, interior margin or separation')
    mask=_rasterize(candidate,width,height)
    x,y=first_point['x'],first_point['y']
    if not 0<=x<width or not 0<=y<height or not mask[y,x] or first_point.get('label')!=1:
        raise ValueError('First positive point must belong to the candidate')
    distance=cv2.distanceTransform(np.pad(mask,1),cv2.DIST_L2,5)[1:-1,1:-1]
    interior=distance>=margin
    if not interior[y,x]:return []
    points=[dict(first_point)]
    yy,xx=np.indices(mask.shape,dtype=np.float32)
    nearest=np.full(mask.shape,np.inf,dtype=np.float32)
    while len(points)<count:
        point=points[-1]
        np.minimum(nearest,(xx-point['x'])**2+(yy-point['y'])**2,out=nearest)
        ranked=np.where(interior,nearest,-1)
        index=int(np.argmax(ranked))
        py,px=np.unravel_index(index,ranked.shape)
        if ranked[py,px]<separation**2:return []
        points.append({'x':int(px),'y':int(py),'label':1})
    return points
