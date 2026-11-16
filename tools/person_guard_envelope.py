"""Build conservative regions from existing person predictions, without edits."""
import cv2
import numpy as np
from source.agent.edit_agent import _rasterize,valid_shape


def person_envelope_mask(shapes,width,height,mode):
    if mode not in ('mask','convex_hull','bounding_box'):
        raise ValueError('Unknown person envelope')
    if type(width) is not int or type(height) is not int or min(width,height)<=0:
        raise ValueError('Positive integer image dimensions required')
    union=np.zeros((height,width),np.uint8)
    for shape in shapes:
        if shape.get('label')!='person':
            continue
        if not valid_shape({**shape,'id':1},width,height):
            raise ValueError('Invalid person prediction geometry')
        mask=_rasterize(shape,width,height)
        rows,columns=np.nonzero(mask)
        if not len(rows):
            raise ValueError('Empty person mask')
        if mode=='mask':
            union|=mask
        elif mode=='bounding_box':
            union[int(rows.min()):int(rows.max())+1,int(columns.min()):int(columns.max())+1]=1
        else:
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            hull=cv2.convexHull(np.concatenate(contours))
            cv2.fillConvexPoly(union,hull,1)
    return union
