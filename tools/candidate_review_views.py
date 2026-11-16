"""Build local VLM views of a candidate's actual pixels and its context."""
import math

import numpy as np
from PIL import Image, ImageDraw

from source.agent.edit_agent import _rasterize, valid_shape


def candidate_review_views(source, shape, background=(128,128,128)):
    """Return cutout, uncolored context and outlined context without edits."""
    source = source.convert('RGB')
    width,height = source.size
    if not valid_shape({**shape,'id':1},width,height):
        raise ValueError('Invalid review candidate geometry')
    if len(background)!=3 or any(type(value) is not int or not 0<=value<=255 for value in background):
        raise ValueError('Background must be an RGB byte tuple')
    mask = _rasterize(shape,width,height).astype(bool)
    rows,columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError('Review candidate has no pixels')
    left,top,right,bottom = int(columns.min()),int(rows.min()),int(columns.max())+1,int(rows.max())+1
    def bounds(fraction,minimum):
        px,py=max(minimum,(right-left)*fraction),max(minimum,(bottom-top)*fraction)
        return (max(0,math.floor(left-px)),max(0,math.floor(top-py)),
            min(width,math.ceil(right+px)),min(height,math.ceil(bottom+py)))
    tight,context=bounds(.15,12),bounds(.5,32)
    pixels=np.array(source)
    isolated=np.empty_like(pixels)
    isolated[:]=background
    isolated[mask]=pixels[mask]
    target=Image.fromarray(isolated).crop(tight)
    original=source.crop(context)
    outlined=original.copy()
    draw=ImageDraw.Draw(outlined)
    for part in shape.get('parts') or [shape]:
        for ring in [part['points'],*part.get('holes',[])]:
            points=[(x-context[0],y-context[1]) for x,y in ring]
            draw.line(points+[points[0]],fill='#ff00ff',width=2)
    return [target,original,outlined],{'tight_bounds':tight,'context_bounds':context,
        'foreground_pixels':int(mask.sum()),'background_rgb':list(background),
        'source_or_candidate_modified':False}
