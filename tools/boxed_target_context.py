"""Fixed three-span context with independent point and corner-box display cues."""
import numpy as np
from PIL import ImageDraw
from tools.centered_target_views import centered_target_views
from tools.evaluate_target_cards import fit_panel

def boxed_target_context(source,target,verified_box):
    old,meta=centered_target_views(source,target,verified_box)
    x1,y1,x2,y2=meta['views']['linked']['source_box'];mask=target[y1:y2,x1:x2]
    assert int(mask.sum())==int(target.sum())
    canvas,fit=fit_panel(source.crop((x1,y1,x2,y2)),512,512);draw=ImageDraw.Draw(canvas)
    yy,xx=np.nonzero(mask);sx,sy=fit['display_size'][0]/(x2-x1),fit['display_size'][1]/(y2-y1);ox,oy=fit['offset']
    left,top=round(ox+int(xx.min())*sx),round(oy+int(yy.min())*sy)
    right,bottom=round(ox+(int(xx.max())+1)*sx),round(oy+(int(yy.max())+1)*sy)
    for x,y,dx,dy in ((left,top,1,1),(right,top,-1,1),(left,bottom,1,-1),(right,bottom,-1,-1)):
        draw.line([(x+dx*8,y),(x,y),(x,y+dy*8)],fill=(255,0,255),width=2)
    return {'overview':old['linked'],'linked':canvas},{'verified_source_box':list(verified_box),'source_box':[x1,y1,x2,y2],
        'context_factor':3,'original_target_pixels':int(target.sum()),'retained_target_pixels':int(mask.sum()),
        'same_RGB_source_crop_both_views':True,'cues':['pointer_and_corners','corners_only'],'reference_images_in_queries':False}
