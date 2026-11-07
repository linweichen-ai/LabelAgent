"""Local textured material prompts away from a captioned background hypothesis."""
import cv2,numpy as np


def propose(rgb,anchor):
    rgb=np.asarray(rgb);h,w=rgb.shape[:2];ax,ay=anchor;r,g,b=rgb.astype('int16').transpose(2,0,1);cue=(((r-b>=8)&(g-b>=4)&(r>=80))|((b-r>=8)&(b>=60))).astype('uint8');distance=cv2.distanceTransform(np.pad(cue,1),cv2.DIST_L2,5)[1:-1,1:-1];yy,xx=np.indices(cue.shape);squared=(xx-ax)**2+(yy-ay)**2;eligible=((distance>=1.4)&(squared>=9)&(squared<=64**2)).astype('uint8');gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);corners=cv2.goodFeaturesToTrack(gray,mask=eligible*255,maxCorners=64,qualityLevel=.01,minDistance=5,blockSize=7,useHarrisDetector=False);items=[]
    for x,y in ([] if corners is None else corners[:,0].tolist()):
        x,y=int(round(x)),int(round(y))
        if eligible[y,x]:items.append((float((x-ax)**2+(y-ay)**2),y,x))
    rows=[]
    for _,y,x in sorted(items)[:2]:rows.append({'cue':'textured_material_corner_'+str(len(rows)),'points':[{'x':x,'y':y,'label':1},{'x':int(ax),'y':int(ay),'label':0}],'negative_is_unconfirmed_background_hypothesis':True,'mask_prior':None})
    return rows,eligible,{'raw_material_corners':len(items),'selected_limit':2,'radius_fixed_px':64,'no_reference_or_area_input':True,'no_output_mask_changes':True,'negative_caption_not_ground_truth':True}
