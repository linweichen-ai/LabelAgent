"""Whole-input box prompts from existing RGB click midpoints; no mask cropping."""

def propose(points,width,height):
    anchor=points[0]
    if anchor['label']!=1:raise ValueError('Original positive anchor must remain first')
    above=[q for q in points if q['label']==0 and q['y']<anchor['y']]
    below=[q for q in points if q['label']==0 and q['y']>anchor['y']]
    if not above or not below:return None,{'reason':'No existing RGB negative hypotheses on both vertical sides'}
    upper=max(above,key=lambda q:q['y']);lower=min(below,key=lambda q:q['y']);top=(anchor['y']+upper['y'])/2;bottom=(anchor['y']+lower['y'])/2;box=[0.,float(top),float(width-1),float(bottom)]
    if not(0<=top<bottom<height):return None,{'reason':'Source midpoint box outside original input'}
    if any(not(box[0]<=q['x']<=box[2] and top<=q['y']<=bottom) for q in points if q['label']==1):return None,{'reason':'An original positive would lie outside proposed layer box'}
    return box,{'rule':'Full original input width; vertical boundaries halfway from original first positive to closest existing negative above/below','upper_negative':upper,'lower_negative':lower,'all_original_positive_points_inside_box':True,'no_reference_points_depth_pose_labels_or_area_inputs':True,'box_is_soft_prompt_not_crop_or_output_clip':True,'physical_single_layer_extent_unconfirmed':True}
