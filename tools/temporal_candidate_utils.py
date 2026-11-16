"""Offline image-only mask transport; inherited scores are not target confidence."""
import math

import cv2
import numpy as np

from source.agent.edit_agent import _rasterize,_shape_from_mask,valid_shape


def estimate_bidirectional_flow(source,target,config):
    """Return source->target and target->source fields on a fixed image grid."""
    source,target=np.asarray(source),np.asarray(target)
    if source.shape!=target.shape or source.ndim!=3 or source.shape[2]!=3 or source.dtype!=np.uint8 or target.dtype!=np.uint8:
        raise ValueError('Expected two equally sized uint8 RGB images')
    size=(config['width'],config['height'])
    def gray(image):
        return cv2.resize(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),size,interpolation=cv2.INTER_AREA)
    first,second=gray(source),gray(target)
    parameters=(None,config['pyr_scale'],config['levels'],config['winsize'],config['iterations'],
        config['poly_n'],config['poly_sigma'],cv2.OPTFLOW_FARNEBACK_GAUSSIAN if config['gaussian'] else 0)
    return (cv2.calcOpticalFlowFarneback(first,second,*parameters),
        cv2.calcOpticalFlowFarneback(second,first,*parameters))


def prepare_flow_warp(forward,backward,width,height,error_threshold=1.5):
    """Backward sampling avoids holes; check the round trip in flow-grid pixels."""
    forward,backward=np.asarray(forward),np.asarray(backward)
    if (forward.shape!=backward.shape or forward.ndim!=3 or forward.shape[2]!=2
            or min(forward.shape[:2])<2 or min(width,height)<2 or error_threshold<=0
            or not np.isfinite(forward).all() or not np.isfinite(backward).all()):
        raise ValueError('Invalid bidirectional optical flow or image dimensions')
    fh,fw=forward.shape[:2]
    sx,sy=width/fw,height/fh
    scale=np.array([sx,sy],dtype=np.float32)
    full_forward=cv2.resize(forward.astype(np.float32),(width,height),interpolation=cv2.INTER_LINEAR)*scale
    full_backward=cv2.resize(backward.astype(np.float32),(width,height),interpolation=cv2.INTER_LINEAR)*scale
    yy,xx=np.indices((height,width),dtype=np.float32)
    map_x,map_y=xx+full_backward[:,:,0],yy+full_backward[:,:,1]
    valid=(map_x>=0)&(map_x<=width-1)&(map_y>=0)&(map_y<=height-1)
    returned=cv2.remap(full_forward,map_x,map_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    error=np.linalg.norm((full_backward+returned)/scale,axis=2)
    return {'map_x':map_x,'map_y':map_y,'consistent':valid&(error<=error_threshold),
        'width':width,'height':height}


def warp_candidate(shape,warp,identifier,settings):
    """Reject an unreliable warp as a whole; retain every pixel of an accepted warp."""
    width,height=warp['width'],warp['height']
    if shape.get('label')!='pallet' or not valid_shape({**shape,'id':1},width,height):
        raise ValueError('Only valid pallet source candidates may be transported')
    score=float(shape.get('score',-1))
    if not math.isfinite(score) or not 0<=score<=1:
        raise ValueError('Invalid inherited source score')
    audit={'accepted':False,'inherited_source_score':score,'score_is_target_confidence':False}
    if score<settings['minimum_inherited_source_score']:
        return None,{**audit,'reason':'source_score_below_frozen_threshold'}
    original=_rasterize(shape,width,height)
    area=int(original.sum())
    if not area:return None,{**audit,'reason':'empty_source_raster'}
    transported=cv2.remap(original,warp['map_x'],warp['map_y'],cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    current=int(transported.sum())
    audit.update(source_pixels=area,transported_pixels=current,area_ratio=current/area)
    if current<settings['minimum_area_pixels_full_resolution']:
        return None,{**audit,'reason':'insufficient_transported_area'}
    low,high=settings['area_ratio_bounds']
    if not low<=current/area<=high:return None,{**audit,'reason':'area_change_outside_frozen_bounds'}
    consistency=int(np.count_nonzero((transported!=0)&warp['consistent']))/current
    audit['forward_backward_consistent_foreground_fraction']=consistency
    if consistency<settings['minimum_consistent_foreground_fraction']:
        return None,{**audit,'reason':'forward_backward_inconsistent'}
    candidate=_shape_from_mask(transported,shape['label'],identifier)
    if candidate is None or not valid_shape({**candidate,'id':1},width,height):
        return None,{**audit,'reason':'invalid_polygon_export'}
    actual=_rasterize(candidate,width,height)
    union=int(np.count_nonzero(actual|transported))
    iou=int(np.count_nonzero(actual&transported))/union if union else 0.
    audit['polygon_raster_iou']=iou
    if iou<settings['minimum_polygon_raster_IoU']:
        return None,{**audit,'reason':'polygon_export_changed_warp_too_much'}
    candidate.update(score=score,source='experimental_temporal_SAM3_warp_unconfirmed',
        temporal_source_score_inherited=True)
    return candidate,{**audit,'accepted':True,'reason':None}


def exclude_predicted_person(candidate,person_mask,settings):
    """Clip only a new temporal proposal using actual target person predictions."""
    from copy import deepcopy
    person_mask=np.asarray(person_mask,dtype=bool)
    if person_mask.ndim!=2:raise ValueError('Expected a 2D predicted person union')
    radius=settings.get('dilation_radius_pixels',0)
    if type(radius) is not int or not 0<=radius<=32:raise ValueError('Invalid predicted-person protection radius')
    if radius:
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
        person_mask=cv2.dilate(person_mask.astype(np.uint8),kernel).astype(bool)
    height,width=person_mask.shape
    if candidate.get('label')!='pallet' or not valid_shape({**candidate,'id':1},width,height):
        raise ValueError('Expected a valid new pallet proposal')
    original=_rasterize(candidate,width,height)
    area=int(original.sum())
    if not area:raise ValueError('Cannot clip an empty temporal proposal')
    removed=int(np.count_nonzero((original!=0)&person_mask))
    audit={'accepted':True,'removed_predicted_person_pixels':removed,'person_predictions_are_truth':False}
    if not removed:return deepcopy(candidate),audit
    retained=original.copy()
    retained[person_mask]=0
    count=int(retained.sum())
    audit.update(retained_fraction=count/area,retained_pixels=count)
    if count<settings['minimum_retained_pixels'] or count/area<settings['minimum_retained_fraction']:
        return None,{**audit,'accepted':False,'reason':'predicted_person_occlusion_removed_too_much'}
    exported=_shape_from_mask(retained,candidate['label'],candidate['id'])
    if exported is None or not valid_shape({**exported,'id':1},width,height):
        return None,{**audit,'accepted':False,'reason':'invalid_person_excluded_polygon'}
    actual=_rasterize(exported,width,height)
    union=int(np.count_nonzero(actual|retained))
    iou=int(np.count_nonzero(actual&retained))/union if union else 0.
    audit['polygon_raster_iou']=iou
    if iou<settings['minimum_polygon_raster_IoU'] or np.any((actual!=0)&person_mask):
        return None,{**audit,'accepted':False,'reason':'person_excluded_polygon_did_not_preserve_protection'}
    result={**deepcopy(candidate),**exported,'predicted_person_occlusion_guarded':True}
    return result,audit


def restrict_to_semantic_support(candidate,support,settings):
    """A refinement may join existing class evidence but cannot add class pixels."""
    from copy import deepcopy
    support=np.asarray(support,dtype=bool)
    if support.ndim!=2:raise ValueError('Expected a 2D target semantic support mask')
    height,width=support.shape
    if candidate.get('label')!='pallet' or not valid_shape({**candidate,'id':1},width,height):
        raise ValueError('Invalid temporal pallet proposal')
    original=_rasterize(candidate,width,height)
    area=int(original.sum())
    if not area:raise ValueError('Empty temporal proposal')
    retained=(original.astype(bool)&support).astype(np.uint8)
    count=int(retained.sum())
    audit={'accepted':False,'retained_pixels':count,'retained_fraction':count/area,'semantic_support_is_truth':False}
    if count<settings['minimum_retained_pixels'] or count/area<settings['minimum_retained_fraction']:
        return None,{**audit,'reason':'insufficient_target_semantic_support'}
    exported=_shape_from_mask(retained,candidate['label'],candidate['id'])
    if exported is None or not valid_shape({**exported,'id':1},width,height):
        return None,{**audit,'reason':'invalid_semantically_constrained_polygon'}
    actual=_rasterize(exported,width,height)
    total=int(np.count_nonzero(actual|retained))
    iou=int(np.count_nonzero(actual&retained))/total if total else 0.
    audit['polygon_raster_iou']=iou
    if iou<settings['minimum_polygon_raster_IoU'] or np.any((actual!=0)&~support):
        return None,{**audit,'reason':'export_exceeds_existing_semantic_support'}
    return {**deepcopy(candidate),**exported,'target_semantic_support_required':True}, {**audit,'accepted':True,'reason':None}


def augment_preserving_tiled(original,tiled,proposals,width,height,settings):
    """Preserve all existing tile masks, adding within the original total budget."""
    from copy import deepcopy
    from tiled_candidate_utils import augment_candidates
    if tiled[:len(original)]!=original or any(value['label']!='pallet' for value in tiled[len(original):]):
        raise ValueError('Tiled prefix or added class differs from the original baseline')
    source_count=sum(value['label']=='pallet' for value in original)
    limit=max(1,math.floor(source_count*settings['addition_ratio']))
    used=len(tiled)-len(original)
    if used>limit:raise ValueError('Existing tiled candidates exceed the original budget')
    remaining=limit-used
    if remaining:
        tiled_count=sum(value['label']=='pallet' for value in tiled)
        ratio=min(1.,math.nextafter(remaining/tiled_count,math.inf)) if tiled_count else 1.
        result,_=augment_candidates(tiled,proposals,width,height,minimum_score=settings['minimum_score'],
            addition_ratio=ratio,nms_iou=settings['nms_iou'])
    else:result=deepcopy(tiled)
    if result[:len(tiled)]!=tiled or len(result)-len(original)>limit:
        raise AssertionError('Preserved tile prefix or original total budget violated')
    return result,{'baseline_pallets':source_count,'addition_budget':limit,
        'added_pallets':len(result)-len(original),'temporal_candidates_added':len(result)-len(tiled),
        'existing_tiled_candidates_preserved':True,'baseline_candidates_preserved':True}
