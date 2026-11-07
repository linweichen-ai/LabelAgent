"""Select point cues from unchanged trained-class agreement, never trim outputs."""
import numpy as np


def points(logits,parent,old_PAL,new_PAL):
    if not(logits.shape==parent.shape==old_PAL.shape==new_PAL.shape):raise ValueError('Point-map source geometry mismatch')
    yy,xx=np.where(parent!=0)
    if not len(xx):return [],{'reason':'Empty teacher parent'}
    eligible=(parent!=0)&(old_PAL!=0)&(new_PAL!=0)&np.isfinite(logits);mid=(int(yy.min())+int(yy.max()))//2;rows=[]
    for side,band in [('upper',np.indices(parent.shape)[0]<=mid),('lower',np.indices(parent.shape)[0]>mid)]:
        allowed=eligible&band
        if not allowed.any():continue
        score=np.where(allowed,logits,-np.inf);y,x=map(int,np.unravel_index(score.argmax(),score.shape));rows.append({'side':side,'points':[{'x':x,'y':y,'label':1}],'old_parent_logit':float(logits[y,x]),'both_stock_panoptics_predict_PAL':True,'point_filter_not_an_annotation_or_output_mask':True})
    return rows,{'eligible_point_pixels':int(eligible.sum()),'original_parent_pixels':int((parent!=0).sum()),'mid_source_row':mid,'max_hypotheses':2,'parent_and_all_output_masks_unchanged':True,'no_reference_inputs':True}
