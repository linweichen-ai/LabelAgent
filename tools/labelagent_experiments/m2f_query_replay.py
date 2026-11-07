"""Replay frozen logits, preserving native masks apart from legacy fusion."""
import numpy as np
import torch
import torch.nn.functional as F


def probability(mask_logits,index,size):
    logits=torch.as_tensor(mask_logits[index],dtype=torch.float32)[None,None]
    return F.interpolate(logits,size=size,mode='bilinear',align_corners=False)[0,0].sigmoid().numpy()


def replay(class_logits,mask_logits,size,parameters):
    classes=torch.as_tensor(class_logits,dtype=torch.float32);probs=classes.softmax(-1);scores,labels=probs.max(-1)
    if parameters['promote_null_pallet']:raise ValueError('This audit cannot promote null queries')
    keep=(labels!=7)&(scores>torch.where(labels==0,scores.new_tensor(parameters['pallet_score_threshold']),scores.new_tensor(.5)))
    best=np.full(size,-np.inf,np.float32);winner=np.full(size,-1,np.int16);rows=[]
    for q in np.flatnonzero(keep.numpy()):
        weighted=probability(mask_logits,int(q),size)*np.float32(scores[q]);better=weighted>best;best[better]=weighted[better];winner[better]=q
    ids=np.zeros(size,np.uint32);metadata=[];stuff={};counter=0
    for q in range(len(scores)):
        row={'query':q,'top_class_id':int(labels[q]),'top_score':float(scores[q]),'p_pallet':float(probs[q,0]),'p_null':float(probs[q,7]),'class_gate':bool(keep[q]),'original_class_gate_reason':'PAL_kept' if labels[q]==0 and keep[q] else 'other_class_kept' if keep[q] else 'argmax_null' if labels[q]==7 else 'below_score_threshold'}
        if keep[q]:
            native=probability(mask_logits,q,size)>=.5;assigned=winner==q;selected=native&assigned;oa=int(native.sum());aa=int(assigned.sum());sa=int(selected.sum());threshold=parameters['pallet_overlap_threshold'] if labels[q]==0 else .8
            accepted=bool(oa and sa and aa/oa>=threshold);row.update(native_pixels=oa,legacy_assigned_pixels=aa,legacy_selected_pixels=sa,legacy_overlap_ratio=aa/oa if oa else 0,postprocess_gate=accepted)
            if accepted:
                label=int(labels[q])
                if label in {2,3,6} and label in stuff:ids[selected]=stuff[label]
                else:
                    counter+=1;ids[selected]=counter;metadata.append({'id':counter,'category_id':label+1,'iscrowd':0,'score':float(scores[q])})
                    if label in {2,3,6}:stuff[label]=counter
        rows.append(row)
    segments=[]
    for s in metadata:
        yy,xx=np.where(ids==s['id']);segments.append(dict(s,area=int(len(yy)),bbox=[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)]))
    return ids,segments,rows,winner
