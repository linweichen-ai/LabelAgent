"""Class-specific fixed postprocessing for one Mask2Former output; no extra network."""
def postprocess(output,size,parameters):
    import torch,numpy as np
    import torch.nn.functional as F
    classes=output['class_queries_logits'][0];scores,labels=classes.softmax(-1).max(-1);scores=scores.clone();labels=labels.clone();threshold=parameters['pallet_score_threshold'];overlap=parameters['pallet_overlap_threshold']
    if parameters['promote_null_pallet']:
        pal=classes.softmax(-1)[:,0];promote=(labels==7)&(pal>threshold);labels[promote]=0;scores[promote]=pal[promote]
    keep=(labels!=7)&(scores>torch.where(labels==0,scores.new_tensor(threshold),scores.new_tensor(.5)));scores,labels=scores[keep],labels[keep];ids=torch.zeros(size,dtype=torch.int32,device=classes.device);stuff={};metadata=[];counter=0
    if len(scores):
        masks=F.interpolate(output['masks_queries_logits'][:,keep],size=size,mode='bilinear',align_corners=False)[0].sigmoid();winners=(masks*scores[:,None,None]).argmax(0)
        for i,(score,label) in enumerate(zip(scores,labels)):
            label=int(label);assigned=winners==i;original=masks[i]>=.5;selected=assigned&original;oa=int(original.sum());aa=int(assigned.sum())
            if not oa or not selected.any() or aa/oa<(overlap if label==0 else .8):continue
            if label in {2,3,6} and label in stuff:ids[selected]=stuff[label];continue
            counter+=1;ids[selected]=counter;metadata.append(dict(id=counter,category_id=label+1,iscrowd=0,score=float(score)))
            if label in {2,3,6}:stuff[label]=counter
    ids=ids.cpu().numpy().astype(np.uint32);segments=[]
    for s in metadata:
        yy,xx=np.where(ids==s['id']);segments.append(dict(s,area=int(len(yy)),bbox=[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)]))
    return ids,segments
