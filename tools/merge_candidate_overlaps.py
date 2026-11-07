"""Experimental geometric merging of added masks; never writes original data."""
from copy import deepcopy

import numpy as np

from source.agent.edit_agent import _rasterize, _shape_from_mask, valid_shape
from tiled_candidate_utils import _mask_record, _iou


def merge_added_overlaps(baseline, additions, width, height, threshold=.7, label='pallet', verifier=None):
    """Merge high-IoU additions only when polygon export preserves the exact union.

    Original candidates keep their IDs and labels. Their output geometry may
    expand by an absorbed addition, while the input objects remain untouched.
    Two original candidates are never collapsed into each other. This is an
    image-only overlap hypothesis, not proof that two masks are one object.
    """
    if not .3 <= threshold <= 1:
        raise ValueError('Overlap threshold must be between .3 and 1')
    if verifier is not None and not callable(verifier):
        raise ValueError('Instance verifier must be callable')
    shapes=[*baseline,*additions]
    if len(shapes)>400 or any(not valid_shape({**shape,'id':index+1},width,height) for index,shape in enumerate(shapes)):
        raise ValueError('Invalid candidate geometry or count')
    if len({shape['id'] for shape in shapes})!=len(shapes):
        raise ValueError('Candidate IDs must be unique within an image')
    if any(shape['label']!=label for shape in additions):
        raise ValueError('Only the configured class may be added to this merger')
    output=deepcopy(baseline)
    records={index:_mask_record(shape,width,height) for index,shape in enumerate(output) if shape['label']==label}
    audit=[]
    # Keep model proposal ordering; an optional verifier can reject a union.
    for addition in additions:
        record=_mask_record(addition,width,height)
        matches=[(_iou(record,value),index) for index,value in records.items()]
        overlap,index=max(matches,default=(0.,-1))
        merged=False
        if overlap>=threshold:
            anchor=output[index]
            expected=_rasterize(anchor,width,height)|_rasterize(addition,width,height)
            candidate=_shape_from_mask(expected,anchor['label'],anchor['id'])
            if candidate is not None and valid_shape({**candidate,'id':index+1},width,height):
                actual=_rasterize(candidate,width,height)
                changed=int(np.count_nonzero(actual!=expected))
            else:
                changed=None
            verification=None
            if changed==0 and verifier is not None:
                verification=verifier(deepcopy(anchor),deepcopy(addition),deepcopy(candidate))
                if not isinstance(verification,dict) or type(verification.get('allow_merge')) is not bool:
                    raise ValueError('Verifier must return an explicit boolean allow_merge')
            permitted=verification is None or verification['allow_merge']
            if changed==0 and permitted:
                replacement={**deepcopy(anchor),**candidate,'source':'experimental_geometric_union_unconfirmed'}
                output[index]=replacement
                records[index]=_mask_record(replacement,width,height)
                merged=True
            audit.append({'anchor_id':anchor['id'],'addition_id':addition['id'],'mask_iou':overlap,
                'merged':merged,'union_raster_different_pixels':changed,
                'verification':verification,
                'fallback_reason':None if merged else 'instance_verifier_rejected' if not permitted
                    else 'polygon_export_would_change_union_or_be_invalid'})
        if not merged:
            output.append(deepcopy(addition))
            records[len(output)-1]=record
    for value in baseline:
        if value['label']!=label:
            actual=next(shape for shape in output if shape['id']==value['id'])
            if actual!=value:raise AssertionError('Other-class candidate changed')
    def union(values):
        canvas=np.zeros((height,width),dtype=np.uint8)
        for value in values:
            if value['label']==label:canvas|=_rasterize(value,width,height)
        return canvas
    if not np.array_equal(union(shapes),union(output)):
        raise AssertionError('Merge changed the class union')
    if not {shape['id'] for shape in baseline}.issubset({shape['id'] for shape in output}):
        raise AssertionError('An original candidate ID disappeared')
    return output,{'input_candidates':len(shapes),'output_candidates':len(output),
        'merged_additions':sum(row['merged'] for row in audit),'exact_class_union_preserved':True,
        'baseline_input_objects_modified':False,'original_ids_preserved':True,'audit':audit}
