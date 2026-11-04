"""Recycle only additions with identical raw-hit columns at all three radii."""
import numpy as np


def certified_duplicate_additions(shapes,hits,protected_ids):
    if len(shapes)!=len(hits) or len({s['id'] for s in shapes})!=len(shapes):raise ValueError('Aligned distinct shapes required')
    size=None;groups={}
    for i,record in enumerate(hits):
        if set(record)!={0,3,6}:raise ValueError('All three radii required')
        for v in record.values():
            if not isinstance(v,np.ndarray) or v.dtype!=np.bool_ or v.ndim!=1:raise ValueError('Boolean raw hits required')
            if size is None:size=len(v)
            if len(v)!=size:raise ValueError('Aligned raw points required')
        if shapes[i]['label']=='pallet':
            signature=tuple(np.packbits(record[r]).tobytes() for r in (0,3,6))
            groups.setdefault(signature,[]).append(i)
    removed=[]
    for indices in groups.values():
        protected=[i for i in indices if shapes[i]['id'] in protected_ids]
        representatives=set(protected or [min(indices,key=lambda i:str(shapes[i]['id']))])
        removed.extend(i for i in indices if i not in representatives)
    return [i for i in range(len(shapes)) if i not in set(removed)],sorted(removed)
