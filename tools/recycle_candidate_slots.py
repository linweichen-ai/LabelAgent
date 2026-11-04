"""Recycle only unmatchable additions with a complete raw-point certificate."""
import numpy as np

RADII=(0,3,6)


def certified_redundant_additions(shapes,hits,protected_ids):
    """Keep protected shapes; remove <3-point additions only if all unions stay."""
    if len(shapes)!=len(hits) or len({s['id'] for s in shapes})!=len(shapes):
        raise ValueError('Distinct shapes with aligned hit records required')
    size=None
    for record in hits:
        if set(record)!=set(RADII):raise ValueError('All three raw radii required')
        for values in record.values():
            if not isinstance(values,np.ndarray) or values.dtype!=np.bool_ or values.ndim!=1:
                raise ValueError('One-dimensional boolean raw hits required')
            if size is None:size=len(values)
            if len(values)!=size:raise ValueError('Aligned raw points required')
    totals={r:sum((h[r].astype(np.int32) for h in hits),np.zeros(size or 0,np.int32)) for r in RADII}
    original={r:totals[r]>0 for r in RADII};removed=[]
    for i in sorted(range(len(shapes)),key=lambda i:str(shapes[i]['id'])):
        if shapes[i]['id'] in protected_ids or shapes[i]['label']!='pallet' or hits[i][3].sum()>=3:continue
        if any(np.any(hits[i][r]&(totals[r]<=1)) for r in RADII):continue
        removed.append(i)
        for r in RADII:totals[r]-=hits[i][r]
    if any(not np.array_equal(totals[r]>0,original[r]) for r in RADII):
        raise ValueError('Raw union certificate failed')
    kept=[i for i in range(len(shapes)) if i not in set(removed)]
    return kept,removed
