"""Append source pallet masks without replacing any original candidate."""
from copy import deepcopy
import hashlib,json,math,re
import numpy as np
from source.agent.edit_agent import valid_shape
from tools.tiled_candidate_utils import _mask_record


def raster_key(shape,width,height):
    if not valid_shape({**shape,'id':1},width,height):raise ValueError('Invalid candidate geometry')
    record=_mask_record(shape,width,height)
    return hashlib.sha256(json.dumps(record['bbox']).encode()+np.packbits(record['mask']!=0).tobytes()).hexdigest()


def fuse_image(original,sources,width,height):
    names=[name for name,_ in sources]
    if len(set(names))!=len(names) or any(not re.fullmatch(r'[a-z][a-z0-9_]{0,31}',name) for name in names):
        raise ValueError('Distinct stable source names required')
    identifiers=[s['id'] for s in original]
    if any(type(v) not in (str,int) for v in identifiers) or len(set(identifiers))!=len(identifiers):
        raise ValueError('Distinct original identifiers required')
    known={raster_key(s,width,height) for s in original if s['label']=='pallet'}
    values=deepcopy(original);used=set(identifiers);provenance=[];counts={}
    for name,shapes in sources:
        stats={'added':0,'duplicate':0,'low_score':0,'non_pallet':0};counts[name]=stats
        for index,shape in enumerate(shapes):
            if shape.get('label')!='pallet':stats['non_pallet']+=1;continue
            score=shape.get('score')
            if type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1:
                raise ValueError('Finite model score required')
            if score<.5:stats['low_score']+=1;continue
            digest=raster_key(shape,width,height)
            if digest in known:stats['duplicate']+=1;continue
            identifier=f'material_raw:{name}:{index}'
            if identifier in used:raise ValueError('Namespaced identifier collision')
            value=deepcopy(shape);value['id']=identifier
            values.append(value);used.add(identifier);known.add(digest);stats['added']+=1
            provenance.append({'pool_id':identifier,'source_name':name,'source_id':shape.get('id'),
                'source_position':index,'raster_sha256':digest,
                'source_shape_sha256':hashlib.sha256(json.dumps(shape,sort_keys=True,separators=(',',':')).encode()).hexdigest()})
    if values[:len(original)]!=original:raise ValueError('Original prefix changed')
    return values,provenance,counts
