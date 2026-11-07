"""Keep original point identities through every edge and the whole RGB cycle."""
import numpy as np
from audit_pallet_temporal_views import track, PARAMETERS


def chain(cv2, grays, path, initial_points):
    initial=np.asarray(initial_points,dtype=np.float32).reshape(-1,2)
    alive={i:xy.tolist() for i,xy in enumerate(initial)};forward=[];backward=[]
    def edge(source,target,current,storage):
        if not current:
            storage.append({'source_frame':source,'target_frame':target,'input_point_ids':[],'track':None});return {}
        ids=sorted(current);result=track(cv2,np,grays[source],grays[target],[current[i] for i in ids]);storage.append({'source_frame':source,'target_frame':target,'input_point_ids':ids,'track':result})
        return {ids[row['point_id']]:row['target_xy'] for row in result['points'] if row['valid_image_correspondence']}
    for source,target in zip(path,path[1:]):alive=edge(source,target,alive,forward)
    endpoints=dict(alive);returned=alive
    for source,target in zip(path[::-1],path[-2::-1]):returned=edge(source,target,returned,backward)
    rows=[];good=[];limit=PARAMETERS['valid_correspondence']['max_forward_backward_error_px']
    for i,xy in enumerate(initial):
        back=returned.get(i);error=float(np.linalg.norm(np.asarray(back)-xy)) if back is not None else None;valid=error is not None and error<=limit
        if valid:good.append(i)
        rows.append({'point_id':i,'source_xy':xy.tolist(),'endpoint_xy':endpoints.get(i),'returned_xy':back,'whole_cycle_error_px':error,'all_edges_and_whole_cycle_valid':bool(valid)})
    goodxy=[endpoints[i] for i in good];delta=np.median(np.asarray(goodxy)-initial[good],axis=0).tolist() if good else None;n=len(initial);supported=n>0 and len(good)>=PARAMETERS['diagnostic_support']['min_valid_points'] and len(good)/n>=PARAMETERS['diagnostic_support']['min_valid_fraction']
    return {'path':path,'initial_points':n,'forward_survivors':len(endpoints),'valid_points':len(good),'valid_fraction':len(good)/n if n else 0.,'anchor_valid':0 in good,'diagnostic_image_correspondence_available':bool(supported),'median_valid_translation_xy':delta,'per_edge_and_whole_cycle_limit_px':limit,'forward_edges':forward,'backward_edges':backward,'points':rows,'object_identity_confirmed':False,'more_complete_visible_base_confirmed':False,'annotation_mask_generated':False}
