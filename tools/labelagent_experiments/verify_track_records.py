"""Reconstruct saved correspondence decisions without trusting saved totals."""
import numpy as np


def edge_valid(row,width=1920,height=1536):
    f=row['target_xy'];b=row['back_to_source_xy']
    if not(row['forward_status'] and row['backward_status'] and f is not None and b is not None):return False
    if not(0<=f[0]<width and 0<=f[1]<height and 0<=b[0]<width and 0<=b[1]<height):return False
    error=float(np.linalg.norm(np.asarray(b,np.float32)-np.asarray(row['source_xy'],np.float32)))
    return error<=1.5


def verify(row,initial):
    initial=np.asarray(initial,np.float32);direct=row['direct'];valid=[]
    for p in direct['points']:
        np.testing.assert_array_equal(p['source_xy'],initial[p['point_id']]);passed=edge_valid(p);assert passed==p['valid_image_correspondence']
        if passed:valid.append(p['point_id'])
    assert len(valid)==direct['valid_points'] and direct['anchor_valid']==(0 in valid);assert direct['diagnostic_image_correspondence_available']==(len(valid)>=6 and len(valid)/len(initial)>=.6)
    linked=row['chain'];alive={i:p.tolist() for i,p in enumerate(initial)}
    for edge in linked['forward_edges']+linked['backward_edges']:
        ids=edge['input_point_ids'];assert ids==sorted(alive)
        if not ids:assert edge['track'] is None;continue
        result=edge['track'];assert len(result['points'])==len(ids);nextalive={}
        for p in result['points']:
            np.testing.assert_allclose(p['source_xy'],alive[ids[p['point_id']]],rtol=0,atol=1e-4);passed=edge_valid(p);assert passed==p['valid_image_correspondence']
            if passed:nextalive[ids[p['point_id']]]=p['target_xy']
        alive=nextalive
    valid=[]
    for p in linked['points']:
        np.testing.assert_array_equal(p['source_xy'],initial[p['point_id']]);back=alive.get(p['point_id']);passed=back is not None and np.linalg.norm(np.asarray(back)-initial[p['point_id']])<=1.5;assert passed==p['all_edges_and_whole_cycle_valid'];assert back==p['returned_xy']
        if passed:valid.append(p['point_id'])
    assert len(valid)==linked['valid_points'] and linked['anchor_valid']==(0 in valid);assert linked['diagnostic_image_correspondence_available']==(len(valid)>=6 and len(valid)/len(initial)>=.6)
    return {'seed':row['seed'],'frame':row['endpoint_frame'],'all_direct_edge_whole_cycle_decisions_recomputed':True,'source_point_ids_exact':True,'chain_valid':len(valid)}
