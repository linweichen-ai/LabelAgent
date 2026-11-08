"""Offline Agent adapter: source-native failures cannot be hidden by polygons.

This adapter grants no final annotation acceptance and modifies no masks. The
original three-round limit and negative7x7 constraint remain unchanged.
"""
import base64,hashlib,io
import numpy as np
from native_prompt_guard import evaluate
SOURCE_SHA='ab8861c57fbd275e8d16b5cc9bfbf734b3e4d7226d02b93d2b0ef9173fc49a81'

def describe_native(mask,points):
    a=np.asarray(mask);guard=evaluate(a,points);border=bool(a[0].any() or a[-1].any() or a[:,0].any() or a[:,-1].any());failures=[]
    for r,p in zip(guard['points'],points):
        if r['valid']:continue
        failures.append({'point_index':r['index'],'label':p['label'],'source_point':[int(p['x']),int(p['y'])],'center_foreground':r['center_covered'],'foreground_pixels_in_required_region':r['pixels_in_required_region'],'required_negative_radius':3 if p['label']==0 else None,'reason':'negative7x7_contains_foreground' if p['label']==0 else 'positive_not_covered'})
    if border:failures.append({'reason':'native_mask_touches_input_border'})
    return {'native_prompt_passed':guard['valid'] and not border,'native_prompt_guard':guard,'touches_input_border':border,'failures':failures,'native_pixels':int((a!=0).sum()),'native_value_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'native_array_not_modified':True,'final_instance_acceptance_granted':False}

def review_response(response,points,round_index=1,max_rounds=3):
    if max_rounds!=3 or round_index not in [1,2,3]:raise ValueError('Original three-round contract required')
    capture=response.get('raw_capture',{});meta=capture.get('metadata',{});base={'round_index':round_index,'max_rounds':3,'remaining_rounds':3-round_index,'keep_current_annotation':True,'annotation_write_permitted':False,'new_model_calls':0,'response_warnings':response.get('warnings',[]),'final_instance_acceptance_granted':False}
    if meta.get('source_sha256')!=SOURCE_SHA or not meta.get('raw_capture_complete') or not meta.get('selected_index_observed'):
        return base|{'native_prompt_passed':False,'failure':'unverifiable_original_native_selection','next_action':'abstain_at_limit' if round_index==3 else 'provide_verifiable_native_capture_within_existing_budget'}
    raw=base64.b64decode(capture['arrays_npz_base64']);arrays=np.load(io.BytesIO(raw),allow_pickle=False);call=meta['selected_prediction_call'];all_masks=arrays[f'{call}_masks'];record=meta['predictions'][call]['masks']
    if hashlib.sha256(all_masks.tobytes()).hexdigest()!=record['sha256']:raise ValueError('Native capture digest mismatch')
    np.testing.assert_array_equal(arrays[f'{call}_point_coords'],[[p['x'],p['y']] for p in points]);np.testing.assert_array_equal(arrays[f'{call}_point_labels'],[p['label'] for p in points]);native=all_masks[meta['selected_index']].squeeze();result=describe_native(native,points);quiet_invalid=not result['native_prompt_passed'] and not response.get('warnings') and meta.get('status')=='segment_returned'
    messages=[]
    for failure in result['failures']:
        if failure['reason']=='negative7x7_contains_foreground':messages.append('Negative point '+str(failure['point_index'])+' has '+str(failure['foreground_pixels_in_required_region'])+' foreground pixels inside its required7x7 region; its center alone is insufficient.')
        elif failure['reason']=='positive_not_covered':messages.append('Positive point '+str(failure['point_index'])+' is not covered by the native mask.')
        else:messages.append('Native output touches the original input border.')
    if not result['native_prompt_passed']:messages.append('Keep current annotation. Use remaining existing operations for a new model prediction or abstain; do not cut the mask or relax constraints.')
    return base|result|{'source_status':meta.get('status'),'service_returned_without_warning_but_native_failed':quiet_invalid,'feedback_for_agent':messages,'next_action':'continue_unchanged_final_instance_checks' if result['native_prompt_passed'] else 'abstain_at_limit' if round_index==3 else 'review_native_failure_within_remaining_original_budget'}
