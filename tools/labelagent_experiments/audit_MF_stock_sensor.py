"""Independently bind sensor-component prompts and untouched native capture pixels."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from run_MF_stock_sensor_SAM import OUT,SOURCE
from run_rgb_layers import read,save
from deliver_context_graph import sha
from native_review_feedback import describe_native,SOURCE_SHA
import audit_MF_first_parts as original
import unlabelled_height_bands as height

def main():
    if (OUT/'independent_readback.json').exists():raise FileExistsError('Preserve complete sensor audit')
    p=read(OUT/'frozen_plan.json');source=read(SOURCE/'completed.json');sp=read(SOURCE/'frozen_plan.json');g=read(OUT/'generation_completed.json');assert read(OUT/'semantic_source_selection.json')['selected_before_support_area_audit'];assert all(sha(f)==h for f,h in p['protected'].items());assert all(sha(f)==h for f,h in sp['protected'].items());assert set(r['relative'] for r in source['sensor_files_read'])==set(sp['allowed_sensor_files']);assert all(r['sha256']==sp['allowed_sensor_files'][r['relative']] for r in source['sensor_files_read']);original.OUT=OUT;arms={a:[] for a in ['sensor_positive_only','sensor_positive_and_other_height']};checks=[]
    for r in g['rows']:
        sensor=np.load(SOURCE/(r['teacher_key']+'_sensor_components.npz'),allow_pickle=False);uv=sensor['pixels'];component=sensor['component'];own=(component==r['component']);assert int(own.sum())==r['observed_component_count']
        # An EM component mean is a responsibility-weighted parameter, not
        # the mean of its later hard assignments. Replay the actual fitted
        # parameter and every assignment instead of loosening a tolerance.
        diag=next(x for x in source['diagnostics'] if x['teacher_key']==r['teacher_key']);fit=height.mixture(sensor['xyz'][:,2],diag['chosen_k']);np.testing.assert_array_equal(fit['labels'],component);assert abs(fit['means'][r['component']]-r['unlabelled_component_mean_z'])<1e-10
        center=np.median(uv[own],axis=0);chosen=uv[own][np.argmin(((uv[own]-center)**2).sum(1))];np.testing.assert_array_equal(chosen,r['source_point']);negative=np.array(r['hypothesized_negative_source_point']);assert ((uv[component!=r['component']]==negative).all(1)).any() and ((chosen-negative)**2).sum()>=64
        body=read(r['request_file']);assert hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()==r['request_sha256'];reply=read(Path(r['folder'])/'response.json')['raw_response'];meta=read(r['metadata_file']);assert meta==reply['raw_capture']['metadata'] and meta['source_sha256']==SOURCE_SHA and meta['image_sha256']==sha(r['input']) and meta['prior_mask_sha256'] is None;arrays=np.load(r['raw_array_file'],allow_pickle=False);call=r['prediction_call'];np.testing.assert_array_equal(arrays[f'{call}_point_coords'],[[q['x'],q['y']] for q in r['points']]);np.testing.assert_array_equal(arrays[f'{call}_point_labels'],[q['label'] for q in r['points']]);assert hashlib.sha256(arrays[f'{call}_masks'].tobytes()).hexdigest()==meta['predictions'][call]['masks']['sha256'];m=(arrays[f'{call}_masks'][r['native_index']].squeeze()!=0).astype('uint8');np.testing.assert_array_equal(m,np.load(Path(r['folder'])/'complete_native_mask.npy',allow_pickle=False));fb=describe_native(m,r['points']);assert fb==r['native_feedback'];x1,y1,x2,y2=r['observed_rectangle'];vx,vy,_,_=r['virtual_box'];im=np.asarray(Image.open(r['input']).convert('RGB'));photo=np.asarray(Image.open(r['source_image']).convert('RGB'));np.testing.assert_array_equal(im[y1:y2,x1:x2],photo[y1+vy:y2+vy,x1+vx:x2+vx]);outside=m.copy();outside[y1:y2,x1:x2]=0;pad=int(outside.sum());assert pad==r['foreground_on_synthetic_padding'];assert r['valid']==(meta['status']=='segment_returned' and not reply.get('warnings') and fb['native_prompt_passed'] and pad==0)
        if pad:assert not (Path(r['folder'])/'complete_full_mask.npy').exists()
        else:
            full=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False);np.testing.assert_array_equal(full[y1+vy:y2+vy,x1+vx:x2+vx],m[y1:y2,x1:x2]);assert int(full.sum())==int(m.sum());arms[r['arm']].append(original.evaluation_row(r))
        checks.append({'index':r['index'],'arm':r['arm'],'sensor_point_and_actual_request_and_all_native_FG_exact':True,'padding_FG':pad,'native_feedback':fb['failures'],'original_service_warnings':reply.get('warnings',[]),'original_service_status':meta['status'],'valid':r['valid']})
    scores=[original.score_arm(arm,rows) for arm,rows in arms.items()];save(OUT/'audit_summary.json',{'arms':scores,'reference_only_used_after_source_and_semantic_selection_freeze':True,'new_model_calls':0,'new_Agent_runs':0,'adopted':False});save(OUT/'independent_readback.json',{'passed':True,'checks':checks,'whitelist_sensor_files':source['sensor_files_read'],'no_GT_coordinates_for_generation':True,'no_original_annotation_or_output_mask_modified':True});print({'source_captures_checked':len(checks),'native_valid':sum(r['valid'] for r in checks),'service_without_warnings_but_rejected_native':sum(not r['valid'] and not r['original_service_warnings'] for r in checks),'arms':len(scores)},flush=True)

if __name__=='__main__':main()
