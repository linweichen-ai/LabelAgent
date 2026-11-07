"""Verify both zoom stock maps instead of fabricating a legacy adapter flag."""
import datetime,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from run_rgb_layers import ROOT,read,save,sha
import run_trained_point_sam as sam
OUT=ROOT/'outputs/labelagent_zoom_cues_20261108'
Z=ROOT/'outputs/labelagent_zoom_teacher_20261108'

def main():
    if (OUT/'SAM').exists():raise FileExistsError('SAM plan already prepared; do not repeat')
    failure=OUT/'engineering_failures';failure.mkdir(exist_ok=True);adapter=OUT/'proposal_adapter/completed.json';previous=read(adapter);shutil.copy2(adapter,failure/'proposal_adapter_missing_consensus_flag_v1.json');save(failure/'legacy_adapter_field_error.json',{'at':datetime.datetime.now().astimezone().isoformat(),'error':'KeyError both_stock_panoptics_predict_PAL in legacy helper before any SAM request','new_SAM_calls_before_fix':0,'do_not_set_field_without_checking_original_maps':True});models=read(Z/'generation_completed.json')['rows'];unique={};evidence=[]
    for cue in previous['rows']:
        checks=[]
        for name in ['old','retained']:
            row=next(x for x in models if x['model']==name and x['case']==cue['case'] and x['box']==cue['box'] and sha(x['input'])==sha(cue['input']));file=Path(row['folder'])/'panoptic.png';assert sha(file)==row['artifacts']['panoptic.png'];a=np.asarray(Image.open(file)).astype('uint32');ids=a[:,:,0]+256*a[:,:,1]+65536*a[:,:,2];q=cue['points'][0];value=int(ids[q['y'],q['x']]);pal_ids={s['id'] for s in row['segments_info'] if s['category_id']==1};checks.append({'model':name,'panoptic':str(file),'sha256':sha(file),'source_relative_point':[q['x'],q['y']],'segment_id':value,'stock_predicts_PAL':value in pal_ids})
        good=all(c['stock_predicts_PAL'] for c in checks);evidence.append({'cue':cue['index'],'both_zoom_stock_maps_checked':checks,'admitted':good})
        if good:
            row={**cue,'both_stock_panoptics_predict_PAL':True,'legacy_flag_evidence_scope':'two exact zoom stock maps on same native source ROI; never assumed from model score','both_zoom_stock_map_evidence':checks};key=sam.key(row)
            if key not in unique:unique[key]=row
            else:unique[key]['logical_zoom_cues_sharing_exact_SAM_input']+=row['logical_zoom_cues_sharing_exact_SAM_input']
    before_bodies={sam.key(r) for r in previous['rows']};after_bodies={sam.key(r) for r in unique.values()};assert after_bodies<=before_bodies
    save(OUT/'consensus_adapter_frozen_plan.json',{'at':datetime.datetime.now().astimezone().isoformat(),'driver_sha256':sha(__file__),'original_neutral_review_sha256':sha(OUT/'review_completed.json'),'original_zoom_generation_sha256':sha(Z/'generation_completed.json'),'checks':evidence,'missing_metadata_fixed_by_readback_not_relaxation':True,'body_hashes_before':sorted(before_bodies),'body_hashes_after':sorted(after_bodies),'no_SAM_requests_before_this_source_check':True});save(adapter,{'rows':list(unique.values()),'source_admissions':previous['source_admissions'],'neutral_review_sha256':sha(OUT/'review_completed.json'),'consensus_evidence_sha256':sha(OUT/'consensus_adapter_frozen_plan.json'),'exact_same_input_deduplicated_before_network':True,'MF_native_prior_not_sent_in_point_only_arm':True});sam.OUT=OUT/'SAM';sam.REVIEW=OUT/'proposal_adapter';sam.prepare();print({'source_verified_unique_requests':len(unique),'dropped_for_map_disagreement':sum(not e['admitted'] for e in evidence),'new_SAM_calls':0},flush=True)

if __name__=='__main__':main()
