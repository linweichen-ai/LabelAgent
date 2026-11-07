"""Post-pilot RMS-matched binary input control; no output/reference feedback."""
import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image
import run_soft_prior_trial as base
ROOT=base.ROOT;PARENT=base.OUT;OUT=PARENT/'strength_control';base.OUT=OUT


def prepare():
    if OUT.exists():raise FileExistsError('Preserve intensity control')
    OUT.mkdir();parent=base.read(PARENT/'frozen_plan.json');generated=base.read(PARENT/'generation_completed.json');protected={str(PARENT/'frozen_plan.json'):base.sha(PARENT/'frozen_plan.json'),str(PARENT/'generation_completed.json'):base.sha(PARENT/'generation_completed.json')};rows=[];bindings=[]
    for p in parent['proposals']:
        if p['mode']!='soft':continue
        capture=next(r for r in generated['rows'] if r['key']==p['key']);arrays=np.load(capture['raw_array_file'],allow_pickle=False);soft=arrays['0_actual_mask_input'];hard=arrays['0_original_hard_mask_input'];rms=lambda a:float(np.sqrt(np.mean(a.astype('float64')**2)));scale=rms(soft)/rms(hard);binary=np.asarray(Image.open(p['prior']).convert('L'))>127;values=np.where(binary,10.*scale,-10.*scale).astype('float32');folder=OUT/'inputs'/p['pair'];folder.mkdir(parents=True);file=folder/'matched_binary_logits.npy';np.save(file,values);actual=cv2.resize(values,(hard.shape[2],hard.shape[1]),interpolation=cv2.INTER_LINEAR)[None];assert np.array_equal(values>=0,binary);assert abs(rms(actual)-rms(soft))<=max(1e-5,1e-6*rms(soft));key=hashlib.sha256((base.sha(p['input'])+base.sha(p['prior'])+base.sha(file)+json.dumps(p['points'],sort_keys=True)+'soft').encode()).hexdigest();rows.append({**p,'key':key,'logits':str(file),'source_logit_file_sha256':base.sha(file),'arm':'rms_matched_hard','mode':'soft','RMS_scale':scale});protected[capture['raw_array_file']]=base.sha(capture['raw_array_file']);bindings.append({'pair':p['pair'],'source_soft_input_rms':rms(soft),'legacy_hard_input_rms':rms(hard),'matched_binary_input_rms':rms(actual),'scalar_from_prior_inputs_only':scale,'no_reference_or_output_mask_used_for_scale':True})
    protected.update(parent['protected']);base.save(OUT/'frozen_plan.json',{'driver_sha256':base.sha(Path(base.__file__)),'actual_wrapper_sha256':base.sha(__file__),'proposals':rows,'protected':protected,'RMS_bindings':bindings,'exploratory_followup_after_pilot':True,'same_RGB_points_source_foreground':True,'scale_definition':'RMS(native_teacher_input)/RMS(original_hard_input), no score tuning','max_requests':len(rows),'one_SAM_call_per_chain':True,'new_Agent_runs':0,'Agent_round_limit':3,'all_final_gates_unchanged':True});print({'matched_binary_requests':len(rows),'real_pairs':sum(r['kind']=='real' for r in rows),'all_input_RMS_matched':True},flush=True)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:
        p=base.read(OUT/'frozen_plan.json');assert base.sha(__file__)==p['actual_wrapper_sha256'];base.run()
