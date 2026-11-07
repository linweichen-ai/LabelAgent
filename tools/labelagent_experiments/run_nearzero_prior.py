"""Paired continuous-prior gain ablation; only one source-wide scalar changes."""
import hashlib,json,os,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image
import run_soft_prior_trial as base

ROOT=base.ROOT;PARENT=base.OUT
OUT=Path(os.environ.get('LABELAGENT_NEARZERO_OUTPUT',str(ROOT/'outputs/labelagent_nearzero_prior_20261108')))


def prepare():
    if OUT.exists():raise FileExistsError('Preserve immutable amplitude trial')
    if shutil.disk_usage(OUT.parent).free<256*2**20:raise RuntimeError('Output volume space guard')
    parent=base.read(PARENT/'frozen_plan.json');assert all(base.sha(p)==h for p,h in parent['protected'].items());OUT.mkdir();rows=[];bindings=[];protected=dict(parent['protected']);inputs={}
    for p in parent['proposals']:
        if p['mode']!='soft':continue
        if p['teacher_key'] not in inputs:
            original=np.load(p['logits'],allow_pickle=False);assert original.dtype==np.float32 and np.isfinite(original).all();bound=float(np.abs(original).max());assert bound>0
            # Chosen before output/reference evaluation. No per-pixel reshaping.
            gain=1e-6/bound;values=(original*gain).astype('float32');assert np.array_equal(values>=0,original>=0);assert np.array_equal(values>=0,np.asarray(Image.open(p['prior']).convert('L'))>127)
            file=OUT/'inputs'/(p['teacher_key']+'_scaled_logits.npy');file.parent.mkdir(exist_ok=True);np.save(file,values);inputs[p['teacher_key']]=(file,gain)
            bindings.append({'teacher_key':p['teacher_key'],'original_source_logits':p['logits'],'original_source_logits_sha256':base.sha(p['logits']),'scaled_source_logits':str(file),'scaled_source_logits_sha256':base.sha(file),'gain':gain,'bound_before':bound,'bound_after':float(np.abs(values).max()),'source_foreground_identical':True,'only_one_global_scalar':True})
        file,gain=inputs[p['teacher_key']];key=hashlib.sha256((base.sha(p['input'])+base.sha(p['prior'])+base.sha(file)+json.dumps(p['points'],sort_keys=True)+'soft').encode()).hexdigest();rows.append({**p,'key':key,'logits':str(file),'source_logit_file_sha256':base.sha(file),'original_teacher_logits':p['logits'],'arm':'nearzero_scaled_continuous_prior','gain':gain,'not_absent_mask_embedding':True})
    assert len(rows)==14 and sum(p['kind']=='real' for p in rows)==12
    protected[str(PARENT/'frozen_plan.json')]=base.sha(PARENT/'frozen_plan.json');base.save(OUT/'frozen_plan.json',{'driver_sha256':base.sha(Path(base.__file__)),'actual_wrapper_sha256':base.sha(__file__),'proposals':rows,'protected':protected,'gain_bindings':bindings,'single_changed_variable':'multiply entire continuous source logit array by fixed 1e-6/max_abs(source_logits)','source_foreground_geometry_unchanged':True,'same_RGB_points_SAM_weights_precision_single_mask_path':True,'no_reference_inputs':True,'bound_fixed_before_output':1e-6,'one_SAM_call_per_chain':True,'max_requests':14,'same_six_teachers_two_real_hypotheses_each':True,'new_Agent_rounds':0,'Agent_round_limit':3,'all_final_gates_unchanged':True,'not_equivalent_to_absent_prior_embedding':True,'no_post_mask_clipping':True,'exploratory_followup_after_prior_and_neighbor_failures':True});print({'real_pairs':12,'controls':2,'max_requests':14,'output':str(OUT)},flush=True)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:
        p=base.read(OUT/'frozen_plan.json');assert base.sha(__file__)==p['actual_wrapper_sha256'];base.OUT=OUT;base.run()
