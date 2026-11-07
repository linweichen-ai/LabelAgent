"""Freeze RGB neighbor hypotheses on the same six unchanged teacher priors."""
import datetime,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import run_soft_prior_trial as base
from rgb_neighbor_policy import propose

ROOT=base.ROOT;PREVIOUS=base.OUT;OUT=ROOT/'outputs/labelagent_rgb_neighbors_20261108'


def prepare():
    if OUT.exists():raise FileExistsError('Preserve frozen study')
    import shutil,subprocess
    if shutil.disk_usage(ROOT).free<256*2**20:raise RuntimeError('Space guard')
    OUT.mkdir();subprocess.run(['compact.exe','/C','/Q',str(OUT)],check=True,capture_output=True)
    previous=base.read(PREVIOUS/'frozen_plan.json');rows=[];diagnostics=[];seen=set();protected=dict(previous['protected'])
    for p in previous['proposals']:
        if p['kind']!='real' or p['teacher_key'] in seen:continue
        seen.add(p['teacher_key']);photo=np.asarray(Image.open(p['input']).convert('RGB'));mask=np.asarray(Image.open(p['prior']).convert('L'))>127
        proposals,diag=propose(photo,mask);diagnostics.append({'teacher_key':p['teacher_key'],**diag})
        for q in proposals:
            points=q['points'];key=hashlib.sha256((base.sha(p['input'])+base.sha(p['prior'])+base.sha(p['logits'])+json.dumps(points,sort_keys=True)+'soft').encode()).hexdigest();rows.append({**p,**q,'pair':p['teacher_key']+'_band'+str(q['band']),'side':'band'+str(q['band']),'points':points,'key':key,'mode':'soft','arm':'RGB_neighbor_exclusion','baseline_point_policy':'single RGB seam with one opposite negative','same_teacher_source_and_crop_as_baseline':True})
    if len(rows)!=12:raise RuntimeError(f'Expected same two hypotheses for six teachers; got {len(rows)}, do not increase search budget')
    for kind in ['gray_control','white_control']:
        old=next(p for p in previous['proposals'] if p['kind']==kind);p=rows[0];assert Image.open(old['input']).size==Image.open(p['input']).size
        row={**p,'pair':p['pair']+'_'+kind,'input':old['input'],'kind':kind,'reference_support_not_applicable':True,'input_sha256':base.sha(old['input'])};row['key']=hashlib.sha256((row['input_sha256']+base.sha(row['prior'])+base.sha(row['logits'])+json.dumps(row['points'],sort_keys=True)+'soft').encode()).hexdigest();rows.append(row)
    protected[str(PREVIOUS/'frozen_plan.json')]=base.sha(PREVIOUS/'frozen_plan.json')
    base.save(OUT/'frozen_plan.json',{'at':datetime.datetime.now().astimezone().isoformat(),'driver_sha256':base.sha(Path(base.__file__)),'actual_wrapper_sha256':base.sha(__file__),'policy_sha256':base.sha(Path(__file__).with_name('rgb_neighbor_policy.py')),'proposals':rows,'teacher_diagnostics':diagnostics,'protected':protected,'no_reference_inputs':True,'max_requests':14,'real_hypotheses':12,'same_six_teachers_and_two_hypotheses_per_teacher':True,'point_limit':3,'cumulative_point_limit':32,'one_SAM_call_per_chain':True,'new_Agent_rounds':0,'Agent_round_limit':3,'no_post_mask_clipping':True,'all_final_gates_unchanged':True,'exploratory_followup_after_soft_prior_results':True,'changed_variable':'RGB-derived layer point policy; no isolated single-factor claim'})
    print({'frozen_real':12,'blank_controls':2,'max_calls':14,'max_points':max(len(p['points']) for p in rows)},flush=True)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:
        base.OUT=OUT;p=base.read(OUT/'frozen_plan.json');assert base.sha(__file__)==p['actual_wrapper_sha256'] and base.sha(Path(__file__).with_name('rgb_neighbor_policy.py'))==p['policy_sha256'];base.run()
