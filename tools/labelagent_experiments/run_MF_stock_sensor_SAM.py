"""Paired unlabelled-height prompts on complete original MF source inputs."""
import base64,hashlib,json,sys
from pathlib import Path
from MF_stock_sensor_parts import OUT as SOURCE
from prepare_palette_parts import BASE
from run_rgb_layers import read,save
from deliver_context_graph import sha
import run_MF_first_parts as driver
OUT=SOURCE/'SAM'

def prepare():
    if OUT.exists():raise FileExistsError('Preserve sensor SAM protocol')
    s=read(SOURCE/'completed.json');plan=read(SOURCE/'frozen_plan.json');assert all(sha(f)==h for f,h in plan['protected'].items());grid=read(BASE/'labelagent_registered_fixed_grid_20261108/generation_completed.json');OUT.mkdir();protected={str(SOURCE/'completed.json'):sha(SOURCE/'completed.json'),str(SOURCE/'frozen_plan.json'):sha(SOURCE/'frozen_plan.json'),str(Path(__file__)):sha(__file__)};rows=[];old={r['request_sha256'] for r in read(BASE/'labelagent_MF_first_parts_20261108/generation_completed.json')['rows']}
    for r in s['rows']:
        t=next(x for x in grid['rows'] if x['model']=='R15_PAL_candidate' and x['teacher_key']==r['original_teacher_key']);assert len(r['points'])==2
        for arm,points in [('sensor_positive_only',r['points'][:1]),('sensor_positive_and_other_height',r['points'])]:
            body={'image':base64.b64encode(Path(r['input']).read_bytes()).decode(),'points':points,'include_raw_capture':True};key=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest();assert key not in old,'Bind an exact prior capture instead of issuing this request';f=OUT/'prepared_requests'/(key+'_PRIVATE.json');save(f,body);protected[str(f)]=sha(f);protected[r['input']]=sha(r['input']);protected[r['source_image']]=sha(r['source_image']);rows.append(r|{'arm':arm,'input_sha256':sha(r['input']),'points':points,'virtual_box':r['box'],'observed_rectangle':t['observed_region_in_input'],'request_file':str(f),'request_sha256':key,'original_neutral_point_PAL_evidence':None,'point_caption_not_previously_run':True,'source_sensor_only_not_ground_truth':True})
    assert len(rows)==32;save(OUT/'frozen_plan.json',{'rows':rows,'protected':protected,'all_final_gates_unchanged':True,'source_points_frozen_before_reference_scoring':True,'source_whitelist_files_verified':s['sensor_files_read'],'sensor_input_differs_from_RGB_only_baseline':True,'negative_component_not_confirmed_separate_instance':True,'no_fake_neutral_subject_pass':True,'Agent_round_limit':3,'new_Agent_runs':0,'new_MF_calls':0,'model_config_sha256':read(BASE/'labelagent_MF_first_parts_20261108/frozen_plan.json')['model_config_sha256'],'adopted':False});print({'source_positive_points':16,'paired_SAM_requests':32,'new_model_calls':0},flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:driver.OUT=OUT;driver.main()
