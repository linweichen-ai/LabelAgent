"""Same source photographs/model/schema; remove descriptive prompt boilerplate."""
import copy,datetime,hashlib,json,shutil,sys
from pathlib import Path
from prepare_balanced_palette_parts import OUT as SOURCE
from prepare_palette_parts import BASE,ROOT
from run_rgb_layers import read,save
from deliver_context_graph import sha
import run_fixed_grid_followup as api
from semantic_subject_guard import compatible_pallet_subject
import prepare_fixed_grid_followup as views
from PIL import Image
OUT=BASE/'labelagent_neutral_prompt_pair_20261108'
PROMPT=('Describe the visible physical object or surface containing the exact center pixel of these two unmarked photographs. '
        'Both photographs have the same source center. Use visible connected structure in the wider photograph to identify that component, '
        'and do not substitute an adjacent object for the center component. Give its material, structure and color in one short English sentence. '
        'If there is no identifiable object or the image is uniform, describe the actual visible surface or uniform image. '
        'Do not infer a category from a task or label list. Return exactly one JSON key object.')

def key(p):return hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()

def prepare():
    if OUT.exists():raise FileExistsError('Preserve paired prompt protocol')
    if shutil.disk_usage(BASE).free<96*2**20:raise RuntimeError('Original96MiB preparation reserve')
    p=read(SOURCE/'frozen_plan.json');old=read(SOURCE/'review_completed.json');assert all(sha(f)==h for f,h in p['protected'].items());OUT.mkdir();views.OUT=OUT;cases=[];protected={str(SOURCE/'frozen_plan.json'):sha(SOURCE/'frozen_plan.json'),str(SOURCE/'review_completed.json'):sha(SOURCE/'review_completed.json'),str(Path(__file__)):sha(__file__),str(Path(__file__).with_name('paired_source_cue.py')):sha(Path(__file__).with_name('paired_source_cue.py'))}
    for name in ['wood_pallet_base','raw_robot','raw_person','raw_floor','gray','white']:
        file=ROOT/'outputs/labelagent_semantic_recovery_20261105/neutral_controls'/(name+'.png');photo=Image.open(file).convert('RGB');protected[str(file)]=sha(file);bindings=views.caption_views(photo,[photo.width//2,photo.height//2],OUT/'controls'/name);cases.append(bindings|{'index':'control_'+name,'kind':'control','expected_PAL':name=='wood_pallet_base','expected_uniform':name in ['gray','white'],'derived_control_not_native_sensor_scene':True})
    cases += [r|{'kind':'real_source_not_truth'} for r in p['rows']];requests=[];cache=[]
    for r in cases:
        original=api.body(*[b['image'] for b in r['bindings']]);new=copy.deepcopy(original);new['messages'][0]['content'][0]['text']=PROMPT;check=copy.deepcopy(new);check['messages'][0]['content'][0]['text']=original['messages'][0]['content'][0]['text'];assert check==original
        for arm,payload in [('original',original),('concise',new)]:
            k=key(payload);file=OUT/'prepared_requests'/(k+'_PRIVATE.json');save(file,payload);row=r|{'arm':arm,'request_file':str(file),'request_sha256':k};requests.append(row);protected[str(file)]=sha(file)
            for b in r['bindings']:protected[b['image']]=b['sha256']
            prior=next((v for v in old['rows'] if v['request_sha256']==k),None)
            if prior:
                source=Path(prior['response_file']);reply=read(source)['raw_response'];assert sha(source)==prior['response_sha256'] and read(prior['request_file'])==payload and reply['local_vlm']['model_config_sha256']==p['model_config_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256,256];dest=OUT/'responses'/(k+'.json');save(dest,{'raw_response':reply,'exact_prior_reuse':True});cache.append({'request_sha256':k,'source_response':str(source),'source_response_sha256':sha(source)})
    save(OUT/'frozen_plan.json',{'at':datetime.datetime.now().astimezone().isoformat(),'requests':requests,'protected':protected,'model_config_sha256':p['model_config_sha256'],'real_source_points':len(p['rows']),'control_cases':6,'only_prompt_text_changes_within_pair':True,'all_images_schema_tokens_weights_decoding_identical':True,'new_prompt_does_not_name_expected_category':True,'unchanged_primary_subject_parser':True,'all_original_final_mask_semantics_area_purity_matching_and_Agent_limits_unchanged':True,'proposal_diagnostic_not_instance_accuracy':True,'matched_control_admission_requires_all6_original_expected_control_labels_for_new_prompt':True,'controls_execute_before_any_new_real_point_review':True,'Agent_round_limit':3,'new_Agent_runs':0,'max_unique_new_calls':len({r['request_sha256'] for r in requests})-len(cache),'adopted':False});save(OUT/'exact_prior_cache.json',{'rows':cache});print({'logical_requests':len(requests),'source_points':len(p['rows']),'controls':6,'exact_prior_reuses':len(cache),'max_unique_new_VLM_calls':read(OUT/'frozen_plan.json')['max_unique_new_calls']},flush=True)

def run():
    if (OUT/'completed.json').exists():raise FileExistsError('Do not repeat completed paired prompt test')
    p=read(OUT/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());api.OUT=OUT;_,live=api.external_live()
    if live:raise RuntimeError('External named GPU run is live')
    rows=[];fresh=0;controls_ok=None
    for r in p['requests']:
        if r['kind']=='real_source_not_truth' and controls_ok is None:
            controls=[x for x in rows if x['kind']=='control' and x['arm']=='concise'];assert len(controls)==6;controls_ok=all(x['PAL_subject_evidence']==x['expected_PAL'] for x in controls)
            if not controls_ok:
                save(OUT/'control_failure.json',{'all_actual_control_replies_kept':True,'rows':controls,'real_new_calls_after_failed_control':0,'no_proposal_admission':True});break
        payload=read(r['request_file']);assert key(payload)==r['request_sha256'];file=OUT/'responses'/(r['request_sha256']+'.json')
        if not file.exists():
            if shutil.disk_usage(OUT).free<16*2**20:raise RuntimeError('Original16MiB reply reserve')
            response=api.request_json('http://127.0.0.1:18868/v1/chat/completions',payload);save(file,{'raw_response':response,'exact_prior_reuse':False});fresh+=1
        saved=read(file);reply=saved['raw_response'];assert reply['local_vlm']['model_config_sha256']==p['model_config_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256,256];decision=json.loads(reply['choices'][0]['message']['content']);row=r|{'decision':decision,'PAL_subject_evidence':compatible_pallet_subject(decision),'response_file':str(file),'response_sha256':sha(file),'exact_prior_reuse':saved['exact_prior_reuse']};rows.append(row);save(OUT/'progress.json',{'rows':rows,'new_calls_this_execution':fresh});print({'index':r['index'],'arm':r['arm'],'object':decision['object'],'PAL_subject_evidence':row['PAL_subject_evidence'],'prior_reuse':row['exact_prior_reuse']},flush=True)
    real=[r for r in rows if r['kind']=='real_source_not_truth'];pairs=[]
    for index in sorted({r['index'] for r in real}):
        rs=[r for r in real if r['index']==index];assert len(rs)==2;pairs.append({'index':index,'original_PAL':next(x['PAL_subject_evidence'] for x in rs if x['arm']=='original'),'concise_PAL':next(x['PAL_subject_evidence'] for x in rs if x['arm']=='concise'),'actual_center_object_truth_unknown':True})
    save(OUT/'completed.json',{'status':'complete_paired_neutral_prompt_diagnostic','rows':rows,'pairs':pairs,'concise_all6_controls_pass':bool(controls_ok),'actual_new_VLM_calls_total':len({r['request_sha256'] for r in rows if not r['exact_prior_reuse']}),'exact_prior_reuses':len({r['request_sha256'] for r in rows if r['exact_prior_reuse']}),'new_calls_this_execution':fresh,'new_SAM_calls':0,'new_Agent_runs':0,'no_real_point_accuracy_claim':True,'new_accepted_instances':0,'adopted':False});print({'controls_pass':bool(controls_ok),'completed_real_pairs':len(pairs),'new_calls':read(OUT/'completed.json')['actual_new_VLM_calls_total']},flush=True)

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
