"""Review every source-frozen native-valid SAM hypothesis with unchanged gates."""
import hashlib,json,shutil,sys
from pathlib import Path
from prepare_mf_first_parts import OUT as PRIMARY
from prepare_palette_parts import ROOT,BASE
from run_rgb_layers import read,save
from deliver_context_graph import sha
from MF_first_native_pool import POOL
import numpy as np
from PIL import Image
import review_box_layers as runner
from semantic_evidence_gate import CAPTION_PROMPT,CAPTION_SCHEMA
from target_identity_policy import centered_photos,extent_views,EXTENT_PROMPT,EXTENT_SCHEMA
from instance_identity_consistency import target_views,INSTANCE_PROMPT,INSTANCE_SCHEMA
S=POOL/'semantic'

def prepare():
    if S.exists():raise FileExistsError('Preserve complete or interrupted native-pool semantic plan')
    pool=read(POOL/'frozen_outputs.json');selected=[r for r in pool['rows'] if r['valid']];assert pool['all_source_native_valid_selected_before_reference_scoring'] and all(sha(f)==h for f,h in pool['protected'].items());S.mkdir();requests=[];protected={str(POOL/'frozen_outputs.json'):sha(POOL/'frozen_outputs.json'),str(Path(__file__)):sha(__file__)}
    for r in selected:
        file=Path(r['folder'])/'complete_full_mask.npy';mask=np.load(file,allow_pickle=False);assert hashlib.sha256(mask.tobytes()).hexdigest()==r['full_mask_sha256'];photo=Image.open(r['source_image']).convert('RGB');folder=S/r['key'];folder.mkdir();protected[str(file)]=sha(file);protected[r['source_image']]=sha(r['source_image'])
        def append(stage,name,images,prompt,schema,kind='candidate'):
            paths=[]
            for i,image in enumerate(images):
                f=folder/f'{stage}_{kind}_{name}_{i}.png';image.save(f);paths.append(str(f));protected[str(f)]=sha(f)
            payload=runner.body(paths,prompt,schema);digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();requests.append({'candidate':r['key'],'stage':stage,'name':name,'kind':kind,'images':paths,'payload':payload,'request_sha256':digest})
        for name,view in centered_photos(photo,mask)[0].items():
            append('neutral',name,[view],CAPTION_PROMPT,CAPTION_SCHEMA)
            for color,value in [('gray',127),('white',255)]:append('neutral',name,[Image.new('RGB',view.size,(value,)*3)],CAPTION_PROMPT,CAPTION_SCHEMA,color)
        for name,view in target_views(photo,mask)[0].items():append('structure',name,[view],INSTANCE_PROMPT,INSTANCE_SCHEMA)
        indicated,binary,_=extent_views(photo,mask);append('extent','whole',[indicated,binary],EXTENT_PROMPT,EXTENT_SCHEMA)
    for name in ['semantic_evidence_gate.py','semantic_subject_guard.py','instance_identity_consistency.py','instance_evidence_views.py','target_identity_policy.py']:protected[str(Path(__file__).with_name(name))]=sha(Path(__file__).with_name(name))
    model=read(PRIMARY/'frozen_plan.json')['model_config_sha256'];save(S/'frozen_plan.json',{'driver_sha256':sha(runner.__file__),'requests':requests,'selected':[r['key'] for r in selected],'protected':protected,'model_config_sha256':model,'default_tokens':256,'all_original_gates_required':True,'source_native_selection_frozen_before_reference_scoring':True,'all76_native_valid_hypotheses_selected_no_GT_support_area_or_semantic_cherry_pick':True,'logical_candidate_ids_are_not_API_request_hashes':True,'source_candidate_bindings':[{'key':r['key'],'actual_source_request_sha256':r['request_sha256'],'native_index':r['native_index'],'source_selected':r['source_selected'],'mask_sha256':r['full_mask_sha256']} for r in selected],'never_mix_same_request_alternatives_as_simultaneous_instances':True,'new_Agent_runs':0,'adopted':False})
    wanted={r['request_sha256']:r for r in requests};cache=[];seen=set()
    for name in ['labelagent_MF_first_parts_20261108','labelagent_balanced_palette_parts_20261108','labelagent_concise_palette_candidates_20261108','labelagent_single_teacher_grid_20261108','labelagent_fixed_grid_followup_20261108','labelagent_source_gap_instances_20261108']:
        path=BASE/name/'semantic/frozen_plan.json'
        if not path.exists():continue
        old=read(path)
        if old.get('model_config_sha256')!=model:continue
        for r in old.get('requests',[]):
            key=r['request_sha256']
            if key in seen or key not in wanted or r['payload']!=wanted[key]['payload']:continue
            response=path.parent/'responses'/(key+'.json')
            if not response.exists():continue
            reply=read(response);stats=reply.get('local_vlm',{})
            if stats.get('model_config_sha256')!=model or stats.get('actual_image_token_counts')!=[256]*len(r['images']):continue
            file=S/'responses'/response.name;file.parent.mkdir(exist_ok=True);shutil.copy2(response,file);assert sha(file)==sha(response);seen.add(key);cache.append({'request_sha256':key,'source_plan':str(path),'source_plan_sha256':sha(path),'source_response':str(response),'source_response_sha256':sha(response),'exact_request_body_model_and_actual_tokens':True})
    save(S/'exact_prior_cache.json',{'rows':cache,'unique_exact_prior_reuses':len(seen)});print({'all_source_native_valid_hypotheses':len(selected),'logical_max_views':len(requests),'unique_max_requests':len(wanted),'exact_prior_reuses':len(seen),'new_model_calls':0},flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:runner.OUT=PRIMARY;runner.S=S;runner.run()
