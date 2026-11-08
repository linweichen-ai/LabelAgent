"""Unchanged whole-instance semantic gates on ALL native-valid MF-first masks."""
import hashlib,json,shutil,sys
from pathlib import Path
from prepare_mf_first_parts import OUT
from prepare_palette_parts import ROOT,BASE
import numpy as np
from PIL import Image
from run_rgb_layers import read,save
from deliver_context_graph import sha
import review_box_layers as runner
from semantic_evidence_gate import CAPTION_PROMPT,CAPTION_SCHEMA
from target_identity_policy import centered_photos,extent_views,EXTENT_PROMPT,EXTENT_SCHEMA
from instance_identity_consistency import target_views,INSTANCE_PROMPT,INSTANCE_SCHEMA
S=OUT/'semantic'

def prepare():
    if S.exists():raise FileExistsError('Preserve frozen MF-first whole-instance review')
    source=OUT/'generation_completed.json';g=read(source);selected=[r for r in g['rows'] if r['valid']];assert selected;S.mkdir();requests=[];protected={str(source):sha(source),str(Path(__file__)):sha(__file__)}
    for r in selected:
        key=r['request_sha256'];maskfile=Path(r['folder'])/'complete_full_mask.npy';mask=np.load(maskfile,allow_pickle=False);assert hashlib.sha256(mask.tobytes()).hexdigest()==r['full_mask_sha256'];photo=Image.open(r['source_image']).convert('RGB');protected[str(maskfile)]=sha(maskfile);protected[r['source_image']]=sha(r['source_image']);folder=S/key;folder.mkdir()
        def append(stage,name,images,prompt,schema,kind='candidate'):
            files=[]
            for i,im in enumerate(images):
                file=folder/f'{stage}_{kind}_{name}_{i}.png';im.save(file);files.append(str(file));protected[str(file)]=sha(file)
            payload=runner.body(files,prompt,schema);digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();requests.append({'candidate':key,'stage':stage,'name':name,'kind':kind,'images':files,'payload':payload,'request_sha256':digest})
        for name,photo_view in centered_photos(photo,mask)[0].items():
            append('neutral',name,[photo_view],CAPTION_PROMPT,CAPTION_SCHEMA)
            for color,value in [('gray',127),('white',255)]:append('neutral',name,[Image.new('RGB',photo_view.size,(value,)*3)],CAPTION_PROMPT,CAPTION_SCHEMA,color)
        for name,view in target_views(photo,mask)[0].items():append('structure',name,[view],INSTANCE_PROMPT,INSTANCE_SCHEMA)
        indicated,binary,_=extent_views(photo,mask);append('extent','whole',[indicated,binary],EXTENT_PROMPT,EXTENT_SCHEMA)
    for name in ['semantic_evidence_gate.py','semantic_subject_guard.py','instance_identity_consistency.py','instance_evidence_views.py','target_identity_policy.py']:protected[str(Path(__file__).with_name(name))]=sha(Path(__file__).with_name(name))
    model=read(OUT/'frozen_plan.json')['model_config_sha256'];save(S/'frozen_plan.json',{'driver_sha256':sha(runner.__file__),'requests':requests,'selected':[r['request_sha256'] for r in selected],'source_selected_outputs':[{'cue':r['index'],'arm':r['arm'],'pixels':r['native_pixels'],'mask_sha256':r['full_mask_sha256'],'original_point_caption_PAL':r['original_neutral_point_PAL_evidence']} for r in selected],'protected':protected,'model_config_sha256':model,'default_tokens':256,'all_original_gates_required':True,'selection':'ALL source-selected complete native outputs satisfying original prompt, native-input-border and zero synthetic-padding-FG conditions; no GT, support or area filter','original_negative_gray_white_single_base_extent_gates_unchanged':True,'no_source_point_PAL_evidence_fabricated':True,'no_reference_files_read_for_selection':True,'extra_Agent_rounds':0,'adopted':False})
    # Inspect only explicitly related old cohorts and retain wanted hashes,
    # avoiding an unbounded in-memory catalogue of every historical image.
    wanted={r['request_sha256']:r for r in requests};cache=[];seen=set()
    for name in ['labelagent_balanced_palette_parts_20261108','labelagent_concise_palette_candidates_20261108','labelagent_single_teacher_grid_20261108','labelagent_fixed_grid_followup_20261108','labelagent_source_gap_instances_20261108']:
        plan=BASE/name/'semantic/frozen_plan.json'
        if not plan.exists():continue
        old=read(plan)
        if old.get('model_config_sha256')!=model:continue
        for r in old.get('requests',[]):
            key=r['request_sha256']
            if key in seen or key not in wanted or r['payload']!=wanted[key]['payload']:continue
            file=plan.parent/'responses'/(key+'.json')
            if not file.is_file():continue
            reply=read(file);stats=reply.get('local_vlm',{})
            if stats.get('model_config_sha256')!=model or stats.get('actual_image_token_counts')!=[256]*len(r['images']):continue
            dest=S/'responses'/file.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(file,dest);assert sha(dest)==sha(file);seen.add(key);cache.append({'request_sha256':key,'source_plan':str(plan),'source_plan_sha256':sha(plan),'source_response':str(file),'source_response_sha256':sha(file),'exact_payload_model_and_actual_tokens':True})
    save(S/'exact_prior_cache.json',{'rows':cache,'unique_exact_prior_reuses':len(seen)});save(OUT/'semantic_source_selection.json',{'selected_all_native_valid':len(selected),'selected_before_support_area_audit':not (OUT/'audits').exists(),'generation_sha256':sha(source),'all_original_semantic_gates_unchanged':True,'preparer_sha256':sha(__file__)});print({'all_native_valid_selected':len(selected),'logical_max_views':len(requests),'unique_max_requests':len(wanted),'exact_prior_reuses':len(seen),'new_model_calls':0},flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:runner.OUT=OUT;runner.S=S;runner.run()
