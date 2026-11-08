"""Original full-instance semantic gates on ALL source-valid gap outputs; no adoption."""
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image
from run_rgb_layers import ROOT,read,save,sha
import review_box_layers as runner
from semantic_evidence_gate import CAPTION_PROMPT,CAPTION_SCHEMA
from target_identity_policy import centered_photos,extent_views,EXTENT_PROMPT,EXTENT_SCHEMA
from instance_identity_consistency import target_views,INSTANCE_PROMPT,INSTANCE_SCHEMA
SOURCE=Path('D:/AI/Codex/Home/visualizations/2026/11/04/01a1a56c-0d6f-7d61-8326-d3bbd344d2df/labelagent_source_gap_instances_20261108/SAM')
OUT=Path('D:/AI/Codex/Home/visualizations/2026/11/04/01a1a56c-0d6f-7d61-8326-d3bbd344d2df/labelagent_source_gap_instances_20261108');S=OUT/'semantic'

def prepare():
    if S.exists():raise FileExistsError('Preserve frozen full-instance semantic review')
    generation=read(SOURCE/'generation_completed.json');selected=[r for r in generation['rows'] if r['valid']];assert selected;S.mkdir(parents=True);requests=[];protected={str(SOURCE/'generation_completed.json'):sha(SOURCE/'generation_completed.json'),str(Path(__file__)):sha(__file__)}
    for row in selected:
        key=row['key'];maskfile=Path(row['folder'])/'complete_full_mask.npy';mask=np.load(maskfile,allow_pickle=False);sourcefile=Path(row['source_image']);source=Image.open(sourcefile).convert('RGB');folder=S/key;folder.mkdir();protected[str(maskfile)]=sha(maskfile);protected[str(sourcefile)]=sha(sourcefile)
        def append(stage,name,images,prompt,schema,kind='candidate'):
            paths=[]
            for i,im in enumerate(images):
                file=folder/f'{stage}_{kind}_{name}_{i}.png';im.save(file);paths.append(str(file));protected[str(file)]=sha(file)
            payload=runner.body(paths,prompt,schema);digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();requests.append({'candidate':key,'stage':stage,'name':name,'kind':kind,'images':paths,'payload':payload,'request_sha256':digest})
        for name,im in centered_photos(source,mask)[0].items():
            append('neutral',name,[im],CAPTION_PROMPT,CAPTION_SCHEMA)
            for color,value in [('gray',127),('white',255)]:append('neutral',name,[Image.new('RGB',im.size,(value,)*3)],CAPTION_PROMPT,CAPTION_SCHEMA,color)
        for name,im in target_views(source,mask)[0].items():append('structure',name,[im],INSTANCE_PROMPT,INSTANCE_SCHEMA)
        indicated,binary,_=extent_views(source,mask);append('extent','whole',[indicated,binary],EXTENT_PROMPT,EXTENT_SCHEMA)
    for name in ['semantic_evidence_gate.py','semantic_subject_guard.py','instance_identity_consistency.py','instance_evidence_views.py','target_identity_policy.py']:protected[str(Path(__file__).with_name(name))]=sha(Path(__file__).with_name(name))
    model=read(SOURCE.parent/'frozen_plan.json')['model_config_sha256'];save(S/'frozen_plan.json',{'driver_sha256':sha(runner.__file__),'requests':requests,'selected':[r['key'] for r in selected],'source_selected_outputs':[{'cue':r['index'],'pixels':r['raw_pixels'],'mask_sha256':r['mask_sha256']} for r in selected],'protected':protected,'model_config_sha256':model,'default_tokens':256,'all_original_gates_required':True,'selection':'ALL complete original-selected source-gap SAM outputs passing original native prompt/border validity; no reference agreement or acceptance-area filtering; original neutral/gray-white/single-base/whole-extent gates unchanged','no_reference_files_read_for_selection':True,'not_an_adoption_or_geometry_pass_claim':True,'extra_Agent_rounds':0,'adopted':False})
    # Reuse only exact prior requests with matching real model/tokens/responses.
    catalogue={}
    for path in list((ROOT/'outputs').glob('labelagent_*/semantic/frozen_plan.json'))+list(OUT.parent.glob('labelagent_*/semantic/frozen_plan.json')):
        old=read(path)
        if old.get('model_config_sha256')!=model:continue
        for r in old.get('requests',[]):
            file=path.parent/'responses'/(r.get('request_sha256','')+'.json')
            if r.get('payload') and file.is_file():catalogue.setdefault(r['request_sha256'],[]).append((r,file,path))
    cache=[]
    for r in requests:
        for old,file,plan in catalogue.get(r['request_sha256'],[]):
            if old['payload']!=r['payload']:continue
            response=read(file);stats=response.get('local_vlm',{})
            if stats.get('model_config_sha256')!=model or stats.get('actual_image_token_counts')!=[256]*len(r['images']):continue
            target=S/'responses'/file.name;target.parent.mkdir(exist_ok=True)
            if not target.exists():shutil.copy2(file,target)
            assert sha(target)==sha(file);cache.append({'request_sha256':r['request_sha256'],'source_plan':str(plan),'source_plan_sha256':sha(plan),'source_response':str(file),'source_response_sha256':sha(file),'exact_body_model_and_actual_tokens_verified':True});break
    save(S/'exact_prior_cache.json',{'rows':cache,'unique_exact_prior_reuses':len({r['request_sha256'] for r in cache})});print({'diagnostic_extrema':[(r['index'],r['raw_pixels']) for r in selected],'logical_max_views':len(requests),'unique_max_requests':len({r['request_sha256'] for r in requests}),'unique_exact_prior_reuses':len({r['request_sha256'] for r in cache}),'new_model_calls':0},flush=True)

if __name__=='__main__':
    runner.OUT=OUT;runner.S=S
    if sys.argv[1]=='prepare':prepare()
    else:runner.run()
