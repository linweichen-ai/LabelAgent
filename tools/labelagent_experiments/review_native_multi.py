"""Keep original neutral, single-base and extent gates on newly supported masks."""
import base64,hashlib,json,sys,time,urllib.request,urllib.error
from pathlib import Path
import numpy as np
from PIL import Image
from run_native_multi import ROOT,OUT,INC,read,save,sha
from run_visual_budget import health
from semantic_evidence_gate import CAPTION_PROMPT,CAPTION_SCHEMA
from semantic_subject_guard import compatible_pallet_subject
from target_identity_policy import centered_photos,extent_views,EXTENT_PROMPT,EXTENT_SCHEMA
from instance_identity_consistency import target_views,INSTANCE_PROMPT,INSTANCE_SCHEMA,single_physical_base,png

S=OUT/'semantic'


def body(images,prompt,schema):
    result={'model':'labelagent-vlm','temperature':0,'max_tokens':256,'messages':[{'role':'user','content':[{'type':'text','text':prompt}]+[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(Path(p).read_bytes()).decode()}} for p in images]}],'response_format':{'type':'json_schema','json_schema':{'name':'labelagent','schema':schema,'strict':True}}}
    if len(images)==1:result['image_token_budget']=256
    return result


def prepare():
    if S.exists() and any(S.iterdir()):raise FileExistsError('Preserve frozen review')
    S.mkdir(exist_ok=True);audit=read(OUT/'support_area_audit.json');selected=[];requests=[]
    # An unloaded worker health response omits its model configuration. Bind
    # the already verified prior model config and verify every real response.
    h=read(ROOT/'outputs/labelagent_aspect_review_20261107/frozen_plan.json')
    protected={str(OUT/'support_area_audit.json'):sha(OUT/'support_area_audit.json')}
    for row in audit['candidates']:
        if not (row['valid'] and row['area_fits'] and not row['person_conflict']):continue
        r=next(x for x in audit['cases'] if x['case']==row['case'])['radii'][1]
        targets=[42,48] if row['case'].startswith('camera0') else [47,49]
        own=next(x for x in r['new_mask_ownership'] if x['key']==row['key'])['ownership']
        if not any(x['object_id'] in targets and x['qualified_support']>=.5 for x in own):continue
        selected.append(row['key']);maskfile=Path(row['folder'])/'complete_full_mask.npy';sourcefile=INC/'inputs'/(row['case']+'.png');protected.update({str(maskfile):sha(maskfile),str(sourcefile):sha(sourcefile)})
        mask=np.load(maskfile,allow_pickle=False);source=Image.open(sourcefile).convert('RGB');folder=S/row['key'];folder.mkdir()
        def append(stage,name,images,prompt,schema,kind='candidate'):
            paths=[]
            for i,im in enumerate(images):
                p=folder/f'{stage}_{kind}_{name}_{i}.png';im.save(p);paths.append(str(p));protected[str(p)]=sha(p)
            b=body(paths,prompt,schema);digest=hashlib.sha256(json.dumps(b,sort_keys=True).encode()).hexdigest();requests.append({'candidate':row['key'],'stage':stage,'name':name,'kind':kind,'images':paths,'payload':b,'request_sha256':digest})
        for name,im in centered_photos(source,mask)[0].items():
            append('neutral',name,[im],CAPTION_PROMPT,CAPTION_SCHEMA)
            for color,value in [('gray',127),('white',255)]:append('neutral',name,[Image.new('RGB',im.size,(value,)*3)],CAPTION_PROMPT,CAPTION_SCHEMA,color)
        for name,im in target_views(source,mask)[0].items():append('structure',name,[im],INSTANCE_PROMPT,INSTANCE_SCHEMA)
        indicated,binary,_=extent_views(source,mask);append('extent','whole',[indicated,binary],EXTENT_PROMPT,EXTENT_SCHEMA)
    protected.update({str(Path(__file__).with_name(n)):sha(Path(__file__).with_name(n)) for n in ['semantic_evidence_gate.py','semantic_subject_guard.py','instance_identity_consistency.py','instance_evidence_views.py','target_identity_policy.py']})
    save(S/'frozen_plan.json',{'driver_sha256':sha(__file__),'requests':requests,'selected':selected,'protected':protected,'model_config_sha256':h['model_config_sha256'],'default_tokens':256,'all_original_gates_required':True,'reviewed_masks_not_generated_from_reference_points':True,'diagnostic_support_filter_after_all_masks_frozen':True,'adopted':False})
    print({'candidates':len(selected),'max_conditional_queries':len({r['request_sha256'] for r in requests})},flush=True)


def run():
    if (S/'completed.json').exists():raise FileExistsError('Complete')
    plan=read(S/'frozen_plan.json');assert sha(__file__)==plan['driver_sha256'];results=[];calls=0
    def query(row):
        nonlocal calls
        file=S/'responses'/(row['request_sha256']+'.json')
        if not file.exists():
            for _ in range(1500):
                if (OUT/'STOP').exists():raise RuntimeError('STOP')
                try:
                    with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18868/v1/chat/completions',data=json.dumps(row['payload']).encode(),headers={'Content-Type':'application/json'}),timeout=300) as f:response=json.load(f)
                    save(file,response);calls+=1;break
                except urllib.error.HTTPError as e:
                    if e.code==503:time.sleep(.2);continue
                    raise
            else:raise RuntimeError('Shared service busy; resume missing only')
        response=read(file);stats=response['local_vlm'];assert stats['model_config_sha256']==plan['model_config_sha256'] and stats['actual_image_token_counts']==[256]*len(row['images']);decision=json.loads(response['choices'][0]['message']['content']);result={k:v for k,v in row.items() if k!='payload'};result.update(decision=decision,response=str(file),response_sha256=sha(file),actual_tokens=stats['actual_image_token_counts']);results.append(result);save(S/'progress.json',results);print(json.dumps({'stage':row['stage'],'kind':row['kind'],'candidate':row['candidate'][-7:],'view':row['name'],'decision':decision},ensure_ascii=False),flush=True);return result
    candidates=[]
    for key in plan['selected']:
        rows=[r for r in plan['requests'] if r['candidate']==key];neutral=[query(r) for r in rows if r['stage']=='neutral'];neutral_ok=all(compatible_pallet_subject(r['decision']) for r in neutral if r['kind']=='candidate');controls_ok=all(not compatible_pallet_subject(r['decision']) for r in neutral if r['kind']!='candidate');structure=[];extent=[]
        if neutral_ok and controls_ok:structure=[query(r) for r in rows if r['stage']=='structure']
        single_ok=bool(structure) and all(single_physical_base(r['decision']) for r in structure)
        if single_ok:extent=[query(r) for r in rows if r['stage']=='extent']
        extent_ok=bool(extent) and all(r['decision'].get('target_extent') in ['whole_visible_object','identifiable_occluded_object'] and compatible_pallet_subject({'object':r['decision'].get('object','')}) for r in extent)
        candidates.append({'key':key,'neutral_passed':neutral_ok,'matched_plain_controls_passed':controls_ok,'single_base_passed':single_ok,'extent_passed':extent_ok,'all_semantic_stages_passed':neutral_ok and controls_ok and single_ok and extent_ok})
    assert all(sha(p)==h for p,h in plan['protected'].items());assert health()['image_token_budget']==256
    save(S/'completed.json',{'rows':results,'candidates':candidates,'actual_unique_VLM_responses':len({r['request_sha256'] for r in results}),'logical_views':len(results),'fresh_this_execution':calls,'adopted':False,'annotation_writes':0,'Agent_runs':0,'remaining_checks_before_any_adoption':'Exact deployment representation and all original joint controls/Agent comparisons; semantic completion alone is insufficient.'})


if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
