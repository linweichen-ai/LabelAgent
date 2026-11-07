"""Actual control-mask review views, exact cache reuse and model-bound replies."""
import hashlib,json,sys,time,urllib.request,urllib.error
from pathlib import Path
import numpy as np
from PIL import Image
from run_soft_prior_trial import OUT,ROOT,read,save,sha
from run_visual_budget import payload,health
from target_identity_policy import centered_photos
from semantic_subject_guard import compatible_pallet_subject

D=OUT/'semantic_controls'


def prepare():
    if D.exists():raise FileExistsError('Preserve control review')
    D.mkdir();prior=read(ROOT/'outputs/labelagent_native_multi_20261107/semantic/completed.json');model=read(ROOT/'outputs/labelagent_native_multi_20261107/semantic/frozen_plan.json')['model_config_sha256'];cache={r['request_sha256']:r for r in prior['rows']};requests=[];protected={}
    for row in read(OUT/'generation_completed.json')['rows']:
        if row['kind']=='real' or 'mask_sha256' not in row:continue
        arrays=np.load(row['raw_array_file'],allow_pickle=False);local=(arrays[f"{row['prediction_call']}_masks"][row['native_index']].squeeze()!=0).astype('uint8');source=Image.open(row['input']).convert('RGB');protected[row['raw_array_file']]=sha(row['raw_array_file']);protected[row['input']]=sha(row['input'])
        for name,im in centered_photos(source,local)[0].items():
            p=D/f"{row['kind']}_{row['mode']}_{name}.png";im.save(p);body=payload(p.read_bytes(),256);key=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest();r={'name':p.stem,'kind':row['kind'],'mode':row['mode'],'view':name,'image':str(p),'sha256':sha(p),'request_sha256':key}
            if key in cache:
                c=cache[key];response=read(c['response']);assert response['local_vlm']['model_config_sha256']==model;r.update(reuse_response=c['response'],reuse_response_sha256=sha(c['response']));protected[c['response']]=sha(c['response'])
            requests.append(r)
    save(D/'frozen_plan.json',{'driver_sha256':sha(__file__),'requests':requests,'protected':protected,'model_config_sha256':model,'all_actual_control_mask_pixels_retained':True,'default_neutral_prompt_and_schema':True,'max_new_requests':len({r['request_sha256'] for r in requests if 'reuse_response' not in r}),'new_real_mask_semantic_reviews':0,'reason':'No new remaining-target mask passed original sparse instance support; control review only.','adopted':False});print({'logical_controls':len(requests),'max_fresh':len({r['request_sha256'] for r in requests if 'reuse_response' not in r})},flush=True)


def run():
    if (D/'completed.json').exists():raise FileExistsError('Already complete')
    p=read(D/'frozen_plan.json');assert sha(__file__)==p['driver_sha256'];rows=[]
    for r in p['requests']:
        assert sha(r['image'])==r['sha256'];file=Path(r['reuse_response']) if 'reuse_response' in r else D/'responses'/(r['request_sha256']+'.json')
        if not file.exists():
            body=payload(Path(r['image']).read_bytes(),256);assert hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()==r['request_sha256']
            for _ in range(1500):
                if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
                try:
                    with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18868/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=300) as f:response=json.load(f)
                    save(file,response);break
                except urllib.error.HTTPError as e:
                    if e.code==503:time.sleep(.2);continue
                    raise
            else:raise RuntimeError('Shared VLM busy; preserve completed replies')
        response=read(file);assert response['local_vlm']['model_config_sha256']==p['model_config_sha256'] and response['local_vlm']['actual_image_token_counts']==[256];decision=json.loads(response['choices'][0]['message']['content']);row={**r,'decision':decision,'PAL_passed':compatible_pallet_subject(decision),'response':str(file),'response_sha256':sha(file),'reused':'reuse_response' in r};rows.append(row);save(D/'progress.json',rows);print(json.dumps({'name':r['name'],'decision':decision,'PAL':row['PAL_passed'],'reused':row['reused']},ensure_ascii=False),flush=True)
    assert all(sha(x)==h for x,h in p['protected'].items());save(D/'completed.json',{'rows':rows,'fresh_unique_VLM_responses':len({r['request_sha256'] for r in rows if not r['reused']}),'reused_logical_views':sum(r['reused'] for r in rows),'logical_views':len(rows),'PAL_false_positives':sum(r['PAL_passed'] for r in rows),'no_new_real_candidate_semantic_pass_claim':True,'adopted':False})


if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
