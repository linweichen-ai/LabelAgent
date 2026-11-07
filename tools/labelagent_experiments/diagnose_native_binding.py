"""Post-failure diagnosis; never override a failed neutral gate."""
import hashlib,json,sys,time,urllib.request,urllib.error
from pathlib import Path
import numpy as np
from PIL import Image
from run_native_multi import ROOT,OUT,INC,read,save,sha
from review_native_multi import body
from instance_identity_consistency import target_views,INSTANCE_PROMPT,INSTANCE_SCHEMA,single_physical_base
from target_identity_policy import extent_views,EXTENT_PROMPT,EXTENT_SCHEMA

D=OUT/'target_binding_diagnostic'


def prepare():
    if D.exists():raise FileExistsError('Preserve diagnostic')
    D.mkdir();review=read(OUT/'semantic/completed.json');prior=read(OUT/'semantic/frozen_plan.json');gen=read(OUT/'generation_completed.json');requests=[];protected={str(OUT/'semantic/completed.json'):sha(OUT/'semantic/completed.json')}
    for c in review['candidates']:
        assert not c['neutral_passed']
        row=next(x for x in gen['rows'] if x['key']==c['key']);maskfile=Path(row['folder'])/'complete_full_mask.npy';mask=np.load(maskfile,allow_pickle=False);sourcefile=INC/'inputs'/(row['case']+'.png');source=Image.open(sourcefile).convert('RGB');protected.update({str(sourcefile):sha(sourcefile),str(maskfile):sha(maskfile)})
        for kind,photo in [('real',source),('marked_gray',Image.new('RGB',source.size,(127,)*3))]:
            pointer=target_views(photo,mask)[0]['pointer'];indicated,binary,_=extent_views(photo,mask)
            for stage,images,prompt,schema in [('pointer',[pointer],INSTANCE_PROMPT,INSTANCE_SCHEMA),('extent',[indicated,binary],EXTENT_PROMPT,EXTENT_SCHEMA)]:
                paths=[]
                for i,im in enumerate(images):
                    p=D/f"{row['seed']}_{kind}_{stage}_{i}.png";im.save(p);paths.append(str(p));protected[str(p)]=sha(p)
                b=body(paths,prompt,schema);digest=hashlib.sha256(json.dumps(b,sort_keys=True).encode()).hexdigest();requests.append({'seed':row['seed'],'candidate':row['key'],'kind':kind,'stage':stage,'images':paths,'payload':b,'request_sha256':digest})
    save(D/'frozen_plan.json',{'driver_sha256':sha(__file__),'requests':requests,'protected':protected,'model_config_sha256':prior['model_config_sha256'],'max_unique_requests':len({r['request_sha256'] for r in requests}),'diagnostic_only':True,'neutral_failures_remain_failed':True,'no_mask_or_gate_changes':True})
    print({'diagnostic_requests':len(requests)},flush=True)


def run():
    if (D/'completed.json').exists():raise FileExistsError('Complete')
    plan=read(D/'frozen_plan.json');assert sha(__file__)==plan['driver_sha256'];rows=[]
    for row in plan['requests']:
        file=D/'responses'/(row['request_sha256']+'.json')
        if not file.exists():
            for _ in range(1500):
                if (OUT/'STOP').exists():raise RuntimeError('STOP')
                try:
                    with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18868/v1/chat/completions',data=json.dumps(row['payload']).encode(),headers={'Content-Type':'application/json'}),timeout=300) as f:response=json.load(f)
                    save(file,response);break
                except urllib.error.HTTPError as e:
                    if e.code==503:time.sleep(.2);continue
                    raise
            else:raise RuntimeError('Busy; resume missing only')
        response=read(file);assert response['local_vlm']['model_config_sha256']==plan['model_config_sha256'] and response['local_vlm']['actual_image_token_counts']==[256]*len(row['images']);decision=json.loads(response['choices'][0]['message']['content']);r={k:v for k,v in row.items() if k!='payload'};r.update(decision=decision,response=str(file),response_sha256=sha(file));rows.append(r);save(D/'progress.json',rows);print(json.dumps({'seed':row['seed'],'kind':row['kind'],'stage':row['stage'],'decision':decision},ensure_ascii=False),flush=True)
    assert all(sha(p)==h for p,h in plan['protected'].items())
    save(D/'completed.json',{'diagnostic_only':True,'rows':rows,'unique_actual_VLM_responses':len({r['request_sha256'] for r in rows}),'adopted':False,'original_neutral_failures_preserved':True,'Agent_runs':0,'original_single_base_gate_not_completed':'One pointer diagnostic is not both required structure views.'})


if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
