"""Exact cached neutral point review, followed by unchanged complete SAM."""
import datetime,hashlib,json,os,shutil,sys,time,urllib.request,urllib.error
from pathlib import Path
from run_rgb_layers import ROOT,read,save,sha
from paired_source_cue import body,external_live
from semantic_subject_guard import compatible_pallet_subject
import run_trained_point_sam as sam
OUT=ROOT/'outputs/labelagent_base_material_20261108'
PAIR=Path(os.environ['LABELAGENT_PAIRED_CUE_OUTPUT'])

def review():
    if (OUT/'review_completed.json').exists():raise FileExistsError('Do not repeat completed source reviews')
    status,live=external_live()
    if live:raise RuntimeError('Named external worker is still live')
    p=read(OUT/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());execution=OUT/'review_execution_plan.json'
    if not execution.exists():save(execution,{'at':datetime.datetime.now().astimezone().isoformat(),'driver_sha256':sha(__file__),'frozen_plan_sha256':sha(OUT/'frozen_plan.json'),'same_requests_before_network':True})
    assert read(execution)['driver_sha256']==sha(__file__) and read(execution)['frozen_plan_sha256']==sha(OUT/'frozen_plan.json');catalogue={}
    for f in [PAIR/'completed.json',ROOT/'outputs/labelagent_trained_point_review_20261108/completed.json']:
        for r in read(f)['rows']:catalogue[r['request_sha256']]=r
    rows=[];fresh=0;seen=set()
    for proposal in p['rows']:
        for i,v in enumerate(proposal['new_point_views']):
            assert all(sha(b['image'])==b['sha256'] for b in v['bindings']);request=body(*[b['image'] for b in v['bindings']]);k=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest();assert k==v['request_sha256'] and request==read(v['request_file']);file=OUT/'responses'/(k+'.json')
            if not file.exists():
                previous=catalogue.get(k);cached=False
                if previous:
                    f=Path(previous.get('response_file') or previous['response']);assert sha(f)==previous['response_sha256'] and read(previous['request_file'])==request;reply=read(f);reply=reply.get('raw_response',reply);cached=True
                else:
                    if shutil.disk_usage(OUT).free<16*2**20:raise RuntimeError('Keep raw-reply reserve')
                    for _ in range(1500):
                        if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
                        try:
                            with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18868/v1/chat/completions',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'}),timeout=300) as f:reply=json.load(f)
                            fresh+=1;break
                        except urllib.error.HTTPError as e:
                            if e.code==503:time.sleep(.2);continue
                            raise
                    else:raise RuntimeError('Shared service busy')
                assert reply['local_vlm']['model_config_sha256']==p['model_config_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256,256];save(file,{'raw_response':reply,'historical_exact_reuse':cached})
            saved=read(file);reply=saved['raw_response'];assert reply['local_vlm']['model_config_sha256']==p['model_config_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256,256];decision=json.loads(reply['choices'][0]['message']['content']);row={**v,'partition':proposal['source_partition_index'],'new_point_index':i+1,'decision':decision,'PAL_subject_evidence':compatible_pallet_subject(decision),'response_file':str(file),'response_sha256':sha(file),'historical_exact_reuse':saved['historical_exact_reuse'],'same_batch_reuse':k in seen};seen.add(k);rows.append(row);save(OUT/'review_progress.json',{'rows':rows,'actual_calls_this_execution':fresh});print({'partition':row['partition'],'point':i+1,'object':decision['object'],'PAL_subject_evidence':row['PAL_subject_evidence'],'reused':row['historical_exact_reuse'] or row['same_batch_reuse']},flush=True)
    save(OUT/'review_completed.json',{'status':'complete_neutral_material_point_review','rows':rows,'logical_views':len(rows),'unique_payloads':len(seen),'actual_new_VLM_calls':len({r['request_sha256'] for r in rows if not r['historical_exact_reuse']}),'actual_calls_this_execution':fresh,'exact_historical_reuses':len({r['request_sha256'] for r in rows if r['historical_exact_reuse']}),'same_batch_reuses':sum(r['same_batch_reuse'] for r in rows),'new_SAM_calls':0,'new_Agent_runs':0})

def prepare_SAM():
    p=read(OUT/'frozen_plan.json');d=read(OUT/'review_completed.json');qualified=[]
    for r in p['rows']:
        views=[v for v in d['rows'] if v['partition']==r['source_partition_index']];assert len(views)==2
        if all(v['PAL_subject_evidence'] for v in views):qualified.append({**r,'all_new_positive_points_neutrally_reviewed':True})
    save(OUT/'proposal_adapter/completed.json',{'rows':qualified,'review_sha256':sha(OUT/'review_completed.json'),'frozen_plan_sha256':sha(OUT/'frozen_plan.json'),'no_rejected_point_applied':True});sam.OUT=OUT/'SAM';sam.REVIEW=OUT/'proposal_adapter';sam.prepare();print({'qualified_hypotheses':len(qualified),'new_SAM_calls':0},flush=True)

def run_SAM():
    p=read(OUT/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());sam.OUT=OUT/'SAM';sam.REVIEW=OUT/'proposal_adapter';sam.run()

if __name__=='__main__':{'review':review,'prepare_SAM':prepare_SAM,'run_SAM':run_SAM}[sys.argv[1]]()
