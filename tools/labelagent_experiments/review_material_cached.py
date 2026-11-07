"""Cache-first neutral review; preserve the historical uncached run separately."""
import hashlib,json,time,urllib.request,urllib.error
from pathlib import Path
from run_material_anchor import ROOT,OUT,SHORT,old
from run_visual_budget import payload
from neutral_caption_cache import lookup
from semantic_subject_guard import compatible_pallet_subject


def review_one(proposal,model_config,catalogue,response_dir,allow_network=False):
    raw=Path(proposal['semantic_image']).read_bytes();body=payload(raw,256);key=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest();assert key==proposal['request_sha256'];cached=lookup(raw,model_config,catalogue)
    if cached:return {**proposal,'decision':json.loads(cached['reply']['choices'][0]['message']['content']),'response':cached['response'],'response_sha256':cached['response_sha256'],'new_model_call':False,'cache_verified':True}
    if not allow_network:raise RuntimeError('Verified cache miss; no network request authorized by this invocation')
    file=Path(response_dir)/(key+'.json')
    for _ in range(1500):
        if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
        try:
            with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18868/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=300) as f:reply=json.load(f)
            old.save(file,reply);break
        except urllib.error.HTTPError as e:
            if e.code==503:time.sleep(.2);continue
            raise
    else:raise RuntimeError('Shared manager busy; resume missing requests only')
    assert reply['local_vlm']['model_config_sha256']==model_config and reply['local_vlm']['actual_image_token_counts']==[256]
    return {**proposal,'decision':json.loads(reply['choices'][0]['message']['content']),'response':str(file),'response_sha256':old.digest(file),'new_model_call':True,'cache_verified':False}


def main():
    target=OUT/'cache_first_replay.json'
    if target.exists():raise FileExistsError('Preserve completed cache replay')
    p=old.read(OUT/'frozen_plan.json');original=old.read(OUT/'semantic_completed.json');catalogue=[]
    for file in [SHORT/'semantic/completed.json',ROOT/'outputs/labelagent_native_multi_20261107/semantic/completed.json',ROOT/'outputs/labelagent_soft_prior_20261108/semantic_controls/completed.json',OUT/'semantic_completed.json']:catalogue+=old.read(file)['rows']
    rows=[]
    for r in p['proposals']:
        row=review_one(r,p['model_config_sha256'],catalogue,OUT/'cache_first_responses',allow_network=False);row['PAL_guard']=compatible_pallet_subject(row['decision']);prior=next(v for v in original['rows'] if v['key']==r['key']);row['same_PAL_guard_as_actual_original_reply']=row['PAL_guard']==prior['PAL_guard'];rows.append(row)
    old.save(target,{'rows':rows,'new_model_calls':sum(r['new_model_call'] for r in rows),'cache_verified_logical_views':len(rows),'all_PAL_guard_decisions_preserved':all(r['same_PAL_guard_as_actual_original_reply'] for r in rows),'historical_actual_13_transmissions_not_erased_or_relabelled':True,'default_future_entry':'review_one checks exact historical catalogue before any network request'});print({'cache_verified_views':len(rows),'new_calls':0,'original_guard_decisions_preserved':all(r['same_PAL_guard_as_actual_original_reply'] for r in rows)},flush=True)


if __name__=='__main__':main()
