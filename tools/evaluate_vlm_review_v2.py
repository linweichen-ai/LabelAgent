"""Classify and review using VLM only; preserve initial and reviewed decisions."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
import psutil
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.ascii_target_recheck import ascii_card_payload
from tools.vlm_proposal_review import review_payload,validated_review,reviewed_category
from tools.evaluate_vlm_grounding import validated_response
from tools.evaluate_confirmed_region_context import check_idle
from tools.connect_local_vlm import require_real_service
from tools.run_local_agent_acceptance import save
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save_review_response(output,number,value):
    path=Path(output)/f'response_{number:03d}.json'
    save(path,value)
    return str(path)

def evaluate(args):
    assert sha(args.plan)==args.plan_sha256;plan=read(args.plan);assert all(sha(p)==h for p,h in plan['source_bindings'].items());f=read(plan['fixture'])
    assets={}
    for ref in [r for c in f['cases'] for rr in c['views'].values() for r in rr]+[f['guard_image']]:
        p=Path(f['asset_root'])/ref['path'];assert sha(p)==ref['sha256'];assets[ref['path']]=(p.read_bytes(),'image/jpeg')
    check_idle();out=Path(args.output);out.mkdir(exist_ok=False);progress=dict(status='running',pid=os.getpid(),create_time=psutil.Process().create_time(),attempted_requests=0,rows=[],wrong_draft_reviews=[],guard=None,default_enabled=False,workbench_writes=False)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));cache={}
    def call(body,digest,kind,cid,view,cached=None):
        check_idle()
        if (out/'STOP').exists():raise InterruptedError('Stop requested')
        if cached:
            assert sha(cached['path'])==cached['sha256'];receipt=read(cached['path']);assert receipt['full_payload_sha256']==digest;raw=receipt['raw_response'];path=cached['path'];reused=True
        elif digest in cache:raw,path=cache[digest];reused=True
        else:
            n=progress['attempted_requests']+1;progress['attempted_requests']=n;job=dict(number=n,kind=kind,case_id=cid,view=view,full_payload_sha256=digest);save(out/'request_inflight.json',job);save(out/'progress.json',progress)
            started=time.monotonic();req=urllib.request.Request('http://127.0.0.1:18868/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with opener.open(req,timeout=300) as r:raw=json.load(r)
            path=save_review_response(out,n,{**job,'seconds':round(time.monotonic()-started,3),'raw_response':raw});(out/'request_inflight.json').unlink();reused=False
        cache[digest]=(raw,path);parsed=validated_review(raw) if kind in ('review','wrong_draft_review') else validated_response(raw,f['categories'],1)
        meta=raw['local_vlm'];assert meta['model_config_sha256']==f['model_config_sha256'] and meta['quantization']=='nf4' and meta['actual_image_token_counts']==([238] if kind=='guard' else [256])
        return dict(parsed=parsed,source_receipt=path,reused_identical_request=reused,full_payload_sha256=digest)
    try:
        health=require_real_service('http://127.0.0.1:18868/v1',f['model']);save(out/'backend_before_PRIVATE.json',health)
        for c in f['cases']:
            for view in ('overview','linked'):
                body,digest=ascii_card_payload(f,assets,c,view);ref=plan.get('baseline_cache',{}).get(c['id'],{}).get(view)
                baseline=call(body,digest,'classification',c['id'],view,ref);draft=baseline['parsed']['category'];review=None;final=draft
                if draft!='uncertain':
                    body,digest=review_payload(f,assets,c,view,draft);assert digest==plan['review_branches'][c['id']][view][draft]
                    review=call(body,digest,'review',c['id'],view);final=reviewed_category(draft,review['parsed']['verdict'])
                row=dict(case_id=c['id'],view=view,initial_category=draft,final_category=final,classification=baseline,review=review)
                progress['rows'].append(row);save(out/'progress.json',progress);print(json.dumps(dict(case_id=c['id'],view=view,initial=draft,final=final,verdict=review and review['parsed']['verdict'])),flush=True)
        for job in plan.get('wrong_draft_jobs',[]):
            c=next(c for c in f['cases'] if c['id']==job['case_id']);body,digest=review_payload(f,assets,c,job['view'],job['draft']);assert digest==job['payload_sha256']
            review=call(body,digest,'wrong_draft_review',c['id'],job['view']);progress['wrong_draft_reviews'].append({**job,'review':review})
            save(out/'progress.json',progress)
        body,digest=ascii_card_payload(f,assets,{'id':'default_budget_guard'},'guard',True);progress['guard']=call(body,digest,'guard','default_budget_guard','guard');progress['status']='complete'
    except InterruptedError:progress['status']='stopped'
    except Exception as e:
        import traceback
        progress.update(status='failed',error_type=type(e).__name__);(out/'error_PRIVATE.txt').write_text(traceback.format_exc())
    finally:
        cases=[]
        for c in f['cases']:
            group=[r for r in progress['rows'] if r['case_id']==c['id']];initial=[next((r['initial_category'] for r in group if r['view']==v),None) for v in ('overview','linked')];final=[next((r['final_category'] for r in group if r['view']==v),None) for v in ('overview','linked')]
            expected=c.get('expected_for_development_only');cases.append(dict(case_id=c['id'],role=c.get('role'),initial=initial,final=final,both_expected=bool(expected and final==[expected,expected])))
        complete=progress['status']=='complete' and len(progress['rows'])==2*len(f['cases'])
        checks={'complete':complete,'sources_unchanged':all(sha(p)==h for p,h in plan['source_bindings'].items()),'plan_unchanged':sha(args.plan)==args.plan_sha256}
        if plan['mode']=='development':checks.update(protected_pallets_preserved=all(c['both_expected'] for c in cases if c['role']=='protected_pallet'),all_cases_expected=all(c['both_expected'] for c in cases),wrong_drafts_not_blindly_accepted=bool(progress['wrong_draft_reviews']) and all(r['review']['parsed']['verdict']!='supported' for r in progress['wrong_draft_reviews']))
        result={**progress,'cases':cases,'checks':checks,'all_checks_passed':all(checks.values()),'mode':plan['mode'],'finished_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':args.plan_sha256,
                'first_classifier_categories_preserved_verbatim':True,'reviewed_output_separate':True,'SAM_or_pixel_filters_used':False,'independent_accuracy_measured':False,'new_SAM_calls':0}
        save(out/'summary.json',result);save(out/'progress.json',progress)
    print(json.dumps(dict(status=result['status'],new_requests=result['attempted_requests'],checks=checks)),flush=True)
    return 0 if result['status']=='complete' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',required=True);raise SystemExit(evaluate(p.parse_args()))
