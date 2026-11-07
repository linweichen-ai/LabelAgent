"""Frozen semantic counterexamples, with explicit identical-payload deduplication."""
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
from tools.reference_semantics import reason_first_payload as ascii_card_payload
from tools.connect_local_vlm import require_real_service
from tools.evaluate_confirmed_region_context import check_idle
from tools.evaluate_text_reference import validated_reason_first as validated_response
from tools.run_local_agent_acceptance import save

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def evaluate(args):
    assert sha(args.plan)==args.plan_sha256;plan=read(args.plan);assert all(sha(p)==h for p,h in plan['source_bindings'].items())
    f=read(plan['fixture']);assert len(f['cases'])==4 and sorted(c['role'] for c in f['cases'])==['blank_control','blank_control','person_fixture','suspected_non_pallet']
    assets={}
    for r in [r for c in f['cases'] for rr in c['views'].values() for r in rr]+[f['guard_image']]:
        p=Path(f['asset_root'])/r['path'];assert sha(p)==r['sha256'];assets[r['path']]=(p.read_bytes(),'image/jpeg')
    check_idle();out=Path(args.output);out.mkdir(exist_ok=False);progress=dict(status='running',pid=os.getpid(),create_time=psutil.Process().create_time(),attempted_requests=0,rows=[],guard=None,default_enabled=False,workbench_writes=False)
    save(out/'progress.json',progress);cache={};opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        health=require_real_service('http://127.0.0.1:18868/v1',f['model']);assert health['model_config_sha256']==f['model_config_sha256'] and health['image_token_budget']==256
        save(out/'backend_before_PRIVATE.json',health)
        for c,v,guard in [(c,v,False) for c in f['cases'] for v in ('overview','linked')]+[({'id':'default_budget_guard'},'guard',True)]:
            check_idle()
            if (out/'STOP').exists():raise InterruptedError('Stop requested')
            body,digest=ascii_card_payload(f,assets,c,v,guard)
            row=dict(case_id=c['id'],view=v,guard=guard,full_payload_sha256=digest)
            if digest in cache:
                row.update(parsed=cache[digest]['parsed'],response_number=cache[digest]['response_number'],reused_identical_request=True)
            else:
                n=progress['attempted_requests']+1;progress['attempted_requests']=n;save(out/'request_inflight.json',{**row,'number':n});save(out/'progress.json',progress)
                started=time.monotonic();request=urllib.request.Request('http://127.0.0.1:18868/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                with opener.open(request,timeout=300) as r:raw=json.load(r)
                save(out/f'response_{n:03d}.json',{**row,'number':n,'raw_response':raw});(out/'request_inflight.json').unlink()
                parsed=validated_response(raw,f['categories'],1);meta=raw['local_vlm'];assert meta['model_config_sha256']==f['model_config_sha256'] and meta['quantization']=='nf4'
                assert meta['actual_image_token_counts']==([238] if guard else [f['image_token_budgets'][v]]) and meta['effective_image_token_budget']==(256 if guard else f['image_token_budgets'][v])
                row.update(parsed=parsed,response_number=n,reused_identical_request=False,seconds=round(time.monotonic()-started,3));cache[digest]=row
            if guard:progress['guard']=row
            else:progress['rows'].append(row)
            save(out/'progress.json',progress);print(json.dumps(dict(case_id=c['id'],view=v,parsed=row['parsed'],cached=row['reused_identical_request'])),flush=True)
        progress['status']='complete'
    except InterruptedError:progress['status']='stopped'
    except Exception as e:
        import traceback
        progress.update(status='failed',error_type=type(e).__name__);(out/'error_PRIVATE.txt').write_text(traceback.format_exc())
    finally:
        index={(r['case_id'],r['view']):r for r in progress['rows']};complete=progress['status']=='complete' and len(index)==8 and progress['attempted_requests']==plan['expected_unique_requests_including_default']
        checks={'all_cases_valid':complete}
        for c in f['cases']:
            cats=[index.get((c['id'],v),{}).get('parsed',{}).get('category') for v in ('overview','linked')];expected='person' if c['role']=='person_fixture' else 'uncertain'
            checks[c['id']]=complete and cats==[expected,expected]
        checks.update(default_budget_restored=bool(progress['guard']),source_bindings_unchanged=all(sha(p)==h for p,h in plan['source_bindings'].items()),plan_unchanged=sha(args.plan)==args.plan_sha256)
        result={**progress,'checks':checks,'all_checks_passed':all(checks.values()),'finished_utc':datetime.now(timezone.utc).isoformat(),
                'plan_sha256':args.plan_sha256,'all_queries_are_excluded_from_references':True,'independent_validation':False,'new_SAM_calls':0,'case_roles_not_sent':True}
        save(out/'summary.json',result)
    print(json.dumps(dict(status=result['status'],requests=result['attempted_requests'],checks=checks)),flush=True)
    return 0 if result['status']=='complete' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',required=True);raise SystemExit(evaluate(p.parse_args()))
