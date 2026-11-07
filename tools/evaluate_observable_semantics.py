"""Frozen bare-VLM comparison with auditable caches and no external class guard."""
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
from tools.observable_semantics import observable_payload,validated_observable
from tools.evaluate_vlm_grounding import validated_response
from tools.evaluate_confirmed_region_context import check_idle
from tools.connect_local_vlm import require_real_service
from tools.run_local_agent_acceptance import save
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def evaluate(args):
    assert sha(args.plan)==args.plan_sha256;plan=read(args.plan);assert all(sha(p)==h for p,h in plan['source_bindings'].items())
    fixtures={k:read(p) for k,p in plan['fixtures'].items()};assets={}
    for f in fixtures.values():
        for r in [r for c in f['cases'] for rr in c['views'].values() for r in rr]+[f['guard_image']]:
            p=Path(f['asset_root'])/r['path'];assert sha(p)==r['sha256'];assets[r['path']]=(p.read_bytes(),'image/jpeg')
    check_idle();out=Path(args.output);out.mkdir(exist_ok=False);progress=dict(status='running',pid=os.getpid(),create_time=psutil.Process().create_time(),attempted_requests=0,rows=[],guard=None,default_enabled=False,workbench_writes=False)
    cache={};opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        health=require_real_service('http://127.0.0.1:18868/v1','labelagent-vlm');save(out/'backend_before_PRIVATE.json',health)
        for job in plan['jobs']:
            check_idle()
            if (out/'STOP').exists():raise InterruptedError('Stop requested')
            variant=job['variant'];f=fixtures[variant];guard=job.get('guard',False);case={'id':'default_budget_guard'} if guard else next(c for c in f['cases'] if c['id']==job['case_id'])
            builder=observable_payload if variant=='candidate' else ascii_card_payload;validator=validated_observable if variant=='candidate' else validated_response
            body,digest=builder(f,assets,case,job['view'],guard);assert digest==job['payload_sha256']
            row=dict(variant=variant,case_id=job['case_id'],view=job['view'],guard=guard,full_payload_sha256=digest)
            if 'cached_response' in job:
                ref=job['cached_response'];assert sha(ref['path'])==ref['sha256'];path=Path(ref['path']);receipt=read(path);assert receipt['full_payload_sha256']==digest
                raw=receipt['raw_response'];row.update(reused_identical_request=True,source_receipt=str(path));cache[digest]=(raw,str(path))
            elif digest in cache:
                raw,path=cache[digest];row.update(reused_identical_request=True,source_receipt=path)
            else:
                n=progress['attempted_requests']+1;progress['attempted_requests']=n;save(out/'request_inflight.json',{**row,'number':n});save(out/'progress.json',progress)
                started=time.monotonic();req=urllib.request.Request('http://127.0.0.1:18868/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                with opener.open(req,timeout=300) as response:raw=json.load(response)
                path=out/f'response_{n:03d}.json';save(path,{**row,'number':n,'raw_response':raw});(out/'request_inflight.json').unlink()
                cache[digest]=(raw,str(path));row.update(reused_identical_request=False,source_receipt=str(path),seconds=round(time.monotonic()-started,3))
            parsed=validator(raw,f['categories'],1);meta=raw['local_vlm'];assert meta['model_config_sha256']==f['model_config_sha256'] and meta['quantization']=='nf4'
            assert meta['actual_image_token_counts']==([238] if guard else [256]) and meta['effective_image_token_budget']==256
            row.update(parsed=parsed,actual_image_token_counts=meta['actual_image_token_counts'],image_grid_thw=meta['image_grid_thw'])
            if guard:progress['guard']=row
            else:progress['rows'].append(row)
            save(out/'progress.json',progress);print(json.dumps(dict(variant=variant,case_id=job['case_id'],view=job['view'],category=parsed['category'],cached=row['reused_identical_request'])),flush=True)
        progress['status']='complete'
    except InterruptedError:progress['status']='stopped'
    except Exception as e:
        import traceback
        progress.update(status='failed',error_type=type(e).__name__);(out/'error_PRIVATE.txt').write_text(traceback.format_exc())
    finally:
        index={(r['variant'],r['case_id'],r['view']):r for r in progress['rows']};by_variant={}
        for variant,f in fixtures.items():
            cases=[]
            for c in f['cases']:
                cats=[index.get((variant,c['id'],v),{}).get('parsed',{}).get('category') for v in ('overview','linked')]
                expected=c.get('expected_for_development_only');cases.append(dict(case_id=c['id'],role=c.get('role'),categories=cats,both_match_expected=bool(expected and cats==[expected,expected])))
            by_variant[variant]=dict(cases=cases,all_development_cases_passed=bool(cases and all(c['both_match_expected'] for c in cases)))
        checks=dict(complete=progress['status']=='complete',all_original_positive_targets_preserved=all(c['both_match_expected'] for c in by_variant['candidate']['cases'] if c['role']=='protected_pallet'),
                    all_development_cases_passed=by_variant['candidate']['all_development_cases_passed'],source_bindings_unchanged=all(sha(p)==h for p,h in plan['source_bindings'].items()),plan_unchanged=sha(args.plan)==args.plan_sha256)
        result={**progress,'by_variant':by_variant,'checks':checks,'all_checks_passed':all(checks.values()),'finished_utc':datetime.now(timezone.utc).isoformat(),
                'plan_sha256':args.plan_sha256,'raw_model_categories_never_rewritten':True,'external_pixel_or_SAM_guards_used':False,'new_SAM_calls':0,'independent_validation':False}
        save(out/'summary.json',result);save(out/'progress.json',progress)
    print(json.dumps(dict(status=result['status'],requests=result['attempted_requests'],checks=checks)),flush=True)
    return 0 if result['status']=='complete' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',required=True);raise SystemExit(evaluate(p.parse_args()))
