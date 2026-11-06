"""Frozen two-image raw VLM comparison, with verbatim archived responses."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import urllib.request
import psutil

REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.source_companion import source_companion_payload
from tools.ascii_target_recheck import ascii_card_payload
from tools.evaluate_vlm_grounding import validated_response
from tools.evaluate_confirmed_region_context import check_idle
from tools.connect_local_vlm import require_real_service
from tools.run_local_agent_acceptance import save
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def evaluate(args):
    assert sha(args.plan)==args.plan_sha256;plan=read(args.plan)
    assert all(sha(p)==h for p,h in plan['source_bindings'].items())
    fixtures={k:read(p) for k,p in plan['fixtures'].items()};assets={}
    for f in fixtures.values():
        for ref in [r for c in f['cases'] for vv in c['views'].values() for r in vv]+[f['guard_image']]:
            p=Path(f['asset_root'])/ref['path'];assert sha(p)==ref['sha256'];assets[ref['path']]=(p.read_bytes(),'image/jpeg')
    check_idle();out=Path(args.output);out.mkdir(exist_ok=False)
    progress=dict(status='loading_model',pid=os.getpid(),create_time=psutil.Process().create_time(),
                  attempted_requests=0,responses_received=0,rows=[],guard=None,default_enabled=False,workbench_writes=False)
    save(out/'progress.json',progress);cache={};opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        save(out/'backend_before_PRIVATE.json',require_real_service('http://127.0.0.1:18868/v1','labelagent-vlm'))
        progress['status']='running';save(out/'progress.json',progress)
        for job in plan['jobs']:
            check_idle()
            if (out/'STOP').exists():raise InterruptedError('Own study stop requested')
            variant=job['variant'];f=fixtures[variant];guard=job.get('guard',False)
            c={'id':'default_budget_guard'} if guard else next(c for c in f['cases'] if c['id']==job['case_id'])
            builder=source_companion_payload if variant=='candidate' else ascii_card_payload
            body,h=builder(f,assets,c,job['view'],guard);assert h==job['payload_sha256']
            row=dict(variant=variant,case_id=c['id'],view=job['view'],guard=guard,full_payload_sha256=h)
            if 'cached_response' in job:
                ref=job['cached_response'];assert sha(ref['path'])==ref['sha256'];receipt=read(ref['path'])
                assert receipt['full_payload_sha256']==h;raw=receipt['raw_response'];path=ref['path'];reused=True
            elif h in cache:raw,path=cache[h];reused=True
            else:
                n=progress['attempted_requests']+1;progress['attempted_requests']=n
                save(out/'request_inflight.json',{**row,'number':n});save(out/'progress.json',progress)
                req=urllib.request.Request('http://127.0.0.1:18868/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                started=time.monotonic()
                with opener.open(req,timeout=300) as response:raw=json.load(response)
                path=str(out/f'response_{n:03d}.json');save(Path(path),{**row,'number':n,'seconds':round(time.monotonic()-started,3),'raw_response':raw})
                progress['responses_received']+=1;(out/'request_inflight.json').unlink();reused=False
            cache[h]=(raw,path);count=1 if guard or variant=='baseline' else 2
            parsed=validated_response(raw,f['categories'],count);meta=raw['local_vlm']
            assert meta['model_config_sha256']==f['model_config_sha256'] and meta['quantization']=='nf4'
            assert meta['actual_image_token_counts']==([238] if guard else [256]*count) and meta['effective_image_token_budget']==256
            row.update(parsed=parsed,source_receipt=path,reused_identical_request=reused,actual_image_token_counts=meta['actual_image_token_counts'])
            if guard:progress['guard']=row
            else:progress['rows'].append(row)
            save(out/'progress.json',progress)
            print(json.dumps(dict(variant=variant,case_id=c['id'],view=job['view'],category=parsed['category'],cached=reused)),flush=True)
        progress['status']='complete'
    except InterruptedError:progress['status']='stopped'
    except Exception as error:
        progress.update(status='failed',error_type=type(error).__name__);(out/'error_PRIVATE.txt').write_text(traceback.format_exc())
    finally:
        index={(r['variant'],r['case_id'],r['view']):r['parsed']['category'] for r in progress['rows']};scores={}
        for variant in fixtures:
            groups=[]
            for c in fixtures[variant]['cases']:
                cats=[index.get((variant,c['id'],v)) for v in ('overview','linked')]
                expected=plan['evaluation_only'][c['id']]['expected'];role=plan['evaluation_only'][c['id']]['role']
                groups.append(dict(case_id=c['id'],role=role,categories=cats,both_expected=cats==[expected,expected],
                                   expected=expected,correct_views=sum(x==expected for x in cats),
                                   wrong_views=sum(x is not None and x not in (expected,'uncertain') for x in cats)))
            scores[variant]=groups
        candidates=scores['candidate'];baselines={r['case_id']:r for r in scores['baseline']}
        checks=dict(complete=progress['status']=='complete',two_original_pallets_all_four_views_correct=all(r['both_expected'] for r in candidates if r['role']=='protected_pallet'),
            approved_person_control_preserved=all(r['both_expected'] for r in candidates if r['role']=='person_fixture'),
            no_confirmed_six_correct_pair_lost=all(not baselines[r['case_id']]['both_expected'] or r['both_expected'] for r in candidates if r['role']=='confirmed_six'),
            no_confirmed_six_opposite_view=all(not r['wrong_views'] for r in candidates if r['role']=='confirmed_six'),
            all_blank_views_abstained=all(r['both_expected'] for r in candidates if r['role'] in ('blank_control','marked_blank_control')),
            pending_suspect_proposals_abstained=all(r['both_expected'] for r in candidates if r['role']=='suspected_non_pallet'),
            strict_raw_blank_or_confirmed_gain=(sum(r['correct_views'] for r in candidates if r['role'] in ('blank_control','marked_blank_control','confirmed_six')) > sum(r['correct_views'] for r in scores['baseline'] if r['role'] in ('blank_control','marked_blank_control','confirmed_six'))),
            source_bindings_unchanged=all(sha(p)==h for p,h in plan['source_bindings'].items()),plan_unchanged=sha(args.plan)==args.plan_sha256,
            sensitive_default_guard_passed=bool(progress['guard']))
        result={**progress,'scores':scores,'checks':checks,'all_checks_passed':all(checks.values()),'plan_sha256':args.plan_sha256,
                'finished_utc':datetime.now(timezone.utc).isoformat(),'raw_model_categories_never_rewritten':True,
                'external_pixel_or_SAM_guards_used':False,'new_SAM_calls':0,'candidate_adopted':False,'independent_validation':False}
        save(out/'summary.json',result);save(out/'progress.json',progress)
    print(json.dumps(dict(status=result['status'],requests=result['attempted_requests'],checks=checks)),flush=True)
    return 0 if result['status']=='complete' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',required=True)
    raise SystemExit(evaluate(p.parse_args()))
