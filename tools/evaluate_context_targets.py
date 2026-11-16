"""Cache-bound context comparison; only missing, distinct requests use the GPU."""
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
from tools.connect_local_vlm import require_real_service
from tools.evaluate_confirmed_region_context import check_idle
from tools.evaluate_focused_targets import verify_cached_locations
from tools.evaluate_source_availability import score
from tools.evaluate_vlm_grounding import validated_response
from tools.run_local_agent_acceptance import save
from tools.sam_evidence_recheck import needs_recheck,rechecked_pair

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def checked_cached_response(ref,payload_sha,fixture):
    if sha(ref['path'])!=ref['sha256']:raise ValueError('Cached response bytes changed')
    response=read(ref['path'])
    if response['full_payload_sha256']!=payload_sha:raise ValueError('Cached response belongs to a different request')
    parsed=validated_response(response['raw_response'],fixture['categories'],1);meta=response['raw_response']['local_vlm']
    if meta['model_config_sha256']!=fixture['model_config_sha256'] or meta['quantization']!='nf4' or meta['actual_image_token_counts']!=[256]:
        raise ValueError('Different model or actual encoding in cached response')
    return response,parsed

def evaluate(args):
    assert sha(args.plan)==args.plan_sha256;plan=read(args.plan);assert all(sha(p)==h for p,h in plan['source_bindings'].items())
    boxes=verify_cached_locations(plan);f=read(plan['fixture']);prior=read(plan['previous_summary']);original=read(plan['original_fixture'])
    eligible=[c['case_id'] for c in prior['cases'] if needs_recheck([c['views'][v]['raw_category'] for v in ('overview','linked')],(c['views']['overview'].get('support') or {}).get('supported_classes',[]))]
    assert eligible==[c['id'] for c in f['cases']] and len(eligible)==len(boxes)==2 and len(original['cases'])==29
    assets={}
    for ref in [r for c in f['cases'] for rr in c['views'].values() for r in rr]+[f['guard_image']]:
        p=Path(f['asset_root'])/ref['path'];assert sha(p)==ref['sha256'];assets[ref['path']]=(p.read_bytes(),'image/jpeg')
    check_idle();out=Path(args.output);out.mkdir(exist_ok=False)
    progress=dict(status='running',pid=os.getpid(),create_time=psutil.Process().create_time(),attempted_new_requests=0,valid_new_responses=0,rows=[],guard=None,adopted=False,default_enabled=False,workbench_writes=False)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        health=require_real_service('http://127.0.0.1:18868/v1',f['model']);assert health['model_config_sha256']==f['model_config_sha256'] and health['image_token_budget']==256
        save(out/'backend_before_PRIVATE.json',health)
        for c in f['cases']:
            body,digest=ascii_card_payload(f,assets,c,'overview');cached,parsed=checked_cached_response(plan['cached_overview'][c['id']],digest,f)
            progress['rows'].append(dict(case_id=c['id'],view='overview',parsed=parsed,full_payload_sha256=digest,cached_source=plan['cached_overview'][c['id']],original_view='linked',new_call=False))
        save(out/'progress.json',progress)
        for c,view,guard in [(c,'linked',False) for c in f['cases']]+[({'id':'default_budget_guard'},'guard',True)]:
            check_idle()
            if (out/'STOP').exists():raise InterruptedError('Stop requested')
            body,digest=ascii_card_payload(f,assets,c,view,guard);n=progress['attempted_new_requests']+1
            row=dict(number=n,case_id=c['id'],view=view,guard=guard,full_payload_sha256=digest,new_call=True)
            save(out/'request_inflight.json',row);progress['attempted_new_requests']=n;save(out/'progress.json',progress)
            started=time.monotonic();request=urllib.request.Request('http://127.0.0.1:18868/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with opener.open(request,timeout=300) as r:raw=json.load(r)
            row['seconds']=round(time.monotonic()-started,3);save(out/f'response_{n:03d}.json',{**row,'raw_response':raw});(out/'request_inflight.json').unlink()
            parsed=validated_response(raw,f['categories'],1);meta=raw['local_vlm'];assert meta['model_config_sha256']==f['model_config_sha256'] and meta['quantization']=='nf4'
            assert meta['actual_image_token_counts']==([238] if guard else [256]) and meta['effective_image_token_budget']==256
            row.update(parsed=parsed,actual_image_token_counts=meta['actual_image_token_counts'],image_grid_thw=meta['image_grid_thw']);progress['valid_new_responses']+=1
            if guard:progress['guard']=row
            else:progress['rows'].append(row)
            save(out/'progress.json',progress);print(json.dumps(dict(case_id=c['id'],view=view,parsed=parsed)),flush=True)
        progress['status']='complete'
    except InterruptedError:progress['status']='stopped'
    except Exception as e:
        import traceback
        progress.update(status='failed',error_type=type(e).__name__);(out/'error_PRIVATE.txt').write_text(traceback.format_exc())
    finally:
        index={(r['case_id'],r['view']):r['parsed'] for r in progress['rows']};rows=[];opposite=[];outcomes=[];truths={c['id']:c for c in original['cases']}
        for c in prior['cases']:
            pair=[c['views'][v]['category'] for v in ('overview','linked')]
            if c['case_id'] in eligible:
                fresh=[index.get((c['case_id'],v)) for v in ('overview','linked')]
                decision=rechecked_pair(pair,c['views']['overview']['support']['supported_classes'],[r['category'] for r in fresh]) if all(fresh) else {'categories':pair,'fresh_pair_matched_unique_SAM_support':False}
                outcomes.append(dict(case_id=c['case_id'],before=pair,fresh=fresh,**decision));pair=decision['categories']
                truth=truths[c['case_id']];expected=truth.get('expected_category') if truth['group']=='human' else ('pallet' if truth['role']=='pallet_fixture' else None)
                for view,r in zip(('overview','linked'),fresh):
                    if r and expected and r['category'] not in (expected,'uncertain'):opposite.append(dict(case_id=c['case_id'],view=view))
            for view,cat in zip(('overview','linked'),pair):rows.append(dict(case_id=c['case_id'],view=view,parsed={'category':cat,'reason':'frozen guarded prior or cache-bound context evidence'}))
        scores=score(original,rows);legacy=sum(p['role']=='pallet_fixture' and p['consensus']=='pallet' for p in scores['legacy_pairs'])
        checks={**scores['original_quality_checks'],'all3_new_responses_valid':progress['status']=='complete' and progress['valid_new_responses']==3,
                'all14_prior_human_correct_retained':not(set(prior['scores']['human_correct_ids'])-set(scores['human_correct_ids'])),
                'zero_wrong_human_consensus':not scores['human_wrong_consensus_ids'],'no_fresh_opposite_class_view':not opposite,
                'strict_gain_vs_guard':scores['human_correct_consensus']>14 or legacy>2,
                'all_sources_unchanged':all(sha(p)==h for p,h in plan['source_bindings'].items()),'plan_unchanged':sha(args.plan)==args.plan_sha256}
        result={**progress,'scores':scores,'outcomes':outcomes,'fresh_opposite_views':opposite,'legacy_pallet_correct':legacy,'checks':checks,'all_checks_passed':all(checks.values()),
                'finished_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':args.plan_sha256,'cached_real_quality_responses':2,'new_SAM_calls':0,
                'independent_validation':False,'pixel_accuracy_measured':False}
        save(out/'summary.json',result);save(out/'progress.json',progress)
    print(json.dumps(dict(status=result['status'],human_correct=scores['human_correct_consensus'],legacy_correct=legacy,checks=checks)),flush=True)
    return 0 if result['status']=='complete' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',required=True)
    raise SystemExit(evaluate(p.parse_args()))
