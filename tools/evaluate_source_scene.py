"""Measure new source observations and replay, without changing raw classes."""
import argparse,json,hashlib,sys,os,traceback
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.evaluate_target_first import ReceiptClient,load_assets,save,validate_backend
from tools.source_scene import scene_payload,reviewed_category,CATEGORIES
from tools.evaluate_vlm_grounding import validated_response
from tools.ascii_target_recheck import ascii_card_payload
from tools.evaluate_confirmed_region_context import check_idle
from tools.connect_local_vlm import require_real_service
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def evaluate(args):
    import psutil
    assert sha(args.plan)==args.plan_sha256;plan=read(args.plan)
    assert all(sha(p)==h for p,h in plan['source_bindings'].items())
    f=read(plan['fixture']);assets=load_assets(f);assert len(f['cases'])==18
    raw=read(plan['prior_summary']);prior={(r['case_id'],r['view']):r for r in raw['rows'] if r['variant']=='candidate'};assert len(prior)==36
    check_idle();out=Path(args.output);out.mkdir(exist_ok=False)
    progress=dict(status='loading_model',pid=os.getpid(),create_time=psutil.Process().create_time(),attempted_requests=0,http_responses_received=0,responses_received=0,observations=[],replay=[],guard=None,default_enabled=False,original_annotations_written=False)
    save(out/'progress.json',progress);client=ReceiptClient(out,progress,check_idle,plan['max_new_requests'],plan.get('cached_responses'))
    try:
        save(out/'backend_before_PRIVATE.json',require_real_service('http://127.0.0.1:18868/v1',f['model']));progress['status']='running'
        for c in f['cases']:
            body,h=scene_payload(f,assets,c);assert h==plan['payload_hashes'][c['id']]
            r,path,reused=client.request(body,h,dict(stage='source_scene',case_id=c['id'],view='clean_only'))
            parsed=validated_response(r,CATEGORIES,1);validate_backend(r,f,1)
            obs=dict(case_id=c['id'],parsed=parsed,source_receipt=path,full_payload_sha256=h,reused=reused)
            progress['observations'].append(obs)
            for v in ('overview','linked'):
                old=prior[c['id'],v];cat=old['parsed']['category']
                progress['replay'].append(dict(case_id=c['id'],view=v,raw_category=cat,scene_category=parsed['category'],reviewed_category=reviewed_category(cat,parsed['category']),prior_receipt=old['source_receipt'],observation_receipt=path))
            save(out/'progress.json',progress);print(json.dumps(dict(case_id=c['id'],scene=parsed['category'],cached=reused)),flush=True)
        body,h=ascii_card_payload(f,assets,{'id':'default_budget_guard'},'guard',True);assert h==plan['guard_payload_sha256']
        r,path,reused=client.request(body,h,dict(stage='guard',case_id='default_budget_guard',view='guard'))
        validate_backend(r,f,1,True);parsed=validated_response(r,f['categories'],1)
        progress['guard']=dict(source_receipt=path,parsed=parsed,full_payload_sha256=h);progress['status']='complete'
    except InterruptedError as e:progress.update(status='stopped',stop_reason=str(e))
    except Exception as e:progress.update(status='failed',error_type=type(e).__name__);(out/'error_PRIVATE.txt').write_text(traceback.format_exc())
    finally:
        by_role={}
        for cid,ev in plan['evaluation_only'].items():
            role=ev['role'];g=by_role.setdefault(role,dict(total_views=0,raw_expected=0,reviewed_expected=0))
            rows=[r for r in progress['replay'] if r['case_id']==cid]
            g['total_views']+=len(rows);g['raw_expected']+=sum(r['raw_category']==ev['expected'] for r in rows);g['reviewed_expected']+=sum(r['reviewed_category']==ev['expected'] for r in rows)
        result={**progress,'counts_by_role':by_role,'plan_sha256':args.plan_sha256,'source_bindings_unchanged':all(sha(p)==h for p,h in plan['source_bindings'].items()),'raw_classifier_answers_never_rewritten':True,'model_observer_veto_replay_not_raw_classifier_gain':True,'candidate_adopted':False,'new_SAM_calls':0,'independent_validation':False}
        save(out/'summary.json',result);save(out/'progress.json',progress)
    print(json.dumps(dict(status=progress['status'],counts=by_role)),flush=True);return 0 if progress['status']=='complete' else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',required=True);raise SystemExit(evaluate(p.parse_args()))
