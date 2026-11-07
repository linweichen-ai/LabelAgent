"""Compare actual single-image VLM token budgets on byte-identical local inputs.

No workbench configuration or annotations are written. Every successful call
must attest its CUDA backend, model configuration and actual encoded grid.
"""
import argparse
import base64
from collections import Counter
from datetime import datetime,timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from PIL import Image

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'tools')]
from connect_local_vlm import loopback_url,request_json,require_real_service
from run_local_agent_acceptance import save

def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def build_payload(fixture,view,raw,kind,budget):
    body={'model':fixture['model'],'temperature':fixture['temperature'],'max_tokens':fixture['max_tokens'],
        'messages':[{'role':'user','content':[{'type':'text','text':fixture['prompts'][kind]},
            {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(raw).decode()}}]}],
        'response_format':{'type':'json_schema','json_schema':{'name':'density_category','schema':fixture['schema'],'strict':True}}}
    fingerprint=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    body['image_token_budget']=budget
    return body,fingerprint

def validate_response(raw,fixture,budget,expected):
    meta=raw.get('local_vlm',{})
    if (raw.get('model')!=fixture['model'] or meta.get('backend')!='transformers_local_cuda'
            or meta.get('quantization')!='nf4' or meta.get('model_config_sha256')!=fixture['model_config_sha256']
            or meta.get('image_count')!=1 or meta.get('requested_image_token_budget')!=budget
            or meta.get('effective_image_token_budget')!=budget or meta.get('actual_image_token_counts')!=[expected]):
        raise ValueError('Real response did not attest the frozen image budget, model and token count')
    grid=meta.get('image_grid_thw')
    if not isinstance(grid,list) or len(grid)!=1 or len(grid[0])!=3 or math.prod(grid[0])//4!=expected:
        raise ValueError('Actual image grid and frozen processor count differ')
    choice=raw['choices'][0]
    if choice.get('finish_reason')!='stop':raise ValueError('Model output did not finish')
    parsed=json.loads(choice['message']['content'])
    if (set(parsed)!={'category','reason'} or parsed['category'] not in {'pallet','person','uncertain'}
            or not isinstance(parsed['reason'],str) or not 1<=len(parsed['reason'])<=180):
        raise ValueError('Invalid category schema response')
    return parsed

def quality_checks(fixture,rows,budget):
    by_key={(row['case_id'],row['view']):row for row in rows if row['budget']==budget}
    complete=len(by_key)==2*len(fixture['cases'])
    by_role={role:[c for c in fixture['cases'] if c['role']==role] for role in
        ('person_fixture','pallet_fixture','blank_control','suspected_non_pallet')}
    def categories(case):return [by_key.get((case['id'],view),{}).get('parsed',{}).get('category') for view in ('single','outline_context')]
    return {'all_expected_real_responses':complete,
        'people_both_views_person':complete and all(set(categories(c))=={'person'} for c in by_role['person_fixture']),
        'blank_views_uncertain':complete and all(set(categories(c))=={'uncertain'} for c in by_role['blank_control']),
        'minimum_three_pallet_consensus':complete and sum(set(categories(c))=={'pallet'} for c in by_role['pallet_fixture'])>=3,
        'pallet_views_never_person':complete and all('person' not in categories(c) for c in by_role['pallet_fixture']),
        'suspect_views_uncertain':complete and all(set(categories(c))=={'uncertain'} for c in by_role['suspected_non_pallet'])}

def checked_fixture(path,asset_root):
    fixture=read(path);root=Path(asset_root).resolve()
    if (fixture['policies']!=[256,1024] or fixture['max_tokens']!=192 or fixture['temperature']!=0
            or fixture['planned_real_requests']!=40 or fixture['per_request_images']!=1
            or fixture['schema']['properties']['category']['enum']!=['pallet','person','uncertain']):
        raise ValueError('Unexpected frozen density fixture')
    roles=Counter(c['role'] for c in fixture['cases'])
    if roles!=Counter(person_fixture=2,pallet_fixture=5,blank_control=2,suspected_non_pallet=1):
        raise ValueError('Fixture must retain the complete original control panel')
    if len({c['id'] for c in fixture['cases']})!=10:raise ValueError('Case IDs must be unique')
    assets={}
    for case in fixture['cases']:
        if set(case['views'])!={'single','outline_context'} or sorted(case['policy_order'])!=[256,1024]:
            raise ValueError('Unexpected single-image views or policy order')
        for view in case['views'].values():
            file=(root/view['path']).resolve()
            if not file.is_relative_to(root):raise ValueError('Input image escapes the asset root')
            raw=file.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=view['sha256']:raise ValueError('Frozen image bytes changed')
            with Image.open(io.BytesIO(raw)) as image:
                if image.format!='PNG' or list(image.size)!=view['size']:raise ValueError('Frozen PNG dimensions changed')
            if not view['expected_actual_image_tokens']['1024']>view['expected_actual_image_tokens']['256']:
                raise ValueError('Image does not have a real processing contrast')
            assets[view['path']]=raw
    return fixture,assets

def evaluate(args):
    fixture,assets=checked_fixture(args.fixture_json,args.asset_root)
    base=loopback_url(args.base)
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    state=Path(args.state_file);before=file_sha(state)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'fixture_sha256':file_sha(args.fixture_json),
        'runner_sha256':file_sha(__file__),'server_sha256':file_sha(REPO/'tools/local_vlm_server.py'),
        'planned_requests':40,'policies':[256,1024],'new_SAM3_requests':0,'workbench_modified':False,
        'comparison_primary':'pallet cases with both views classified pallet',
        'density_gain_requires':'strict primary increase over 256 and all frozen guards preserved',
        'fixture_is_human_truth':False,'scope':fixture['comparison_scope']}
    save(output/'study_plan.json',plan);plan_hash=file_sha(output/'study_plan.json')
    summary={'status':'running','pid':os.getpid(),'attempted_requests':0,'valid_real_CUDA_responses':0,
        'rows':[],'workbench_modified':False,'new_SAM3_requests':0,'pixel_accuracy_evaluated':False}
    save(output/'summary.json',summary)
    try:
        if request_json('http://127.0.0.1:8765/api/run/status')['status']!='idle':raise RuntimeError('Workbench is active')
        health=require_real_service(base,fixture['model'])
        if (health.get('quantization')!='nf4' or health.get('model_config_sha256')!=fixture['model_config_sha256']
                or health.get('image_token_budget')!=256 or health.get('supports_single_image_budget_override') is not True):
            raise ValueError('Default physical model or explicit budget override support was not verified')
        save(output/'backend_before.json',health)
        for case in fixture['cases']:
            for kind,view in case['views'].items():
                for budget in case['policy_order']:
                    if (output/'STOP').exists():raise InterruptedError('Stop requested')
                    body,fingerprint=build_payload(fixture,view,assets[view['path']],kind,budget)
                    summary['attempted_requests']+=1;save(output/'summary.json',summary)
                    start=time.monotonic()
                    raw=request_json(base+'/chat/completions',body)
                    parsed=validate_response(raw,fixture,budget,view['expected_actual_image_tokens'][str(budget)])
                    row={'case_id':case['id'],'role':case['role'],'view':kind,'budget':budget,'parsed':parsed,
                        'image_sha256':view['sha256'],'input_fingerprint_excluding_policy':fingerprint,
                        'seconds':round(time.monotonic()-start,3),'actual_image_tokens':raw['local_vlm']['actual_image_token_counts'][0],
                        'inference_stats':raw['local_vlm']['inference_stats']}
                    save(output/f"response_{summary['attempted_requests']:03d}.json",{'input':{key:value for key,value in row.items() if key!='parsed'},'raw_response':raw})
                    summary['rows'].append(row);summary['valid_real_CUDA_responses']+=1
                    save(output/'summary.json',summary)
                    print(json.dumps({'completed':summary['valid_real_CUDA_responses'],'case':case['id'],'view':kind,
                        'budget':budget,'tokens':row['actual_image_tokens'],'category':parsed['category']}),flush=True)
        after=request_json(base[:-3]+'/health');save(output/'backend_after.json',after)
        summary['default_budget_restored']=after['image_token_budget']==256
        summary['status']='complete'
    except InterruptedError:summary['status']='stopped'
    except Exception as error:
        summary.update(status='failed',error_type=type(error).__name__)
        (output/'local_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        summary['policy_checks']={str(b):quality_checks(fixture,summary['rows'],b) for b in (256,1024)}
        pairs={}
        for row in summary['rows']:pairs.setdefault((row['case_id'],row['view']),[]).append(row)
        summary['all_twenty_input_pairs_identical']=len(pairs)==20 and all(len(rows)==2 and len({r['input_fingerprint_excluding_policy'] for r in rows})==1 for rows in pairs.values())
        summary['state_unchanged']=file_sha(state)==before
        summary['fixture_unchanged']=file_sha(args.fixture_json)==plan['fixture_sha256']
        summary['plan_unchanged']=file_sha(output/'study_plan.json')==plan_hash
        summary['server_code_unchanged']=file_sha(REPO/'tools/local_vlm_server.py')==plan['server_sha256']
        try:checked_fixture(args.fixture_json,args.asset_root);summary['image_assets_unchanged']=True
        except Exception:summary['image_assets_unchanged']=False
        summary['quality_passed_by_policy']={str(b):all(summary['policy_checks'][str(b)].values()) for b in (256,1024)}
        summary['protocol_checks']={
            'all_forty_real_responses':summary['status']=='complete' and summary['valid_real_CUDA_responses']==40,
            'all_twenty_input_pairs_identical':summary['all_twenty_input_pairs_identical'],
            'default_budget_restored':summary.get('default_budget_restored',False),
            **{key:summary[key] for key in ('state_unchanged','fixture_unchanged','plan_unchanged','server_code_unchanged','image_assets_unchanged')}}
        summary['pallet_consensus_by_policy']={str(b):sum(all(any(row['case_id']==c['id'] and row['view']==kind
            and row['budget']==b and row['parsed']['category']=='pallet' for row in summary['rows'])
            for kind in ('single','outline_context')) for c in fixture['cases'] if c['role']=='pallet_fixture') for b in (256,1024)}
        summary['strict_pallet_consensus_gain']=summary['pallet_consensus_by_policy']['1024']>summary['pallet_consensus_by_policy']['256']
        summary['all_protocol_checks_passed']=all(summary['protocol_checks'].values())
        summary['higher_density_all_quality_checks_passed']=summary['quality_passed_by_policy']['1024']
        summary['higher_density_adoption_checks_passed']=summary['all_protocol_checks_passed'] and summary['higher_density_all_quality_checks_passed'] and summary['strict_pallet_consensus_gain']
        summary['finished_utc']=datetime.now(timezone.utc).isoformat();save(output/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','attempted_requests','valid_real_CUDA_responses','quality_passed_by_policy','state_unchanged')}),flush=True)
    return 0 if summary['status']=='complete' else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fixture-json','asset-root','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--base',default='http://127.0.0.1:18868/v1')
    parser.add_argument('--state-file',default=str(REPO/'work/state.json'))
    raise SystemExit(evaluate(parser.parse_args()))
