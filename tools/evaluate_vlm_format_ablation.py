"""Diagnose constrained output on a declared subset and verify default processing.

This reuses audited constrained responses from the completed density study;
new unconstrained responses are diagnostic, not independent validation.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'tools')]
from connect_local_vlm import loopback_url,request_json,require_real_service
from evaluate_vlm_density import build_payload,checked_fixture,file_sha,read
from run_local_agent_acceptance import save

def checked_identity(raw,fixture,expected,requested):
    meta=raw.get('local_vlm',{})
    if (raw.get('model')!=fixture['model'] or meta.get('backend')!='transformers_local_cuda'
            or meta.get('quantization')!='nf4' or meta.get('model_config_sha256')!=fixture['model_config_sha256']
            or meta.get('image_count')!=1 or meta.get('requested_image_token_budget')!=requested
            or meta.get('effective_image_token_budget')!=256 or meta.get('actual_image_token_counts')!=[expected]):
        raise ValueError('Response model or actual image encoding differs from the original 256 study')
    grid=meta.get('image_grid_thw')
    if not isinstance(grid,list) or len(grid)!=1 or math.prod(grid[0])//4!=expected:
        raise ValueError('Image grid and expected processor tokens differ')

def parsed_diagnostic(raw):
    choice=raw['choices'][0]
    try:
        parsed=json.loads(choice['message']['content'])
        strict=(choice['finish_reason']=='stop' and isinstance(parsed,dict) and set(parsed)=={'category','reason'}
            and parsed['category'] in {'pallet','person','uncertain'} and isinstance(parsed['reason'],str)
            and 1<=len(parsed['reason'])<=180)
        return parsed,strict
    except (ValueError,TypeError):return None,False

def evaluate(args):
    fixture,assets=checked_fixture(args.fixture_json,args.asset_root)
    original=read(Path(args.density_study)/'summary.json')
    if original['status']!='complete' or original['valid_real_CUDA_responses']!=40 or not original['all_protocol_checks_passed']:
        raise ValueError('A complete attested density study is required')
    selected=args.case_ids.split(',')
    cases={c['id']:c for c in fixture['cases']}
    if not selected or len(set(selected))!=len(selected) or any(c not in cases for c in selected):
        raise ValueError('Invalid diagnostic subset')
    prior={(r['case_id'],r['view']):r for r in original['rows'] if r['budget']==256}
    base=loopback_url(args.base)
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    state=Path(args.state_file);before=file_sha(state)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'source_code_sha256':file_sha(__file__),
        'server_sha256':file_sha(REPO/'tools/local_vlm_server.py'),'fixture_sha256':file_sha(args.fixture_json),
        'parent_density_summary_sha256':file_sha(Path(args.density_study)/'summary.json'),
        'diagnostic_cases':selected,'new_unconstrained_requests':2*len(selected),'default_restoration_requests':1,
        'prior_real_constrained_responses_reused':2*len(selected),'image_token_budget':256,
        'difference':'omit response_format only; keep image bytes, prompt, budget, model and generation limits',
        'subset_selected_after_prior_failure':True,'independent_validation':False,'workbench_modified':False}
    save(output/'study_plan.json',plan);plan_sha=file_sha(output/'study_plan.json')
    summary={'status':'running','pid':os.getpid(),'attempted_requests':0,'valid_identity_responses':0,
        'rows':[],'workbench_modified':False,'diagnostic_only':True,'new_SAM3_requests':0}
    save(output/'summary.json',summary)
    try:
        if request_json('http://127.0.0.1:8765/api/run/status')['status']!='idle':raise RuntimeError('Workbench is active')
        health=require_real_service(base,fixture['model']);save(output/'backend_before.json',health)
        default_case=next(c for c in fixture['cases'] if c['role']=='blank_control')
        view=default_case['views']['single']
        body,binding=build_payload(fixture,view,assets[view['path']],'single',256)
        body.pop('image_token_budget')
        summary['attempted_requests']+=1;save(output/'summary.json',summary)
        raw=request_json(base+'/chat/completions',body)
        checked_identity(raw,fixture,view['expected_actual_image_tokens']['256'],None)
        parsed,strict=parsed_diagnostic(raw)
        if not strict:raise ValueError('Default restoration probe did not return valid structured JSON')
        summary['valid_identity_responses']+=1
        summary['default_no_override_verified']=True
        save(output/'default_restoration_response.json',{'raw_response':raw,'parsed':parsed,
            'expected_default_tokens':view['expected_actual_image_tokens']['256'],'input_fingerprint':binding,
            'response_format_retained':True,'image_budget_parameter_omitted':True})
        save(output/'summary.json',summary)
        print(json.dumps({'default_restored':True,'actual_tokens':raw['local_vlm']['actual_image_token_counts']}),flush=True)
        for case_id in selected:
            case=cases[case_id]
            for kind,view in case['views'].items():
                if (output/'STOP').exists():raise InterruptedError('Stop requested')
                body,binding=build_payload(fixture,view,assets[view['path']],kind,256)
                baseline=prior[case_id,kind]
                if binding!=baseline['input_fingerprint_excluding_policy']:raise ValueError('Diagnostic input no longer matches the constrained request')
                body.pop('response_format')
                summary['attempted_requests']+=1;save(output/'summary.json',summary)
                start=time.monotonic();raw=request_json(base+'/chat/completions',body)
                checked_identity(raw,fixture,view['expected_actual_image_tokens']['256'],256)
                if raw['local_vlm']['inference_stats'].get('structured_output') is not False:
                    raise ValueError('Unconstrained diagnostic still used a format constraint')
                parsed,strict=parsed_diagnostic(raw)
                category=parsed.get('category') if isinstance(parsed,dict) else None
                row={'case_id':case_id,'role':case['role'],'view':kind,'prior_constrained_category':baseline['parsed']['category'],
                    'unconstrained_category':category,'strict_original_schema_valid':strict,
                    'finish_reason':raw['choices'][0]['finish_reason'],'seconds':round(time.monotonic()-start,3),
                    'same_source_input_fingerprint':binding,'actual_image_tokens':raw['local_vlm']['actual_image_token_counts'][0]}
                summary['rows'].append(row);summary['valid_identity_responses']+=1
                save(output/f"unconstrained_{len(summary['rows']):03d}.json",{'row':row,'raw_response':raw})
                save(output/'summary.json',summary)
                print(json.dumps(row),flush=True)
        summary['status']='complete'
    except InterruptedError:summary['status']='stopped'
    except Exception as error:
        summary.update(status='failed',error_type=type(error).__name__)
        (output/'local_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        summary['strict_unconstrained_responses']=sum(r['strict_original_schema_valid'] for r in summary['rows'])
        summary['state_unchanged']=file_sha(state)==before
        summary['fixture_unchanged']=file_sha(args.fixture_json)==plan['fixture_sha256']
        summary['server_code_unchanged']=file_sha(REPO/'tools/local_vlm_server.py')==plan['server_sha256']
        summary['plan_unchanged']=file_sha(output/'study_plan.json')==plan_sha
        summary['finished_utc']=datetime.now(timezone.utc).isoformat();save(output/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','attempted_requests','valid_identity_responses','strict_unconstrained_responses','state_unchanged')}),flush=True)
    return 0 if summary['status']=='complete' else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fixture-json','asset-root','density-study','case-ids','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--base',default='http://127.0.0.1:18868/v1')
    parser.add_argument('--state-file',default=str(REPO/'work/state.json'))
    raise SystemExit(evaluate(parser.parse_args()))
