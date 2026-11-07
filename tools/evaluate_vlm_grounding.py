"""Compare real loopback VLMs on a frozen local fixture; never writes annotations."""
import argparse
import base64
from collections import Counter
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
import urllib.request

from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent))
from connect_local_vlm import loopback_url,require_real_service
from run_local_agent_acceptance import save


def sha(raw):return hashlib.sha256(raw).hexdigest()


def checked_fixture(path):
    path=Path(path).resolve()
    fixture=json.loads(path.read_text(encoding='utf-8'))
    root=Path(fixture['asset_root'])
    root=(root if root.is_absolute() else path.parent/root).resolve()
    cases=fixture['cases']
    if not cases or len({row['id'] for row in cases})!=len(cases):raise ValueError('Empty or duplicate fixture cases')
    roles={row['role'] for row in cases}
    if not {'blank_control','pallet_fixture','suspected_non_pallet'}.issubset(roles):
        raise ValueError('Include positive, blank and suspected-non-pallet fixtures')
    if not roles.issubset({'blank_control','pallet_fixture','suspected_non_pallet','unconfirmed'}):
        raise ValueError('Unknown fixture role')
    minimum=fixture['minimum_positive_accepts']
    if type(minimum) is not int or not 1<=minimum<=sum(row['role']=='pallet_fixture' for row in cases):
        raise ValueError('A non-vacuous positive usefulness requirement is required')
    categories=fixture['categories']
    if (not isinstance(categories,list) or len(set(categories))!=len(categories)
            or not all(isinstance(value,str) and value for value in categories)
            or not {'pallet','uncertain'}.issubset(categories)):
        raise ValueError('Invalid category vocabulary')
    if fixture.get('temperature')!=0 or type(fixture.get('max_tokens')) is not int or not 1<=fixture['max_tokens']<=4096:
        raise ValueError('Expected deterministic bounded decoding')
    assets={}
    for case in cases:
        if set(case['views'])!={'single','context'}:raise ValueError('Each case needs both views')
        for kind,refs in case['views'].items():
            if not 1<=len(refs)<=8 or not isinstance(fixture['prompts'][kind],str):raise ValueError('Invalid view or prompt')
            for ref in refs:
                key=ref['path']
                file=(root/key).resolve()
                if not file.is_relative_to(root):raise ValueError('Fixture asset escapes the allowed image root')
                raw=file.read_bytes()
                if len(raw)>16*1024*1024 or sha(raw)!=ref['sha256']:raise ValueError('Fixture image bytes changed')
                with Image.open(io.BytesIO(raw)) as picture:
                    if picture.format not in {'JPEG','PNG'} or picture.width*picture.height>32_000_000:
                        raise ValueError('Unsupported fixture image')
                    mime='image/jpeg' if picture.format=='JPEG' else 'image/png'
                    picture.verify()
                assets[key]=(raw,mime)
    return fixture,assets


def build_payload(fixture,assets,case,kind,model):
    refs=case['views'][kind]
    schema={'type':'object','properties':{'category':{'type':'string','enum':fixture['categories']},
        'reason':{'type':'string','minLength':1,'maxLength':100}},
        'required':['category','reason'],'additionalProperties':False}
    content=[{'type':'text','text':fixture['prompts'][kind]}]
    for ref in refs:
        raw,mime=assets[ref['path']]
        if sha(raw)!=ref['sha256']:raise ValueError('Loaded image does not match the frozen request')
        content.append({'type':'image_url','image_url':{'url':'data:'+mime+';base64,'+base64.b64encode(raw).decode('ascii')}})
    body={'model':model,'temperature':fixture['temperature'],'max_tokens':fixture['max_tokens'],
        'messages':[{'role':'user','content':content}],
        'response_format':{'type':'json_schema','json_schema':{'name':'proposal_review','schema':schema,'strict':True}}}
    binding={'prompt':fixture['prompts'][kind],'image_sha256':[ref['sha256'] for ref in refs],
        'schema':schema,'temperature':fixture['temperature'],'max_tokens':fixture['max_tokens']}
    fingerprint=sha(json.dumps(binding,sort_keys=True,separators=(',',':')).encode())
    return body,fingerprint


def validated_response(raw,categories,image_count):
    meta=raw.get('local_vlm',{})
    if meta.get('backend')!='transformers_local_cuda' or meta.get('image_count')!=image_count:
        raise ValueError('A real CUDA response confirming every image is required')
    choice=raw['choices'][0]
    if choice.get('finish_reason')!='stop':raise ValueError('Truncated or unfinished model output')
    parsed=json.loads(choice['message']['content'])
    if (set(parsed)!={'category','reason'} or parsed['category'] not in categories
            or not isinstance(parsed['reason'],str) or not 1<=len(parsed['reason'])<=100):
        raise ValueError('Invalid constrained category response')
    return parsed


def pair_decision(first,second):
    a,b=first['category'],second['category']
    return 'accept' if a==b=='pallet' else 'reject' if a==b and a not in ('pallet','uncertain') else 'uncertain'


def fixture_checks(fixture,results,valid_responses):
    by_id={row['id']:row for row in results}
    complete=len(results)==len(fixture['cases']) and valid_responses==2*len(fixture['cases'])
    def selected(role):return [by_id[row['id']] for row in fixture['cases'] if row['role']==role and row['id'] in by_id]
    positives,blanks,suspects=[selected(role) for role in ('pallet_fixture','blank_control','suspected_non_pallet')]
    return {'fixture_fully_evaluated_with_valid_real_responses':complete,
        'blank_controls_not_accepted':complete and all(row['decision']!='accept' for row in blanks),
        'no_qualitative_pallet_rejected':complete and all(row['decision']!='reject' for row in positives),
        'minimum_qualitative_pallet_accepts':complete and sum(row['decision']=='accept' for row in positives)>=fixture['minimum_positive_accepts'],
        'suspected_non_pallet_not_accepted':complete and all(row['decision']!='accept' for row in suspects)}


def evaluate(args):
    base=loopback_url(args.base)
    fixture,assets=checked_fixture(args.fixture_json)
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'fixture_file_sha256':sha(Path(args.fixture_json).read_bytes()),
        'source_code_sha256':sha(Path(__file__).read_bytes()),'model_alias':args.model,
        'expected_quantization':args.expected_quantization,'cases':len(fixture['cases']),
        'planned_requests':2*len(fixture['cases']),'predeclared_checks':fixture['predeclared_checks'],
        'minimum_positive_accepts':fixture['minimum_positive_accepts'],
        'fixture_labels_are_human_truth':False,'independent_validation':False,'workbench_modified':False}
    save(output/'study_plan.json',plan)
    summary={'status':'running','pid':os.getpid(),'plan_sha256':sha((output/'study_plan.json').read_bytes()),
        'http_requests_attempted':0,'received_JSON_responses':0,'valid_real_CUDA_responses':0,'cases':[],
        'workbench_modified':False,'pixel_accuracy_evaluated':False,'independent_validation':False}
    save(output/'summary.json',summary)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        health=require_real_service(base,args.model)
        if health.get('quantization')!=args.expected_quantization or health.get('image_token_budget')!=fixture['image_token_budget']:
            raise ValueError('Backend quantization or image token budget differs from the frozen comparison')
        save(output/'backend_receipt.json',health)
        for case in fixture['cases']:
            row={'id':case['id'],'role':case['role'],'views':{}}
            for kind in ('single','context'):
                if (output/'STOP').exists():raise InterruptedError('Probe stop requested')
                body,binding=build_payload(fixture,assets,case,kind,args.model)
                started=time.monotonic()
                summary['http_requests_attempted']+=1
                save(output/'summary.json',summary)
                request=urllib.request.Request(base+'/chat/completions',json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                with opener.open(request,timeout=300) as response:raw=json.load(response)
                summary['received_JSON_responses']+=1
                receipt={'case_id':case['id'],'view':kind,'model_input_fingerprint_excluding_alias':binding,
                    'image_sha256':[ref['sha256'] for ref in case['views'][kind]],
                    'request_body_sha256':sha(json.dumps(body).encode()),'raw_response':raw,
                    'seconds':round(time.monotonic()-started,3),'valid_real_CUDA_response':False}
                receipt_path=output/f"response_{summary['http_requests_attempted']:03d}.json"
                save(receipt_path,receipt)
                parsed=validated_response(raw,fixture['categories'],len(case['views'][kind]))
                receipt.update(parsed=parsed,valid_real_CUDA_response=True)
                save(receipt_path,receipt)
                summary['valid_real_CUDA_responses']+=1
                row['views'][kind]={'parsed':parsed,'input_fingerprint_excluding_alias':binding,'seconds':receipt['seconds']}
                save(output/'summary.json',summary)
            row['decision']=pair_decision(row['views']['single']['parsed'],row['views']['context']['parsed'])
            summary['cases'].append(row)
            save(output/'summary.json',summary)
            print(json.dumps({'case_id':row['id'],'decision':row['decision']}),flush=True)
        summary['status']='complete'
    except InterruptedError:
        summary['status']='stopped'
    except Exception as error:
        summary.update(status='failed',error_type=type(error).__name__)
        (output/'local_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        summary['checks']=fixture_checks(fixture,summary['cases'],summary['valid_real_CUDA_responses'])
        summary['all_frozen_checks_passed']=all(summary['checks'].values())
        summary['decision_counts']=dict(Counter(row['decision'] for row in summary['cases']))
        summary['finished_utc']=datetime.now(timezone.utc).isoformat()
        summary['plan_unchanged']=sha((output/'study_plan.json').read_bytes())==summary['plan_sha256']
        save(output/'summary.json',summary)
    print(json.dumps({key:summary[key] for key in ('status','http_requests_attempted','valid_real_CUDA_responses','checks','decision_counts')}),flush=True)
    return 0 if summary['status']=='complete' else 2 if summary['status']=='stopped' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ('base','model','fixture-json','output'):parser.add_argument('--'+option,required=True)
    parser.add_argument('--expected-quantization',choices=['nf4','none'],default='nf4')
    raise SystemExit(evaluate(parser.parse_args()))
