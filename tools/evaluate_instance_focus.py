"""Real instance-focused class diagnosis with blank and pallet controls.

Candidate classes and fixture roles are never sent to the model. A shared class
across two views is only a proposal, not a confirmed annotation or pixel truth.
"""
import argparse
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
import urllib.request

import cv2
import numpy as np
from PIL import Image,ImageDraw

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'tools')]
from connect_local_vlm import loopback_url,require_real_service
from evaluate_point_merges import file_sha,read
from evaluate_vlm_grounding import build_payload,validated_response
from run_local_agent_acceptance import save
from source.agent.edit_agent import _rasterize


def largest_mask_candidate(shapes,width,height):
    if not shapes:raise ValueError('A nonempty candidate list is required')
    return max(shapes,key=lambda shape:(int(_rasterize(shape,width,height).sum()),-shape['id']))


def render_instance_views(image,shape,padding=.1):
    if not 0<=padding<=.5:raise ValueError('Invalid crop padding')
    photo=Image.open(io.BytesIO(image)).convert('RGB')
    width,height=photo.size
    mask=_rasterize(shape,width,height)
    yy,xx=np.nonzero(mask)
    if not len(xx):raise ValueError('Selected candidate has no foreground pixels')
    px=math.ceil((int(xx.max())-int(xx.min())+1)*padding)
    py=math.ceil((int(yy.max())-int(yy.min())+1)*padding)
    box=(max(0,int(xx.min())-px),max(0,int(yy.min())-py),
        min(width,int(xx.max())+1+px),min(height,int(yy.max())+1+py))
    raw=photo.crop(box)
    cropped=mask[box[1]:box[3],box[0]:box[2]]
    foreground=Image.composite(raw,Image.new('RGB',raw.size,(127,127,127)),Image.fromarray(cropped*255))
    outlined=raw.copy()
    draw=ImageDraw.Draw(outlined)
    contours,_=cv2.findContours(cropped,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points=[tuple(point) for point in contour.reshape(-1,2).tolist()]
        if len(points)>=2:draw.line(points+points[:1],fill=(255,0,255),width=2)
    return [foreground,raw,outlined],{'original_crop_box':list(box),'selected_pixels':int(mask.sum()),
        'width':raw.width,'height':raw.height,'label_text_rendered':False,'image_resized':False}


def consensus_category(first,second):
    return first if first==second and first in {'pallet','person'} else 'uncertain'


def focus_checks(fixture,rows,valid):
    expected={row['id']:row for row in fixture['cases']}
    actual={row['id']:row for row in rows}
    complete=(len(rows)==len(expected)==len(actual) and set(actual)==set(expected)
        and valid==2*len(expected))
    def selected(role):return [row for row in rows if expected[row['id']]['role']==role]
    persons=selected('person_fixture');pallets=selected('pallet_fixture')
    blanks=selected('blank_control');suspects=selected('suspected_non_pallet')
    def raw_categories(row):return [row['views'][kind]['parsed']['category'] for kind in ('single','context')]
    return {'all_cases_have_valid_real_CUDA_responses':bool(complete),
        'all_person_fixtures_agree_person':bool(complete and persons and all(row['consensus']=='person' for row in persons)),
        'pallet_fixtures_never_called_person':bool(complete and pallets and all('person' not in raw_categories(row) for row in pallets)),
        'minimum_pallet_consensus':bool(complete and sum(row['consensus']=='pallet' for row in pallets)>=fixture['minimum_pallet_consensus']),
        'blank_views_always_uncertain':bool(complete and blanks and all(set(raw_categories(row))=={'uncertain'} for row in blanks)),
        'suspected_non_target_never_accepted':bool(complete and suspects and all(set(raw_categories(row))=={'uncertain'} for row in suspects))}


def checked_focus_fixture(path):
    path=Path(path).resolve()
    fixture=read(path)
    root=Path(fixture['asset_root']).resolve()
    counts=Counter(row['role'] for row in fixture['cases'])
    if (not set(counts).issubset({'person_fixture','pallet_fixture','blank_control','suspected_non_pallet'})
            or counts['person_fixture']<2 or counts['pallet_fixture']<5 or counts['blank_control']<2 or counts['suspected_non_pallet']<1):
        raise ValueError('Include at least two people, five pallets, two blanks and one suspect')
    if len({row['id'] for row in fixture['cases']})!=len(fixture['cases']):raise ValueError('Duplicate cases')
    if (fixture['categories']!=['pallet','person','uncertain'] or fixture['temperature']!=0
            or fixture['max_tokens']!=192 or fixture['image_token_budget']!=256
            or type(fixture['minimum_pallet_consensus']) is not int
            or not 3<=fixture['minimum_pallet_consensus']<=counts['pallet_fixture']):
        raise ValueError('Unexpected frozen diagnosis settings')
    assets={}
    for case in fixture['cases']:
        if set(case['views'])!={'single','context'} or len(case['views']['single'])!=1 or len(case['views']['context'])!=3:
            raise ValueError('Expected one cutout and a three-view context')
        for refs in case['views'].values():
            for ref in refs:
                file=(root/ref['path']).resolve()
                if not file.is_relative_to(root):raise ValueError('Image outside the allowed asset root')
                raw=file.read_bytes()
                if len(raw)>16*1024*1024 or hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('Image bytes changed')
                with Image.open(io.BytesIO(raw)) as image:
                    if image.format not in {'JPEG','PNG'} or image.width*image.height>32_000_000:raise ValueError('Unsupported image')
                    mime='image/jpeg' if image.format=='JPEG' else 'image/png'
                    image.verify()
                assets[ref['path']]=(raw,mime)
    return fixture,assets


def evaluate(args):
    fixture,assets=checked_focus_fixture(args.fixture_json)
    base=loopback_url(args.base)
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    state_path=REPO/'work/state.json'
    state_before=file_sha(state_path)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'fixture_sha256':file_sha(args.fixture_json),
        'source_code_sha256':file_sha(__file__),'model_alias':args.model,
        'expected_model_config_sha256':args.expected_model_config_sha256,'planned_requests':2*len(fixture['cases']),
        'predeclared_checks':fixture['predeclared_checks'],'fixture_is_human_truth':False,
        'workbench_modified':False,'independent_validation':False,'pixel_accuracy_evaluated':False}
    save(output/'study_plan.json',plan)
    plan_sha=file_sha(output/'study_plan.json')
    summary={'status':'running','pid':os.getpid(),'http_requests_attempted':0,'valid_real_CUDA_responses':0,
        'cases':[],'workbench_modified':False,'pixel_accuracy_evaluated':False,'candidates_reclassified':0}
    save(output/'summary.json',summary)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        health=require_real_service(base,args.model)
        if (health.get('model_config_sha256')!=args.expected_model_config_sha256
                or health.get('quantization')!='nf4' or health.get('image_token_budget')!=256):
            raise ValueError('Unexpected physical model or image budget')
        save(output/'backend_receipt.json',health)
        for case in fixture['cases']:
            row={'id':case['id'],'role':case['role'],'views':{}}
            for kind in ('single','context'):
                if (output/'STOP').exists():raise InterruptedError('Stop requested')
                body,binding=build_payload(fixture,assets,case,kind,args.model)
                summary['http_requests_attempted']+=1
                save(output/'summary.json',summary)
                started=time.monotonic()
                request=urllib.request.Request(base+'/chat/completions',json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                with opener.open(request,timeout=300) as response:raw=json.load(response)
                parsed=validated_response(raw,fixture['categories'],len(case['views'][kind]))
                receipt={'case_id':case['id'],'view':kind,'raw_response':raw,'parsed':parsed,
                    'input_fingerprint_excluding_model_alias':binding,'seconds':round(time.monotonic()-started,3),
                    'valid_real_CUDA_response':True}
                save(output/f"response_{summary['http_requests_attempted']:03d}.json",receipt)
                row['views'][kind]={'parsed':parsed,'seconds':receipt['seconds'],'input_fingerprint':binding}
                summary['valid_real_CUDA_responses']+=1
                save(output/'summary.json',summary)
            row['consensus']=consensus_category(row['views']['single']['parsed']['category'],row['views']['context']['parsed']['category'])
            summary['cases'].append(row)
            save(output/'summary.json',summary)
            print(json.dumps({'case_id':row['id'],'consensus':row['consensus']}),flush=True)
        summary['status']='complete'
    except InterruptedError:
        summary['status']='stopped'
    except Exception as error:
        summary.update(status='failed',error_type=type(error).__name__)
        (output/'local_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        summary['checks']=focus_checks(fixture,summary['cases'],summary['valid_real_CUDA_responses'])
        summary['state_unchanged']=file_sha(state_path)==state_before
        summary['plan_unchanged']=file_sha(output/'study_plan.json')==plan_sha
        summary['fixture_unchanged']=file_sha(args.fixture_json)==plan['fixture_sha256']
        try:checked_focus_fixture(args.fixture_json);summary['image_assets_unchanged']=True
        except Exception:summary['image_assets_unchanged']=False
        summary['all_predeclared_checks_passed']=all(summary['checks'].values()) and all(summary[key] for key in (
            'state_unchanged','plan_unchanged','fixture_unchanged','image_assets_unchanged'))
        summary['finished_utc']=datetime.now(timezone.utc).isoformat()
        save(output/'summary.json',summary)
    print(json.dumps({key:summary[key] for key in ('status','valid_real_CUDA_responses','checks','all_predeclared_checks_passed')}),flush=True)
    return 0 if summary['status']=='complete' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fixture-json','output','expected-model-config-sha256'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--base',default='http://127.0.0.1:18868/v1')
    parser.add_argument('--model',default='labelagent-vlm')
    raise SystemExit(evaluate(parser.parse_args()))
