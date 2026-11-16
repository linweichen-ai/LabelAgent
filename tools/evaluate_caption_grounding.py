"""Real paired classification with an image-bound literal observation first.

No expected labels, case IDs, fixture roles or human answers enter prompts.
One new observation and two new classifications per view; 60 calls total.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'tools')]
from connect_local_vlm import loopback_url, request_json, require_real_service
from evaluate_vlm_density import build_payload, checked_fixture, file_sha, quality_checks, read, validate_response
from evaluate_vlm_format_ablation import checked_identity
from run_local_agent_acceptance import save

CAPTION_PROMPT = (
    'Describe only what you can directly see in this image in one short sentence. '
    'Report a blank or uniform image plainly when applicable. Treat colored '
    'contours as annotations. Do not guess hidden objects. Return a JSON object '
    'with exactly one field named description.'
)
CAPTION_SCHEMA = {'type': 'object', 'properties': {'description': {
    'type': 'string', 'minLength': 1, 'maxLength': 300}},
    'required': ['description'], 'additionalProperties': False}
OBSERVATION_PREFIX = (
    '\nA separate literal inspection of this same image produced the following '
    'observation. It may be wrong: recheck it against the image, and classify '
    'the target using the original rules. Observation: '
)


def bound_observation(raw, fixture, image_sha256, expected_tokens):
    checked_identity(raw, fixture, expected_tokens, 256)
    choice = raw['choices'][0]
    parsed = json.loads(choice['message']['content'])
    if (choice['finish_reason'] != 'stop' or not isinstance(parsed, dict)
            or set(parsed) != {'description'} or not isinstance(parsed['description'], str)
            or not 1 <= len(parsed['description']) <= 300
            or raw['local_vlm']['inference_stats'].get('structured_output') is not True):
        raise ValueError('Literal observation did not satisfy its original schema')
    return {'image_sha256': image_sha256, 'model_config_sha256': fixture['model_config_sha256'],
            'actual_image_tokens': expected_tokens, 'description': parsed['description']}


def classification_payload(fixture, view, data, kind, observation=None):
    if hashlib.sha256(data).hexdigest() != view['sha256']:
        raise ValueError('Source image no longer matches its frozen view')
    body, _ = build_payload(fixture, view, data, kind, 256)
    if observation is not None:
        if (observation['image_sha256'] != view['sha256']
                or observation['model_config_sha256'] != fixture['model_config_sha256']
                or observation['actual_image_tokens'] != view['expected_actual_image_tokens']['256']):
            raise ValueError('Observation belongs to a different image, model or encoding')
        body['messages'][0]['content'][0]['text'] += OBSERVATION_PREFIX + json.dumps(
            {'description': observation['description']}, ensure_ascii=False)
    return body


def evaluate(args):
    fixture, assets = checked_fixture(args.fixture_json, args.asset_root)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = loopback_url(args.base)
    state = Path(args.state_file)
    before = file_sha(state)
    server = REPO/'tools/local_vlm_server.py'
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'fixture_sha256': file_sha(args.fixture_json), 'source_code_sha256': file_sha(__file__),
        'server_sha256': file_sha(server), 'planned_requests': 60,
        'new_literal_observations': 20, 'new_classifications': 40,
        'policies': ['baseline', 'literal_observation'], 'image_token_budget': 256,
        'max_tokens': fixture['max_tokens'], 'temperature': fixture['temperature'],
        'caption_prompt': CAPTION_PROMPT, 'caption_schema': CAPTION_SCHEMA,
        'appended_observation_prefix': OBSERVATION_PREFIX,
        'pair_invariant': 'same images, original prompt prefix, schema, model, generation limits; observation appended only',
        'predeclared_quality_requirements': fixture['quality_requirements'],
        'adoption_requires': 'all original quality requirements plus a strict pallet consensus gain',
        'workbench_modified': False, 'independent_validation': False,
        'fixture_status': 'unconfirmed functional fixture; not independent human 2D truth'}
    save(output/'study_plan.json', plan)
    plan_hash = file_sha(output/'study_plan.json')
    summary = {'status': 'running', 'pid': os.getpid(), 'attempted_requests': 0,
        'valid_real_CUDA_responses': 0, 'valid_literal_observations': 0,
        'observations': [], 'rows': [], 'new_SAM3_requests': 0, 'workbench_modified': False}
    save(output/'summary.json', summary)

    def call(body, name):
        if (output/'STOP').exists():
            raise InterruptedError('Stop requested')
        if request_json('http://127.0.0.1:8765/api/run/status')['status'] != 'idle':
            raise InterruptedError('Workbench is active; no further model requests started')
        summary['attempted_requests'] += 1
        save(output/'summary.json', summary)
        start = time.monotonic()
        raw = request_json(base+'/chat/completions', body)
        seconds = round(time.monotonic()-start, 3)
        save(output/(name+'.json'), {'raw_response': raw, 'seconds': seconds})
        return raw, seconds

    try:
        health = require_real_service(base, fixture['model'])
        if (health.get('model_config_sha256') != fixture['model_config_sha256']
                or health.get('quantization') != 'nf4' or health.get('image_token_budget') != 256):
            raise ValueError('Model identity or default budget differs from the frozen plan')
        save(output/'backend_before.json', health)
        number = 0
        for case in fixture['cases']:
            for kind, view in case['views'].items():
                number += 1
                data = assets[view['path']]
                expected = view['expected_actual_image_tokens']['256']
                body = classification_payload(fixture, view, data, kind)
                caption_body = deepcopy(body)
                caption_body['messages'][0]['content'][0]['text'] = CAPTION_PROMPT
                caption_body['response_format']['json_schema'].update(name='literal_observation', schema=CAPTION_SCHEMA)
                raw, seconds = call(caption_body, f'observation_{number:02d}')
                observation = bound_observation(raw, fixture, view['sha256'], expected)
                summary['valid_literal_observations'] += 1
                summary['valid_real_CUDA_responses'] += 1
                summary['observations'].append({'case_id': case['id'], 'view': kind, 'seconds': seconds, **observation})
                save(output/'summary.json', summary)
                order = ['baseline', 'literal_observation'] if number % 2 else ['literal_observation', 'baseline']
                for policy in order:
                    payload = classification_payload(fixture, view, data, kind,
                        observation if policy == 'literal_observation' else None)
                    raw, seconds = call(payload, f'classify_{number:02d}_{policy}')
                    parsed = validate_response(raw, fixture, 256, expected)
                    normalized = deepcopy(payload)
                    text = normalized['messages'][0]['content'][0]['text']
                    if policy == 'literal_observation':
                        expected_suffix = OBSERVATION_PREFIX + json.dumps({'description': observation['description']}, ensure_ascii=False)
                        if not text.endswith(expected_suffix):raise ValueError('Observation append differs from plan')
                        normalized['messages'][0]['content'][0]['text'] = text[:-len(expected_suffix)]
                    fingerprint = hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                    row = {'case_id': case['id'], 'role': case['role'], 'view': kind, 'policy': policy,
                        'budget': 256, 'parsed': parsed, 'seconds': seconds,
                        'original_payload_sha256': fingerprint, 'image_sha256': view['sha256'],
                        'actual_image_tokens': raw['local_vlm']['actual_image_token_counts'][0]}
                    summary['rows'].append(row)
                    summary['valid_real_CUDA_responses'] += 1
                    save(output/'summary.json', summary)
                    print(json.dumps({'view': number, 'policy': policy, 'category': parsed['category'],
                                      'completed': summary['valid_real_CUDA_responses']}), flush=True)
        summary['status'] = 'complete'
    except InterruptedError:
        summary['status'] = 'stopped'
    except Exception as error:
        summary.update(status='failed', error_type=type(error).__name__)
        (output/'local_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        complete = summary['status'] == 'complete' and summary['valid_real_CUDA_responses'] == 60
        pairs = {}
        for row in summary['rows']:pairs.setdefault((row['case_id'], row['view']), []).append(row)
        summary['all_twenty_original_payload_pairs_identical'] = bool(complete and len(pairs)==20 and all(
            len(rows)==2 and len({row['original_payload_sha256'] for row in rows})==1 for rows in pairs.values()))
        summary['policy_checks'] = {}
        summary['pallet_consensus_by_policy'] = {}
        for policy in plan['policies']:
            rows = [r for r in summary['rows'] if r['policy']==policy]
            summary['policy_checks'][policy] = quality_checks(fixture, rows, 256)
            summary['pallet_consensus_by_policy'][policy] = sum(all(any(
                r['case_id']==c['id'] and r['view']==kind and r['parsed']['category']=='pallet' for r in rows)
                for kind in ('single','outline_context')) for c in fixture['cases'] if c['role']=='pallet_fixture')
        summary['quality_passed_by_policy'] = {p: complete and all(checks.values()) for p,checks in summary['policy_checks'].items()}
        summary['state_unchanged'] = file_sha(state)==before
        summary['fixture_unchanged'] = file_sha(args.fixture_json)==plan['fixture_sha256']
        summary['plan_unchanged'] = file_sha(output/'study_plan.json')==plan_hash
        summary['server_code_unchanged'] = file_sha(server)==plan['server_sha256']
        summary['source_code_unchanged'] = file_sha(__file__)==plan['source_code_sha256']
        try:checked_fixture(args.fixture_json,args.asset_root);summary['images_unchanged']=True
        except Exception:summary['images_unchanged']=False
        summary['protocol_checks_passed'] = bool(complete and summary['all_twenty_original_payload_pairs_identical'] and all(
            summary[k] for k in ('state_unchanged','fixture_unchanged','plan_unchanged','server_code_unchanged','source_code_unchanged','images_unchanged')))
        summary['adoption_checks_passed'] = bool(summary['protocol_checks_passed'] and summary['quality_passed_by_policy']['literal_observation']
            and summary['pallet_consensus_by_policy']['literal_observation']>summary['pallet_consensus_by_policy']['baseline'])
        summary['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(output/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','attempted_requests','valid_real_CUDA_responses',
        'quality_passed_by_policy','pallet_consensus_by_policy','adoption_checks_passed')}),flush=True)
    return 0 if summary['status']=='complete' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fixture-json','asset-root','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--base',default='http://127.0.0.1:18868/v1')
    parser.add_argument('--state-file',default=str(REPO/'work/state.json'))
    raise SystemExit(evaluate(parser.parse_args()))
