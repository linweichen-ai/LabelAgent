"""Matched measured-encoding capacity test of the conditional scene recheck."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import urllib.request

import psutil

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'tools')]
from tools.connect_local_vlm import loopback_url, require_real_service
from tools.evaluate_confirmed_region_context import check_idle
from tools.evaluate_source_availability import score
from tools.evaluate_target_cards import card_payload
from tools.evaluate_vlm_grounding import validated_response
from tools.run_local_agent_acceptance import save
from tools.sam_evidence_recheck import needs_recheck, rechecked_pair


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def without_budget(body):
    result = deepcopy(body); result.pop('image_token_budget', None)
    return result


def content_fingerprint(body):
    return hashlib.sha256(json.dumps(without_budget(body), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def same_actual_encoding(a, b):
    return bool(a.get('actual_image_token_counts') and a.get('image_grid_thw')
                and a['actual_image_token_counts'] == b.get('actual_image_token_counts')
                and a['image_grid_thw'] == b.get('image_grid_thw'))


def evaluate(args):
    path = Path(args.plan).resolve(); assert sha(path) == args.plan_sha256
    plan = read(path); assert all(sha(p) == h for p, h in plan['source_bindings'].items())
    fixture = read(plan['fixture']); previous = read(plan['previous_summary']); original = read(plan['original_fixture'])
    derived = [c['case_id'] for c in previous['cases'] if needs_recheck(
        [c['views'][v]['raw_category'] for v in ('overview', 'linked')],
        (c['views']['overview'].get('support') or {}).get('supported_classes', []))]
    assert derived == [c['id'] for c in fixture['cases']] and len(derived) == 2 and len(original['cases']) == 29
    start, before = read(args.remote_start), read(args.remote_before)
    assert start['comparison_plan_sha256'] == args.plan_sha256 and before['matching_owned_worker_live'] and before['listening_owned']
    assert start['max_image_tokens'] == 1024 and start['lifetime_seconds'] == 900 and start['profile'] == 'shared-4090'
    base = loopback_url(args.base); check_idle()
    assets = {}
    for ref in [r for c in fixture['cases'] for refs in c['views'].values() for r in refs]:
        asset = Path(fixture['asset_root']) / ref['path']; assert sha(asset) == ref['sha256']
        assets[ref['path']] = (asset.read_bytes(), 'image/jpeg')
    health = require_real_service(base, fixture['model'])
    assert health['model_config_sha256'] == plan['remote_model_config_sha256'] and health['quantization'] == 'nf4'
    assert health['image_token_budget'] == 1024 and health.get('last_inference_stats') is None
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    save(output / 'runtime_PRIVATE.json', {'plan_sha256': args.plan_sha256, 'remote_start_sha256': sha(args.remote_start),
        'remote_before_sha256': sha(args.remote_before), 'owned_worker': start['worker'], 'health': health,
        'local_cached_quality_responses_reused': 4, 'new_remote_quality_calls': 4, 'new_local_calls': 0})
    progress = {'status': 'running', 'pid': os.getpid(), 'create_time': psutil.Process().create_time(),
        'attempted_requests': 0, 'valid_remote_responses': 0, 'rows': [], 'remote_stop_required': True}
    save(output / 'progress.json', progress)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        for pair in plan['pairs']:
            check_idle()
            if (output / 'STOP').exists():
                raise InterruptedError('Stop requested')
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(start['started_utc'])).total_seconds()
            assert elapsed < 850, 'Do not start near the bounded worker deadline'
            case = next(c for c in fixture['cases'] if c['id'] == pair['case_id'])
            local_body, local_hash = card_payload(fixture, assets, case, pair['view'])
            local = read(pair['local_response'])
            validated_response(local['raw_response'], fixture['categories'], 1)
            assert sha(pair['local_response']) == pair['local_sha256'] and local_hash == local['full_payload_sha256']
            body = without_budget(local_body); fingerprint = content_fingerprint(body)
            assert fingerprint == content_fingerprint(local_body) == pair['content_fingerprint']
            number = progress['attempted_requests'] + 1
            row = {'number': number, 'case_id': case['id'], 'view': pair['view'], 'content_fingerprint': fingerprint,
                'full_remote_payload_sha256': hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                'local_response_sha256': pair['local_sha256']}
            save(output / 'request_inflight.json', row); progress['attempted_requests'] = number
            save(output / 'progress.json', progress); began = time.monotonic()
            request = urllib.request.Request(base.rstrip('/') + '/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with opener.open(request, timeout=300) as response:
                raw = json.load(response)
            row['seconds'] = round(time.monotonic() - began, 3)
            save(output / f'response_{number:03d}.json', {**row, 'raw_response': raw}); (output / 'request_inflight.json').unlink()
            parsed = validated_response(raw, fixture['categories'], 1); meta = raw['local_vlm']
            assert meta['model_config_sha256'] == plan['remote_model_config_sha256'] and meta['quantization'] == 'nf4'
            assert meta['effective_image_token_budget'] == 1024 and same_actual_encoding(local['raw_response']['local_vlm'], meta)
            assert meta['actual_image_token_counts'] == pair['expected_actual_tokens']
            row.update(parsed=parsed, actual_image_token_counts=meta['actual_image_token_counts'], image_grid_thw=meta['image_grid_thw'])
            progress['rows'].append(row); progress['valid_remote_responses'] += 1; save(output / 'progress.json', progress)
            print(json.dumps({'completed': number, 'planned': 4, 'case_id': case['id'], 'view': pair['view'],
                              'category': parsed['category'], 'actual_tokens': meta['actual_image_token_counts']}), flush=True)
        progress['status'] = 'complete'
    except InterruptedError:
        progress['status'] = 'stopped'
    except Exception as error:
        progress.update(status='failed', error_type=type(error).__name__)
        (output / 'local_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        progress['finished_utc'] = datetime.now(timezone.utc).isoformat(); save(output / 'progress.json', progress)
        indexed = {(r['case_id'], r['view']): r['parsed'] for r in progress['rows']}
        final, outcomes, opposite = [], [], []
        truths = {c['id']: c for c in original['cases']}
        for old in previous['cases']:
            pair = [old['views'][v]['category'] for v in ('overview', 'linked')]
            if old['case_id'] in derived:
                fresh = [indexed.get((old['case_id'], v)) for v in ('overview', 'linked')]
                supported = old['views']['overview']['support']['supported_classes']
                decision = rechecked_pair(pair, supported, [r['category'] for r in fresh]) if all(fresh) else {
                    'categories': pair, 'fresh_pair_matched_unique_SAM_support': False}
                outcomes.append({'case_id': old['case_id'], 'before': pair, 'fresh': fresh, **decision}); pair = decision['categories']
                truth = truths[old['case_id']]
                expected = truth.get('expected_category') if truth['group'] == 'human' else ('pallet' if truth['role'] == 'pallet_fixture' else None)
                for view, r in zip(('overview', 'linked'), fresh):
                    if r and expected and r['category'] not in (expected, 'uncertain'):
                        opposite.append({'case_id': old['case_id'], 'view': view, 'category': r['category']})
            for view, category in zip(('overview', 'linked'), pair):
                final.append({'case_id': old['case_id'], 'view': view, 'parsed': {'category': category, 'reason': 'frozen guarded prior or fresh8B recheck'}})
        scores = score(original, final)
        legacy = sum(r['role'] == 'pallet_fixture' and r['consensus'] == 'pallet' for r in scores['legacy_pairs'])
        checks = {**scores['original_quality_checks'], 'all4_new_responses_valid_real_CUDA_and_measured_encoding_matched': progress['status'] == 'complete' and progress['valid_remote_responses'] == 4,
            'all14_previous_human_correct_retained': not (set(previous['scores']['human_correct_ids']) - set(scores['human_correct_ids'])),
            'zero_wrong_human_consensus': not scores['human_wrong_consensus_ids'], 'no_fresh_explicit_opposite_class_view': not opposite,
            'strict_categorical_gain_vs_previous_guard': scores['human_correct_consensus'] > 14 or legacy > 2,
            'source_code_config_state_bindings_unchanged': all(sha(p) == h for p, h in plan['source_bindings'].items()),
            'plan_unchanged': sha(path) == args.plan_sha256}
        result = {**progress, 'plan_sha256': args.plan_sha256, 'scores': scores, 'outcomes': outcomes,
            'legacy_pallet_correct': legacy, 'fresh_opposite_class_views': opposite, 'checks': checks, 'all_checks_passed': all(checks.values()),
            'new_remote_VLM_calls': progress['attempted_requests'], 'new_local_VLM_calls': 0, 'new_SAM_calls': 0,
            'conditional_hybrid_pipeline_not_full8B_replacement': True, 'adopted': False, 'default_enabled': False,
            'workbench_writes': False, 'independent_validation': False, 'independent_pixel_accuracy': False}
        save(output / 'summary.json', result)
    print(json.dumps({'status': result['status'], 'human_correct': scores['human_correct_consensus'],
        'human_wrong': scores['human_wrong_consensus_ids'], 'legacy_correct': legacy, 'fresh_opposite': opposite, 'checks': checks}), flush=True)
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True); p.add_argument('--plan-sha256', required=True); p.add_argument('--output', required=True)
    p.add_argument('--remote-start', required=True); p.add_argument('--remote-before', required=True)
    p.add_argument('--base', default='http://127.0.0.1:18888/v1')
    raise SystemExit(evaluate(p.parse_args()))
