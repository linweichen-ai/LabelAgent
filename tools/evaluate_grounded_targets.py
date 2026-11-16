"""Verify target location, optionally refocus once, then classify the same target."""
import argparse
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
from tools.connect_local_vlm import require_real_service
from tools.evaluate_confirmed_region_context import check_idle
from tools.evaluate_source_availability import score
from tools.evaluate_target_cards import card_payload
from tools.evaluate_vlm_grounding import validated_response
from tools.grounded_target_views import localization_payload, validated_location
from tools.run_local_agent_acceptance import save
from tools.sam_evidence_recheck import needs_recheck, rechecked_pair


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def evaluate(args):
    plan_path = Path(args.plan).resolve(); assert sha(plan_path) == args.plan_sha256
    plan = read(plan_path); assert all(sha(p) == h for p, h in plan['source_bindings'].items())
    f = read(plan['fixture']); previous = read(plan['previous_summary']); original = read(plan['original_fixture'])
    derived = [c['case_id'] for c in previous['cases'] if needs_recheck(
        [c['views'][v]['raw_category'] for v in ('overview', 'linked')], (c['views']['overview'].get('support') or {}).get('supported_classes', []))]
    assert derived == [c['id'] for c in f['cases']] and len(derived) == 2 and len(original['cases']) == 29
    assets = {}
    for c in f['cases']:
        refs = list(c['localization_images'].values()) + [r for branch in c['branches'].values() for group in branch['views'].values() for r in group]
        for ref in refs:
            p = Path(f['asset_root']) / ref['path']; assert sha(p) == ref['sha256']
            assets[ref['path']] = (p.read_bytes(), 'image/jpeg')
    ref = f['guard_image']; p = Path(f['asset_root']) / ref['path']; assert sha(p) == ref['sha256']; assets[ref['path']] = (p.read_bytes(), 'image/jpeg')
    check_idle(); output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    progress = {'status': 'starting', 'pid': os.getpid(), 'create_time': psutil.Process().create_time(),
        'attempted_requests': 0, 'valid_responses': 0, 'rows': [], 'target_results': [], 'guard': None,
        'planned_max_requests': 9, 'new_SAM_calls': 0, 'workbench_writes': False, 'default_enabled': False}
    save(output / 'progress.json', progress)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def call(case, stage, view, global_cell=None):
        check_idle()
        if (output / 'STOP').exists(): raise InterruptedError('Stop requested')
        if stage == 'localization':
            local_f = {**f, 'localization_prompt': f['localization_prompt'] if view == 'whole' else f['localization_focus_prompt']}
            local_case = {**case, 'localization_image': case['localization_images'][view]}
            body = localization_payload(local_f, assets[local_case['localization_image']['path']], local_case); expected = [1024]
        elif stage == 'classification':
            assert global_cell in case['branches']
            body, _ = card_payload(f, assets, {'views': case['branches'][global_cell]['views']}, view); expected = [f['image_token_budgets'][view]]
        else:
            body, _ = card_payload(f, assets, case, 'guard', True); expected = f['guard_expected_actual_tokens']
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        number = progress['attempted_requests'] + 1
        row = {'number': number, 'case_id': case['id'], 'stage': stage, 'view': view, 'selected_global_cell': global_cell, 'full_payload_sha256': digest}
        save(output / 'request_inflight.json', row); progress['attempted_requests'] = number; save(output / 'progress.json', progress)
        started = time.monotonic(); request = urllib.request.Request('http://127.0.0.1:18868/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with opener.open(request, timeout=300) as response: raw = json.load(response)
        row['seconds'] = round(time.monotonic() - started, 3)
        save(output / f'response_{number:03d}.json', {**row, 'raw_response': raw}); (output / 'request_inflight.json').unlink()
        parsed = validated_location(raw) if stage == 'localization' else validated_response(raw, f['categories'], 1)
        meta = raw['local_vlm']
        assert meta['model_config_sha256'] == f['model_config_sha256'] and meta['quantization'] == 'nf4'
        assert meta['actual_image_token_counts'] == expected
        assert meta['effective_image_token_budget'] == (256 if stage == 'guard' else expected[0])
        row.update(parsed=parsed, actual_image_token_counts=expected, image_grid_thw=meta['image_grid_thw'])
        progress['valid_responses'] += 1
        if stage == 'guard': progress['guard'] = row
        else: progress['rows'].append(row)
        save(output / 'progress.json', progress)
        print(json.dumps({'completed': number, 'stage': stage, 'case_id': case['id'], 'view': view, 'answer': parsed.get('cell', parsed.get('category'))}), flush=True)
        return row
    try:
        health = require_real_service('http://127.0.0.1:18868/v1', f['model'])
        assert health['model_config_sha256'] == f['model_config_sha256'] and health['image_token_budget'] == 256
        save(output / 'backend_before_PRIVATE.json', health); progress['status'] = 'running'
        for case in f['cases']:
            result = {'case_id': case['id'], 'locations': [], 'location_verified': False, 'classifications': [], 'global_cell': None}
            progress['target_results'].append(result)
            first = call(case, 'localization', 'whole'); result['locations'].append(first)
            if first['parsed']['cell'] in case['acceptable_whole_cells']:
                result.update(location_verified=True, global_cell=first['parsed']['cell'])
            else:
                second = call(case, 'localization', 'focus'); result['locations'].append(second)
                if second['parsed']['cell'] in case['acceptable_focus_cells']:
                    result.update(location_verified=True, global_cell=case['fallback_branch'])
            if result['location_verified']:
                for view in ('overview', 'linked'):
                    result['classifications'].append(call(case, 'classification', view, result['global_cell']))
            result['classification_skipped_due_to_location'] = not result['location_verified']
            save(output / 'progress.json', progress)
        call({'id': 'default_budget_guard'}, 'guard', 'guard'); progress['status'] = 'complete'
    except InterruptedError:
        progress['status'] = 'stopped'
    except Exception as error:
        progress.update(status='failed', error_type=type(error).__name__)
        (output / 'local_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        progress['finished_utc'] = datetime.now(timezone.utc).isoformat(); save(output / 'progress.json', progress)
        targets = {r['case_id']: r for r in progress['target_results']}
        truths = {c['id']: c for c in original['cases']}
        final, outcomes, opposite = [], [], []
        for old in previous['cases']:
            pair = [old['views'][v]['category'] for v in ('overview', 'linked')]
            if old['case_id'] in derived:
                r = targets.get(old['case_id'], {}); classifications = r.get('classifications', [])
                fresh = [item['parsed'] for item in classifications]
                supported = old['views']['overview']['support']['supported_classes']
                decision = rechecked_pair(pair, supported, [item['category'] for item in fresh]) if r.get('location_verified') and len(fresh) == 2 else {
                    'categories': pair, 'fresh_pair_matched_unique_SAM_support': False}
                outcomes.append({'case_id': old['case_id'], 'before': pair, 'location_verified': r.get('location_verified', False), 'fresh': fresh, **decision}); pair = decision['categories']
                truth = truths[old['case_id']]; expected = truth.get('expected_category') if truth['group'] == 'human' else ('pallet' if truth['role'] == 'pallet_fixture' else None)
                for item in classifications:
                    if expected and item['parsed']['category'] not in (expected, 'uncertain'):
                        opposite.append({'case_id': old['case_id'], 'view': item['view'], 'category': item['parsed']['category']})
            for view, category in zip(('overview', 'linked'), pair):
                final.append({'case_id': old['case_id'], 'view': view, 'parsed': {'category': category, 'reason': 'prior or location-verified fresh pair'}})
        scores = score(original, final); legacy = sum(r['role'] == 'pallet_fixture' and r['consensus'] == 'pallet' for r in scores['legacy_pairs'])
        localized = sum(r.get('location_verified', False) for r in progress['target_results'])
        class_count = sum(r['stage'] == 'classification' for r in progress['rows'])
        location_count = sum(r['stage'] == 'localization' for r in progress['rows'])
        checks = {**scores['original_quality_checks'], 'all_targets_location_verified': localized == 2,
            'classification_only_after_verified_location': all(not r['classifications'] or r['location_verified'] for r in progress['target_results']),
            'all_conditional_calls_valid_real_CUDA': progress['status'] == 'complete' and progress['attempted_requests'] == progress['valid_responses'] == location_count + class_count + 1 and class_count == 2 * localized,
            'all14_previous_human_correct_retained': not (set(previous['scores']['human_correct_ids']) - set(scores['human_correct_ids'])),
            'zero_wrong_human_consensus': not scores['human_wrong_consensus_ids'], 'no_fresh_explicit_opposite_class_view': not opposite,
            'strict_categorical_gain': scores['human_correct_consensus'] > 14 or legacy > 2,
            'sensitive_default_guard_passed': bool(progress['guard'] and progress['guard']['actual_image_token_counts'] == [238]),
            'source_code_config_state_bindings_unchanged': all(sha(p) == h for p, h in plan['source_bindings'].items()), 'plan_unchanged': sha(plan_path) == args.plan_sha256}
        summary = {**progress, 'plan_sha256': args.plan_sha256, 'scores': scores, 'outcomes': outcomes, 'legacy_pallet_correct': legacy,
            'localized_targets': localized, 'localization_calls': location_count, 'classification_calls': class_count, 'fresh_opposite_views': opposite,
            'checks': checks, 'all_checks_passed': all(checks.values()), 'adopted': False, 'independent_validation': False, 'independent_pixel_accuracy': False}
        save(output / 'summary.json', summary)
    print(json.dumps({'status': summary['status'], 'localized_targets': localized, 'localization_calls': location_count, 'classification_calls': class_count,
        'human_correct': scores['human_correct_consensus'], 'legacy_correct': legacy, 'fresh_opposite': opposite, 'checks': checks}), flush=True)
    return 0 if summary['status'] == 'complete' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--plan', required=True); p.add_argument('--plan-sha256', required=True); p.add_argument('--output', required=True)
    raise SystemExit(evaluate(p.parse_args()))
