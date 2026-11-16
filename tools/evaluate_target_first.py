"""Bounded, source-bound target-first comparison; never writes annotations.

Every returned response is archived before parsing or applying any decision.
Expected labels are consulted only by aggregate(), never by request builders.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import urllib.error
import urllib.request


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_sha(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())
    os.replace(temp, path)


class ReceiptClient:
    """Exact-payload cache and write-ahead audit, independent of output schemas."""
    def __init__(self, output, progress, before_request, maximum, cached=None, opener=None):
        self.output = Path(output)
        self.progress = progress
        self.before_request = before_request
        self.maximum = maximum
        self.cache = dict(cached or {})
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, body, digest, identity):
        if payload_sha(body) != digest:
            raise ValueError('Payload digest mismatch')
        if digest in self.cache:
            ref = self.cache[digest]
            if sha(ref['path']) != ref['sha256']:
                raise ValueError('Cached response bytes changed')
            receipt = read(ref['path'])
            if receipt['full_payload_sha256'] != digest:
                raise ValueError('Cached response belongs to a different request')
            return receipt['raw_response'], str(ref['path']), True
        if (self.output / 'STOP').exists():
            raise InterruptedError('Own study stop requested')
        self.before_request()
        if self.progress['attempted_requests'] >= self.maximum:
            raise ValueError('Frozen request budget exhausted')
        number = self.progress['attempted_requests'] + 1
        self.progress['attempted_requests'] = number
        identity = dict(identity, number=number, full_payload_sha256=digest)
        # Keep the exact request, including generated grounding evidence, private.
        save(self.output / f'request_{number:03d}_PRIVATE.json', {**identity, 'body': body})
        save(self.output / 'request_inflight.json', identity)
        save(self.output / 'progress.json', self.progress)
        request = urllib.request.Request(
            'http://127.0.0.1:18868/v1/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        started = time.monotonic()
        try:
            with self.opener.open(request, timeout=300) as response:
                wire = response.read()
            # An invalid JSON response remains inspectable too.
            (self.output / f'http_response_{number:03d}_PRIVATE.bin').write_bytes(wire)
            self.progress['http_responses_received'] += 1
            raw = json.loads(wire)
        except Exception as error:
            if isinstance(error, urllib.error.HTTPError):
                (self.output / f'http_error_{number:03d}_PRIVATE.bin').write_bytes(error.read())
            save(self.output / f'attempt_error_{number:03d}.json',
                 {**identity, 'error_type': type(error).__name__,
                  'http_status': getattr(error, 'code', None)})
            save(self.output / 'progress.json', self.progress)
            raise
        path = self.output / f'response_{number:03d}.json'
        # This MUST precede any model/backend/schema/geometry validation.
        save(path, {**identity, 'seconds': round(time.monotonic() - started, 3), 'raw_response': raw})
        self.progress['responses_received'] += 1
        (self.output / 'request_inflight.json').unlink()
        self.cache[digest] = {'path': str(path), 'sha256': sha(path)}
        save(self.output / 'progress.json', self.progress)
        return raw, str(path), False


def load_assets(fixture):
    assets = {}
    refs = [ref for case in fixture['cases'] for refs in case['views'].values() for ref in refs]
    refs += [ref for case in fixture['cases'] for ref in case.get('localization', [])]
    refs += [fixture['guard_image']]
    root = Path(fixture['asset_root']).resolve()
    for ref in refs:
        path = (root / ref['path']).resolve()
        if not path.is_relative_to(root) or sha(path) != ref['sha256']:
            raise ValueError('Asset escapes frozen root or bytes changed')
        mime = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
        value = (path.read_bytes(), mime)
        if ref['path'] in assets and assets[ref['path']] != value:
            raise ValueError('Ambiguous asset name')
        assets[ref['path']] = value
    return assets


def validate_backend(raw, fixture, image_count, guard=False):
    meta = raw['local_vlm']
    expected_tokens = fixture.get('guard_expected_actual_tokens', [238]) if guard else [256] * image_count
    if isinstance(expected_tokens, int):
        expected_tokens = [expected_tokens]
    if (meta.get('backend') != 'transformers_local_cuda'
            or meta.get('model_config_sha256') != fixture['model_config_sha256']
            or meta.get('quantization') != 'nf4'
            or meta.get('image_count') != image_count
            or meta.get('actual_image_token_counts') != expected_tokens
            or meta.get('effective_image_token_budget') != 256):
        raise ValueError('Live model or image encoding differs from frozen comparison')
    return meta


def aggregate(plan, fixture, progress, baseline):
    """Evaluate raw categories separately from conservative localization guards."""
    index = {(row['case_id'], row['view']): row for row in progress['rows']}
    groundings = {row['case_id']: row for row in progress['groundings']}
    baseline_index = {(r['case_id'], r['view']): r for r in baseline['rows'] if r['variant'] == 'candidate'}
    scores = {'baseline': [], 'candidate': [], 'guarded_candidate': []}
    for case in fixture['cases']:
        cid = case['id']
        expected = plan['evaluation_only'][cid]['expected']
        role = plan['evaluation_only'][cid]['role']
        for variant in scores:
            if variant == 'baseline':
                cats = [baseline_index.get((cid, view), {}).get('parsed', {}).get('category')
                        for view in ('overview', 'linked')]
            else:
                cats = [index.get((cid, view), {}).get(
                    'guarded_category' if variant == 'guarded_candidate' else 'raw_category')
                    for view in ('overview', 'linked')]
            scores[variant].append({
                'case_id': cid, 'role': role, 'expected': expected, 'categories': cats,
                'both_expected': cats == [expected, expected],
                'correct_views': sum(c == expected for c in cats),
                'wrong_views': sum(c is not None and c not in (expected, 'uncertain') for c in cats)})
    candidates = scores['candidate']
    positive_roles = {'protected_pallet', 'confirmed_six', 'person_fixture'}
    blank_roles = {'blank_control', 'marked_blank_control'}
    def selected(roles):
        return [r for r in candidates if r['role'] in roles]
    positive_ids = [r['case_id'] for r in selected(positive_roles)]
    blank_ids = [r['case_id'] for r in selected(blank_roles)]
    checks = {
        'complete': progress['status'] == 'complete',
        'two_original_pallets_all_four_views_correct': len(selected({'protected_pallet'})) == 2 and all(r['both_expected'] for r in selected({'protected_pallet'})),
        'confirmed_six_all_twelve_views_correct': len(selected({'confirmed_six'})) == 6 and all(r['both_expected'] for r in selected({'confirmed_six'})),
        'approved_person_control_preserved': len(selected({'person_fixture'})) == 1 and all(r['both_expected'] for r in selected({'person_fixture'})),
        'all_blank_views_abstained': len(blank_ids) == 4 and all(r['both_expected'] for r in selected(blank_roles)),
        'pending_suspect_proposals_abstained': len(selected({'suspected_non_pallet'})) == 1 and all(r['both_expected'] for r in selected({'suspected_non_pallet'})),
        'positive_localization_geometry_consistent': len(positive_ids) == 9 and all(groundings.get(cid, {}).get('localization_passed', False) for cid in positive_ids),
        'blank_localization_empty': len(blank_ids) == 4 and all(groundings.get(cid, {}).get('parsed', {}).get('visibility') == 'empty' for cid in blank_ids),
        'strict_raw_gain': sum(r['correct_views'] for r in candidates) > sum(r['correct_views'] for r in scores['baseline']),
        'sensitive_default_guard_passed': bool(progress['guard']),
    }
    return scores, checks


def evaluate(args):
    # Lazy runtime imports let transport/audit tests run without loading a model.
    repo = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repo), str(repo / 'tools')]
    import psutil
    from tools.target_first import grounding_payload, classify_payload, validated_grounding, geometry_evidence
    from tools.source_companion_v2 import source_companion_payload
    from tools.ascii_target_recheck import ascii_card_payload
    from tools.evaluate_vlm_grounding import validated_response
    from tools.evaluate_confirmed_region_context import check_idle
    from tools.connect_local_vlm import require_real_service

    if sha(args.plan) != args.plan_sha256:
        raise ValueError('Frozen plan digest mismatch')
    plan = read(args.plan)
    if not all(sha(p) == h for p, h in plan['source_bindings'].items()):
        raise ValueError('Frozen source binding changed')
    fixture = read(plan['fixtures']['candidate'])
    baseline_fixture = read(plan['baseline_fixture'])
    baseline = read(plan['baseline_summary'])
    assets = load_assets(fixture)
    baseline_assets = load_assets(baseline_fixture)
    diagnostic_plan = plan.get('diagnostics')
    diagnostic_fixture = read(diagnostic_plan['fixture']) if diagnostic_plan else None
    diagnostic_assets = load_assets(diagnostic_fixture) if diagnostic_fixture else None
    if set(plan['evaluation_only']) != {c['id'] for c in fixture['cases']}:
        raise ValueError('Evaluation does not cover the exact frozen cases')
    check_idle()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    progress = {
        'status': 'loading_model', 'pid': os.getpid(), 'create_time': psutil.Process().create_time(),
        'attempted_requests': 0, 'http_responses_received': 0, 'responses_received': 0,
        'groundings': [], 'rows': [], 'diagnostics': [], 'guard': None, 'default_enabled': False, 'workbench_writes': False}
    save(output / 'progress.json', progress)
    client = ReceiptClient(output, progress, check_idle, plan.get('total_max_new_requests', plan['expected_max_new_requests']), plan.get('cached_responses'))
    try:
        save(output / 'backend_before_PRIVATE.json', require_real_service('http://127.0.0.1:18868/v1', fixture['model']))
        progress['status'] = 'running'
        save(output / 'progress.json', progress)
        for case in fixture['cases']:
            cid = case['id']
            body, digest = grounding_payload(fixture, assets, case)
            if 'localization_jobs' in plan:
                jobs = [job for job in plan['localization_jobs'] if job['case_id'] == cid]
                if len(jobs) != 1 or jobs[0]['payload_sha256'] != digest:
                    raise ValueError('Localization request differs from frozen plan')
            raw, receipt, reused = client.request(body, digest, {'stage': 'grounding', 'case_id': cid, 'view': 'source_only'})
            grounding = validated_grounding(raw)
            meta = validate_backend(raw, fixture, len(case['localization']))
            geometry = geometry_evidence(grounding, fixture['geometry'][cid])
            localized = grounding['visibility'] == 'visible' and geometry.get('point_inside_target') is True
            grounding_row = {
                'case_id': cid, 'parsed': grounding, 'geometry': geometry, 'localization_passed': localized,
                'full_payload_sha256': digest, 'source_receipt': receipt,
                'reused_identical_request': reused, 'actual_image_token_counts': meta['actual_image_token_counts']}
            progress['groundings'].append(grounding_row)
            save(output / 'progress.json', progress)
            print(json.dumps({'stage': 'grounding', 'case_id': cid, 'visibility': grounding['visibility'],
                              'point_inside_target': geometry.get('point_inside_target'), 'cached': reused}), flush=True)
            # Classify every case even after an empty/failed localization. Raw
            # category improvements are therefore measured, never imputed.
            for view in ('overview', 'linked'):
                body, digest = classify_payload(fixture, assets, case, view, grounding)
                identity = {'stage': 'classification', 'case_id': cid, 'view': view,
                            'grounding_receipt': grounding_row['source_receipt'],
                            'grounding_receipt_sha256': sha(grounding_row['source_receipt'])}
                raw, receipt, reused = client.request(body, digest, identity)
                parsed = validated_response(raw, fixture['categories'], len(case['views'][view]))
                meta = validate_backend(raw, fixture, len(case['views'][view]))
                row = {
                    **identity, 'parsed': parsed, 'raw_category': parsed['category'],
                    'guarded_category': parsed['category'] if localized else 'uncertain',
                    'full_payload_sha256': digest, 'source_receipt': receipt,
                    'reused_identical_request': reused, 'actual_image_token_counts': meta['actual_image_token_counts']}
                progress['rows'].append(row)
                save(output / 'progress.json', progress)
                print(json.dumps({'stage': 'classification', 'case_id': cid, 'view': view,
                                  'category': parsed['category'], 'guarded_category': row['guarded_category'], 'cached': reused}), flush=True)
        if diagnostic_plan:
            # The marked comparison is a byte-bound archived answer. Only the
            # matched clean+clean request is new; these are diagnostic, not
            # candidate score rows, and keep the exact earlier prompt/schema.
            for job in diagnostic_plan['jobs']:
                case = next(c for c in diagnostic_fixture['cases'] if c['id'] == job['case_id'])
                view = job['view']
                body, digest = source_companion_payload(diagnostic_fixture, diagnostic_assets, case, view)
                if digest != job['payload_sha256']:
                    raise ValueError('Matched diagnostic request differs from frozen plan')
                ref = job['marked_cached']
                if sha(ref['path']) != ref['sha256']:
                    raise ValueError('Marked diagnostic comparison bytes changed')
                marked_receipt = read(ref['path'])
                original_rows = [r for r in baseline['rows'] if r['variant'] == 'candidate'
                                 and r['case_id'] == case['id'] and r['view'] == view]
                if len(original_rows) != 1 or original_rows[0]['full_payload_sha256'] != marked_receipt['full_payload_sha256']:
                    raise ValueError('Marked diagnostic reference is not the matching baseline request')
                marked_raw = marked_receipt['raw_response']
                marked = validated_response(marked_raw, diagnostic_fixture['categories'], 2)
                validate_backend(marked_raw, diagnostic_fixture, 2)
                raw, receipt, reused = client.request(body, digest, {
                    'stage': 'matched_blank_diagnostic', 'case_id': case['id'], 'view': view})
                parsed = validated_response(raw, diagnostic_fixture['categories'], 2)
                meta = validate_backend(raw, diagnostic_fixture, 2)
                row = {
                    'case_id': case['id'], 'view': view, 'clean_clean_parsed': parsed,
                    'clean_marked_parsed': marked, 'full_payload_sha256': digest,
                    'source_receipt': receipt, 'reused_identical_request': reused,
                    'marked_source_receipt': ref['path'], 'marked_source_receipt_sha256': ref['sha256'],
                    'marked_full_payload_sha256': marked_receipt['full_payload_sha256'],
                    'actual_image_token_counts': meta['actual_image_token_counts'],
                    'included_in_candidate_score': False}
                progress['diagnostics'].append(row)
                save(output / 'progress.json', progress)
                print(json.dumps({'stage': 'matched_blank_diagnostic', 'case_id': case['id'], 'view': view,
                                  'clean_clean': parsed['category'], 'clean_marked': marked['category'], 'cached': reused}), flush=True)
        body, digest = ascii_card_payload(baseline_fixture, baseline_assets, {'id': 'default_budget_guard'}, 'guard', True)
        raw, receipt, reused = client.request(body, digest, {'stage': 'guard', 'case_id': 'default_budget_guard', 'view': 'guard'})
        parsed = validated_response(raw, baseline_fixture['categories'], 1)
        meta = validate_backend(raw, baseline_fixture, 1, guard=True)
        progress['guard'] = {'parsed': parsed, 'source_receipt': receipt, 'full_payload_sha256': digest,
                             'reused_identical_request': reused, 'actual_image_token_counts': meta['actual_image_token_counts']}
        progress['status'] = 'complete'
    except InterruptedError as error:
        progress.update(status='stopped', stop_reason=str(error))
    except Exception as error:
        progress.update(status='failed', error_type=type(error).__name__)
        (output / 'error_PRIVATE.txt').write_text(traceback.format_exc(), encoding='utf8')
    finally:
        scores, checks = aggregate(plan, fixture, progress, baseline)
        checks['source_bindings_unchanged'] = all(Path(p).is_file() and sha(p) == h for p, h in plan['source_bindings'].items())
        checks['plan_unchanged'] = sha(args.plan) == args.plan_sha256
        checks['matched_diagnostics_complete'] = len(progress['diagnostics']) == len(diagnostic_plan['jobs'] if diagnostic_plan else [])
        result = {
            **progress, 'scores': scores, 'checks': checks, 'automated_checks_passed': all(checks.values()),
            'all_checks_passed': False, 'semantic_review_pending': True,
            'plan_sha256': args.plan_sha256, 'finished_utc': datetime.now(timezone.utc).isoformat(),
            'raw_model_categories_never_rewritten': True, 'category_correctness_is_not_localization_truth': True,
            'geometry_guard_is_SAM_consistency_only': True, 'new_SAM_calls': 0,
            'candidate_adopted': False, 'independent_validation': False}
        save(output / 'summary.json', result)
        save(output / 'progress.json', progress)
    print(json.dumps({'status': result['status'], 'requests': result['attempted_requests'], 'checks': checks}), flush=True)
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', required=True)
    raise SystemExit(evaluate(parser.parse_args()))
