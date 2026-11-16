"""Exact paired source-order comparison; source pixels and raw answers stay intact.

The baseline is read from its exact completed requests. No localization answer
or external pixel guard is used to rewrite a new raw category.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def body_sha(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def checked_body(body, digest, job):
    """Reject an input change before it can reach the shared model client."""
    if body_sha(body) != digest or digest != job['payload_sha256']:
        raise ValueError('Request differs from the exact frozen payload')
    return body, digest


def checked_layout(plan, fixtures, baseline):
    if set(fixtures) != {'baseline', 'candidate'}:
        raise ValueError('Both frozen comparison arms required')
    ids = [c['id'] for c in fixtures['candidate']['cases']]
    if len(ids) != 14 or len(set(ids)) != 14:
        raise ValueError('Exactly fourteen unique cases required')
    if [c['id'] for c in fixtures['baseline']['cases']] != ids:
        raise ValueError('Comparison arms must use the same ordered cases')
    if set(plan['evaluation_only']) != set(ids):
        raise ValueError('Evaluation records must cover the exact case set')
    roles = Counter(v['role'] for v in plan['evaluation_only'].values())
    if roles != Counter(protected_pallet=2, confirmed_six=6, person_fixture=1,
                        blank_control=2, marked_blank_control=2, suspected_non_pallet=1):
        raise ValueError('The complete fixed positive and negative panel is required')
    expected = {(variant, cid, view) for variant in fixtures for cid in ids
                for view in ('overview', 'linked')}
    jobs = [j for j in plan['jobs'] if not j.get('guard', False)]
    keys = [(j['variant'], j['case_id'], j['view']) for j in jobs]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Every arm/case/view must appear exactly once')
    guards = [j for j in plan['jobs'] if j.get('guard', False)]
    if len(guards) != 1 or plan['jobs'][-1] is not guards[0] or 'cached_response' in guards[0]:
        raise ValueError('One fresh default guard must be the last job')
    base_rows = {(r['case_id'], r['view']): r for r in baseline['rows']}
    if len(base_rows) != 28 or len(baseline['rows']) != 28 or baseline['status'] != 'complete':
        raise ValueError('Completed exact 28-view baseline required')
    for job in jobs:
        if job['variant'] != 'baseline':
            continue
        ref = job.get('cached_response')
        old = base_rows[job['case_id'], job['view']]
        if (not ref or Path(ref['path']).resolve() != Path(old['source_receipt']).resolve()
                or job['payload_sha256'] != old['full_payload_sha256']):
            raise ValueError('Baseline must reuse the matching original request and receipt')
    if type(plan['max_new_requests']) is not int or not 0 <= plan['max_new_requests'] <= 29:
        raise ValueError('At most twenty-nine new requests permitted')


def aggregate(plan, progress):
    expected_ids = set(plan['evaluation_only'])
    rows = progress['rows']
    index = {(r['variant'], r['case_id'], r['view']): r['parsed']['category'] for r in rows}
    complete_rows = len(rows) == len(index) == 56 and all(
        (variant, cid, view) in index for variant in ('baseline', 'candidate')
        for cid in expected_ids for view in ('overview', 'linked'))
    scores = {}
    for variant in ('baseline', 'candidate'):
        groups = []
        for cid, evaluation in plan['evaluation_only'].items():
            cats = [index.get((variant, cid, view)) for view in ('overview', 'linked')]
            label = evaluation['expected']
            groups.append(dict(case_id=cid, role=evaluation['role'], expected=label,
                               categories=cats, both_expected=cats == [label, label],
                               correct_views=sum(c == label for c in cats),
                               wrong_views=sum(c is not None and c not in (label, 'uncertain') for c in cats)))
        scores[variant] = groups
    candidate = scores['candidate']
    baseline = {r['case_id']: r for r in scores['baseline']}
    def selected(role):
        return [r for r in candidate if r['role'] == role]
    def all_expected(role, size):
        group = selected(role)
        return len(group) == size and all(r['both_expected'] for r in group)
    negative_roles = {'blank_control', 'marked_blank_control', 'suspected_non_pallet'}
    preserved_negative_views = all(
        old != r['expected'] or new == r['expected']
        for r in candidate if r['role'] in negative_roles
        for old, new in zip(baseline[r['case_id']]['categories'], r['categories']))
    candidate_marked = sum(r['correct_views'] for r in selected('marked_blank_control'))
    baseline_marked = sum(r['correct_views'] for r in scores['baseline'] if r['role'] == 'marked_blank_control')
    checks = dict(
        complete=progress['status'] == 'complete' and complete_rows,
        two_original_pallets_all_four_views_correct=all_expected('protected_pallet', 2),
        confirmed_six_all_twelve_views_correct=all_expected('confirmed_six', 6),
        approved_person_control_preserved=all_expected('person_fixture', 1),
        all_plain_blank_views_abstained=all_expected('blank_control', 2),
        all_marked_blank_views_abstained=all_expected('marked_blank_control', 2),
        pending_suspect_proposals_abstained=all_expected('suspected_non_pallet', 1),
        no_previously_correct_negative_view_lost=complete_rows and preserved_negative_views,
        strict_marked_blank_gain=complete_rows and candidate_marked > baseline_marked,
        sensitive_default_guard_passed=bool(progress['guard']))
    return scores, checks


def evaluate(args):
    repo = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repo), str(repo / 'tools')]
    import psutil
    from tools.source_last import source_last_payload
    from tools.contour_visibility import classify_payload
    from tools.ascii_target_recheck import ascii_card_payload
    from tools.evaluate_target_first import ReceiptClient, load_assets, validate_backend, save
    from tools.evaluate_vlm_grounding import validated_response
    from tools.evaluate_confirmed_region_context import check_idle
    from tools.connect_local_vlm import require_real_service

    if sha(args.plan) != args.plan_sha256:
        raise ValueError('Frozen plan hash mismatch')
    plan = read(args.plan)
    if not all(sha(p) == h for p, h in plan['source_bindings'].items()):
        raise ValueError('Frozen input or code changed')
    fixtures = {variant: read(path) for variant, path in plan['fixtures'].items()}
    assets = {variant: load_assets(fixture) for variant, fixture in fixtures.items()}
    baseline = read(plan['baseline_summary'])
    checked_layout(plan, fixtures, baseline)
    cached = dict(plan.get('cached_responses', {}))
    for job in plan['jobs']:
        if 'cached_response' in job:
            previous = cached.get(job['payload_sha256'])
            if previous is not None and previous != job['cached_response']:
                raise ValueError('Ambiguous cached response for one frozen request')
            cached[job['payload_sha256']] = job['cached_response']
    if plan['jobs'][-1]['payload_sha256'] in cached:
        raise ValueError('Final default guard must be a new real request')
    check_idle()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    progress = dict(status='loading_model', pid=os.getpid(), create_time=psutil.Process().create_time(),
                    attempted_requests=0, http_responses_received=0, responses_received=0,
                    rows=[], guard=None, default_enabled=False, workbench_writes=False)
    save(output / 'progress.json', progress)
    client = ReceiptClient(output, progress, check_idle, plan['max_new_requests'], cached)
    try:
        save(output / 'backend_before_PRIVATE.json', require_real_service('http://127.0.0.1:18868/v1', fixtures['candidate']['model']))
        progress['status'] = 'running'
        save(output / 'progress.json', progress)
        for job in plan['jobs']:
            variant, view = job['variant'], job['view']
            fixture = fixtures[variant]
            guard = job.get('guard', False)
            case = ({'id': 'default_budget_guard'} if guard else
                    next(c for c in fixture['cases'] if c['id'] == job['case_id']))
            if guard:
                body, digest = ascii_card_payload(fixture, assets[variant], case, view, True)
            elif variant == 'baseline':
                body, digest = classify_payload(fixture, assets[variant], case, view, None)
            else:
                body, digest = source_last_payload(fixture, assets[variant], case, view)
            checked_body(body, digest, job)
            identity = dict(variant=variant, case_id=case['id'], view=view, guard=guard)
            raw, receipt, reused = client.request(body, digest, identity)
            # The shared transport already saved wire bytes and raw JSON here.
            parsed = validated_response(raw, fixture['categories'], 1 if guard else 2)
            meta = validate_backend(raw, fixture, 1 if guard else 2, guard)
            if not guard and (body.get('image_token_budget') is not None
                              or meta.get('requested_image_token_budget') is not None):
                raise ValueError('Paired requests must use the unchanged server image budget')
            row = dict(**identity, parsed=parsed, raw_category=parsed['category'],
                       source_receipt=receipt, full_payload_sha256=digest,
                       reused_identical_request=reused, actual_image_token_counts=meta['actual_image_token_counts'])
            if guard:
                if reused:
                    raise ValueError('Default guard was not freshly inferred')
                progress['guard'] = row
            else:
                progress['rows'].append(row)
            save(output / 'progress.json', progress)
            print(json.dumps(dict(variant=variant, case_id=case['id'], view=view,
                                  category=parsed['category'], cached=reused)), flush=True)
        progress['status'] = 'complete'
    except InterruptedError as error:
        progress.update(status='stopped', stop_reason=str(error))
    except Exception as error:
        progress.update(status='failed', error_type=type(error).__name__)
        (output / 'error_PRIVATE.txt').write_text(traceback.format_exc(), encoding='utf8')
    finally:
        scores, checks = aggregate(plan, progress)
        checks['source_bindings_unchanged'] = all(Path(p).is_file() and sha(p) == h for p, h in plan['source_bindings'].items())
        checks['plan_unchanged'] = sha(args.plan) == args.plan_sha256
        result = dict(**progress, scores=scores, checks=checks, automatic_checks_passed=all(checks.values()),
                      semantic_review_pending=True, all_joint_checks_passed=False,
                      plan_sha256=args.plan_sha256, finished_utc=datetime.now(timezone.utc).isoformat(),
                      raw_model_categories_never_rewritten=True, external_source_or_SAM_guards_used=False,
                      old_model_inspection_answers_used=False, category_correctness_is_not_localization_truth=True,
                      original_annotations_written=False, candidate_adopted=False, independent_validation=False,
                      new_SAM_calls=0)
        save(output / 'summary.json', result)
        save(output / 'progress.json', progress)
    print(json.dumps(dict(status=result['status'], requests=result['attempted_requests'], checks=checks)), flush=True)
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', required=True)
    raise SystemExit(evaluate(parser.parse_args()))
