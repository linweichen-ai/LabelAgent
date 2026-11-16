"""Fixed 18-case contour-alpha comparison; keep all raw categories separate."""
import argparse
from collections import Counter
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'tools')]


def layout(plan, fixtures, baseline):
    ids = [c['id'] for c in fixtures['candidate']['cases']]
    assert len(ids) == len(set(ids)) == 18
    assert ids == [c['id'] for c in fixtures['baseline']['cases']]
    assert set(ids) == set(plan['evaluation_only'])
    assert Counter(r['role'] for r in plan['evaluation_only'].values()) == Counter(
        protected_pallet=2, confirmed_six=6, person_fixture=1, blank_control=2,
        marked_blank_control=2, stress_negative=4, suspected_non_pallet=1)
    jobs = [j for j in plan['jobs'] if not j.get('guard')]
    assert len(jobs) == len({(j['variant'], j['case_id'], j['view']) for j in jobs}) == 72
    assert len(plan['jobs']) == 73 and plan['jobs'][-1].get('guard')
    assert plan['max_new_requests'] == 33
    assert baseline['status'] == 'complete' and len(baseline['rows']) == 36
    prior = {(r['case_id'], r['view']): r for r in baseline['rows']}
    for j in jobs:
        if j['variant'] == 'baseline':
            assert j['payload_sha256'] == prior[j['case_id'], j['view']]['full_payload_sha256']
            assert j['cached_response']['path'] == prior[j['case_id'], j['view']]['source_receipt']
        elif plan['evaluation_only'][j['case_id']]['role'] == 'blank_control':
            assert 'cached_response' in j
        else:
            assert 'cached_response' not in j
    assert 'cached_response' not in plan['jobs'][-1]


def aggregate(plan, progress):
    idx = {(r['variant'], r['case_id'], r['view']): r['parsed']['category'] for r in progress['rows']}
    scores = {}
    for arm in ('baseline', 'candidate'):
        scores[arm] = [dict(case_id=cid, **ev,
            categories=[idx.get((arm, cid, view)) for view in ('overview', 'linked')],
            correct_views=sum(idx.get((arm, cid, view)) == ev['expected'] for view in ('overview', 'linked')))
            for cid, ev in plan['evaluation_only'].items()]
    groups = {arm: {} for arm in scores}
    for arm, rows in scores.items():
        for r in rows:
            g = groups[arm].setdefault(r['role'], dict(views=0, correct=0))
            g['views'] += 2
            g['correct'] += r['correct_views']
    def passed(role, total):
        return groups['candidate'][role] == dict(views=total, correct=total)
    complete = len(progress['rows']) == len(idx) == 72 and progress['status'] == 'complete'
    preserved = all(idx.get(('baseline', cid, v)) != e['expected'] or idx.get(('candidate', cid, v)) == e['expected']
                    for cid, e in plan['evaluation_only'].items() for v in ('overview', 'linked'))
    checks = dict(complete=complete,
        confirmed_six_12_of_12=passed('confirmed_six', 12), old_pallets_4_of_4=passed('protected_pallet', 4),
        person_control_2_of_2=passed('person_fixture', 2), plain_blanks_4_of_4=passed('blank_control', 4),
        original_marked_blanks_4_of_4=passed('marked_blank_control', 4), stress_blanks_8_of_8=passed('stress_negative', 8),
        suspect_conservative_abstention_2_of_2=passed('suspected_non_pallet', 2),
        no_correct_baseline_view_lost=complete and preserved,
        strict_stress_gain=groups['candidate']['stress_negative']['correct'] > groups['baseline']['stress_negative']['correct'],
        default_budget_guard_passed=bool(progress['guard']))
    scores['groups'] = groups
    return scores, checks


def evaluate(args):
    import tools.contour_visibility as contour
    import tools.evaluate_source_last as runner
    from tools.source_last import source_last_payload
    contour.classify_payload = lambda f, a, c, v, grounding: source_last_payload(f, a, c, v)
    runner.checked_layout = layout
    runner.aggregate = aggregate
    return runner.evaluate(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--output', required=True)
    raise SystemExit(evaluate(p.parse_args()))
