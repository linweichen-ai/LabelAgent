"""Frozen 18-case source-evidence instruction comparison, raw replies retained."""
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
    expected = {(arm, cid, v) for arm in fixtures for cid in ids for v in ('overview', 'linked')}
    assert len(jobs) == 72 and {(j['variant'], j['case_id'], j['view']) for j in jobs} == expected
    assert len(plan['jobs']) == 73 and plan['jobs'][-1].get('guard')
    assert plan['max_new_requests'] == 37
    assert baseline['status'] == 'complete' and len(baseline['rows']) == 36
    prior = {(r['case_id'], r['view']): r for r in baseline['rows']}
    for j in jobs:
        if j['variant'] == 'baseline':
            assert j['payload_sha256'] == prior[j['case_id'], j['view']]['full_payload_sha256']
            assert j['cached_response']['path'] == prior[j['case_id'], j['view']]['source_receipt']
        else:
            assert 'cached_response' not in j
    assert 'cached_response' not in plan['jobs'][-1]


def evaluate(args):
    import tools.contour_visibility as contour
    import tools.source_last as source
    import tools.evaluate_source_last as runner
    from tools.source_last import source_last_payload as baseline_payload
    from tools.source_evidence import source_evidence_payload
    from tools.evaluate_contour_alpha import aggregate
    # Both functions capture the original source-last builder before mutation.
    # These assignments affect this evaluator process only, not the service.
    contour.classify_payload = lambda f, a, c, v, grounding: baseline_payload(f, a, c, v)
    source.source_last_payload = source_evidence_payload
    runner.checked_layout = layout
    runner.aggregate = aggregate
    return runner.evaluate(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--output', required=True)
    raise SystemExit(evaluate(p.parse_args()))
