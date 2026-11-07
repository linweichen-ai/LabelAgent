"""Expand the frozen development panel with controlled synthetic negatives."""
import argparse
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'tools')]

def layout(plan,fixtures,baseline):
    ids=[c['id'] for c in fixtures['candidate']['cases']]
    assert len(ids)==len(set(ids))==18
    assert ids==[c['id'] for c in fixtures['baseline']['cases']]
    assert set(ids)==set(plan['evaluation_only'])
    assert len(plan['original_ids'])==14 and len(set(ids)-set(plan['original_ids']))==4
    jobs=[j for j in plan['jobs'] if not j.get('guard')]
    assert len(jobs)==72 and len({(j['variant'],j['case_id'],j['view']) for j in jobs})==72
    assert len(plan['jobs'])==73 and plan['jobs'][-1].get('guard') and 'cached_response' not in plan['jobs'][-1]
    assert plan['max_new_requests']==17
    previous={(r['case_id'],r['view']):r for r in baseline['rows']}
    assert len(previous)==28 and baseline['status']=='complete'
    for j in jobs:
        if j['case_id'] in plan['original_ids']:
            assert 'cached_response' in j
            if j['variant']=='baseline':assert j['payload_sha256']==previous[j['case_id'],j['view']]['full_payload_sha256']
        else:assert plan['evaluation_only'][j['case_id']]==dict(expected='uncertain',role='stress_negative') and 'cached_response' not in j

def evaluate(args):
    import tools.contour_visibility as contour
    import tools.evaluate_source_last as runner
    from tools.source_last import source_last_payload
    original_aggregate=runner.aggregate
    def aggregate(plan,progress):
        ids=set(plan['original_ids']);p={**plan,'evaluation_only':{k:v for k,v in plan['evaluation_only'].items() if k in ids}}
        old={**progress,'rows':[r for r in progress['rows'] if r['case_id'] in ids]}
        scores,checks=original_aggregate(p,old)
        stress={variant:[r for r in progress['rows'] if r['variant']==variant and r['case_id'] not in ids] for variant in ('baseline','candidate')}
        scores['stress']={k:[dict(case_id=r['case_id'],view=r['view'],category=r['parsed']['category']) for r in rows] for k,rows in stress.items()}
        checks['all_new_negative_views_received']=all(len(r)==8 for r in stress.values())
        checks['all_new_negative_candidate_views_abstained']=len(stress['candidate'])==8 and all(r['parsed']['category']=='uncertain' for r in stress['candidate'])
        b={(r['case_id'],r['view']):r['parsed']['category'] for r in stress['baseline']}
        checks['no_new_negative_correct_baseline_view_lost']=len(stress['candidate'])==8 and all(b.get((r['case_id'],r['view']))!='uncertain' or r['parsed']['category']=='uncertain' for r in stress['candidate'])
        return scores,checks
    contour.classify_payload=lambda f,a,c,v,grounding:source_last_payload(f,a,c,v)
    runner.checked_layout=layout;runner.aggregate=aggregate
    return runner.evaluate(args)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',required=True);raise SystemExit(evaluate(p.parse_args()))
