"""Paired candidate-review comparison; refuses mismatched cases or input grids."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True);p.add_argument('--out',required=True);p.add_argument('--allow-density-change',action='store_true');a=p.parse_args()
    b=Path(a.baseline);c=Path(a.candidate)
    bm=json.loads((b/'metrics.json').read_text());cm=json.loads((c/'metrics.json').read_text())
    br=[json.loads(s) for s in (b/'responses.jsonl').read_text().splitlines() if s.strip()]
    cr=[json.loads(s) for s in (c/'responses.jsonl').read_text().splitlines() if s.strip()]
    if bm['records_sha256']!=cm['records_sha256']:raise ValueError('Different reviewed cases')
    if len(br)!=len(cr) or [r['sample_id'] for r in br]!=[r['sample_id'] for r in cr]:raise ValueError('Different/missing cases')
    if not a.allow_density_change and any(x['image_grid_thw']!=y['image_grid_thw'] for x,y in zip(br,cr)):raise ValueError('Different actual image grids')
    delta=cm['balanced_accuracy_over_present_decisions']-bm['balanced_accuracy_over_present_decisions']
    checks={'balanced_accuracy_gain_at_least_10pp':delta>=.1-1e-12,
        'all_candidate_responses_strict_json':cm['valid_json']==cm['cases'],
        'no_destructive_action_on_keep':cm['destructive_action_on_keep']==0,
        'no_invalid_or_noop_relabel':cm['invalid_or_noop_relabel']==0}
    result={'checks':checks,'screening_passed':all(checks.values()),'balanced_accuracy_delta':delta,
        'baseline_wrong_candidate_right':sum(not x['correct'] and y['correct'] for x,y in zip(br,cr)),
        'baseline_right_candidate_wrong':sum(x['correct'] and not y['correct'] for x,y in zip(br,cr)),
        'cases':len(br),'density_changed':a.allow_density_change,
        'independent_test':bm['independent_test'] and cm['independent_test'],
        'full_agent_verified':False,'default_adopted':False,
        'caveat':'Pilot screening only; no statistical generalization or full-Agent claim'}
    out=Path(a.out)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
