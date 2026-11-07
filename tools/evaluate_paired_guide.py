"""Two freshly inferred matched arms; all original quality gates remain."""
from pathlib import Path
from collections import Counter
import argparse,sys
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'tools')]


def layout(plan,fixtures,reference):
    ids=[c['id'] for c in fixtures['candidate']['cases']]
    assert len(ids)==len(set(ids))==18 and ids==[c['id'] for c in fixtures['baseline']['cases']]
    assert set(ids)==set(plan['evaluation_only'])
    assert Counter(e['role'] for e in plan['evaluation_only'].values())==Counter(protected_pallet=2,confirmed_six=6,person_fixture=1,blank_control=2,marked_blank_control=2,stress_negative=4,suspected_non_pallet=1)
    assert fixtures['baseline']['painted_locator'] and not fixtures['candidate']['painted_locator']
    jobs=[j for j in plan['jobs'] if not j.get('guard')]
    assert len(jobs)==72 and {(j['variant'],j['case_id'],j['view']) for j in jobs}=={(arm,cid,v) for arm in fixtures for cid in ids for v in ('overview','linked')}
    assert len(plan['jobs'])==73 and plan['jobs'][-1].get('guard')
    assert plan['max_new_requests']==len({j['payload_sha256'] for j in plan['jobs']})<=73
    assert not any('cached_response' in j for j in plan['jobs'])
    # This is historical reference only; it never satisfies a fresh guide arm.
    assert reference['status']=='complete' and len(reference['rows'])==36


def evaluate(args):
    import tools.contour_visibility as contour
    import tools.source_last as source
    import tools.evaluate_source_last as runner
    import tools.evaluate_vlm_grounding as grounding
    import tools.evaluate_target_first as transport
    from tools.paired_guide import paired_guide_payload
    from tools.native_context import validate_response,validate_backend
    from tools.evaluate_contour_alpha import aggregate
    contour.classify_payload=lambda f,a,c,v,grounding:paired_guide_payload(f,a,c,v)
    source.source_last_payload=paired_guide_payload
    grounding.validated_response=validate_response;transport.validate_backend=validate_backend
    runner.checked_layout=layout;runner.aggregate=aggregate
    return runner.evaluate(args)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',required=True)
    raise SystemExit(evaluate(p.parse_args()))
