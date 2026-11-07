"""Bound free-text evidence to ASCII, keeping category decisions unconstrained."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.evaluate_target_cards import card_payload
from tools.evaluate_focused_targets import verify_cached_locations
import tools.evaluate_sam_evidence_recheck as runner

def ascii_card_payload(fixture,assets,case,view,guard=False):
    body,_=card_payload(fixture,assets,case,view,guard)
    if not guard:
        # The installed parser rejects simultaneous pattern + min/maxLength.
        # The equivalent regex preserves the same 1..100-character bound.
        body['response_format']['json_schema']['schema']['properties']['reason']={'type':'string','pattern':'[ -~]{1,100}'}
    digest=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return body,digest

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();assert hashlib.sha256(Path(args.plan).read_bytes()).hexdigest()==args.plan_sha256
    plan=json.loads(Path(args.plan).read_bytes());assert plan['reason_pattern']=='[ -~]{1,100}'
    boxes=verify_cached_locations(plan)
    mapping=json.loads(Path(plan['centered_mapping']).read_bytes())
    assert all(m['display_metadata']['verified_source_box']==boxes[m['case_id']] and m['display_metadata']['all_target_retained'] for m in mapping)
    # Explicit request-builder composition; the frozen runner file is unchanged.
    runner.card_payload=ascii_card_payload
    raise SystemExit(runner.evaluate(args))
