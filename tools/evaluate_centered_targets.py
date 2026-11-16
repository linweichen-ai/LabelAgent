"""Verify prior localization before fresh centered source classification."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.evaluate_focused_targets import verify_cached_locations
from tools.evaluate_sam_evidence_recheck import evaluate

def read(p):return json.loads(Path(p).read_bytes())

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();assert hashlib.sha256(Path(args.plan).read_bytes()).hexdigest()==args.plan_sha256
    plan=read(args.plan);boxes=verify_cached_locations(plan)
    for m in read(plan['centered_mapping']):
        assert m['display_metadata']['verified_source_box']==boxes[m['case_id']]
        assert m['display_metadata']['all_target_retained']
    raise SystemExit(evaluate(args))
