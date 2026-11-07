"""Fixed 1024-token comparison on every original conflict-triggered target."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.evaluate_focused_targets import verify_cached_locations
from tools.resolution_recheck import resolution_payload
import tools.evaluate_sam_evidence_recheck as runner

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();assert hashlib.sha256(Path(args.plan).read_bytes()).hexdigest()==args.plan_sha256
    plan=json.loads(Path(args.plan).read_bytes());boxes=verify_cached_locations(plan)
    assert all(m['display_metadata']['verified_source_box']==boxes[m['case_id']] for m in json.loads(Path(plan['centered_mapping']).read_bytes()))
    assert json.loads(Path(plan['fixture']).read_bytes())['image_token_budgets']=={'overview':1024,'linked':1024}
    runner.card_payload=resolution_payload
    raise SystemExit(runner.evaluate(args))
