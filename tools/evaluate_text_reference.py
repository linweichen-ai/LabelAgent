"""Text from approved references, single query image, visible reason before class."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.reference_semantics import reason_first_payload
from tools.evaluate_focused_targets import verify_cached_locations
from tools.evaluate_vlm_grounding import validated_response

def validated_reason_first(raw,categories,image_count):
    parsed=validated_response(raw,categories,image_count)
    if raw['local_vlm'].get('requested_image_token_budget') is not None and list(parsed)!=['reason','category']:
        raise ValueError('Actual output did not put visible evidence before category')
    return parsed

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();assert hashlib.sha256(Path(args.plan).read_bytes()).hexdigest()==args.plan_sha256
    plan=json.loads(Path(args.plan).read_bytes());assert plan['reference_mode']=='text_only_before_query'
    verify_cached_locations(plan)
    import tools.evaluate_sam_evidence_recheck as runner
    runner.card_payload=reason_first_payload;runner.validated_response=validated_reason_first
    raise SystemExit(runner.evaluate(args))
