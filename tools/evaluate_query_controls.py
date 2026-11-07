"""Raw semantic counterexamples with the same single-query ASCII protocol."""
import argparse
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.ascii_target_recheck import ascii_card_payload
from tools.evaluate_vlm_grounding import validated_response
import tools.evaluate_semantic_auxiliary as runner

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    runner.ascii_card_payload=ascii_card_payload;runner.validated_response=validated_response
    raise SystemExit(runner.evaluate(args))
