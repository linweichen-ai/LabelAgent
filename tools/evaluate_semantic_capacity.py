"""Matched image/prompt/ASCII-schema comparison against a bounded larger VLM."""
import argparse
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.ascii_target_recheck import ascii_card_payload
import tools.evaluate_scene_capacity as runner

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--base',required=True)
    p.add_argument('--remote-start',required=True);p.add_argument('--remote-before',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();runner.card_payload=ascii_card_payload
    raise SystemExit(runner.evaluate(args))
