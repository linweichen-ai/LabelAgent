import argparse
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.cautious_semantics import cautious_payload
from tools.evaluate_text_reference import validated_reason_first
import tools.evaluate_observable_semantics as runner

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    runner.observable_payload=cautious_payload;runner.validated_observable=validated_reason_first
    raise SystemExit(runner.evaluate(args))
