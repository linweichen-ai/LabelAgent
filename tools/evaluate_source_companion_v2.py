import argparse
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'tools')]
from tools.source_companion_v2 import source_companion_payload
import tools.evaluate_source_companion as runner
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',required=True)
    runner.source_companion_payload=source_companion_payload
    raise SystemExit(runner.evaluate(p.parse_args()))
