"""Reuse the frozen source-order evaluator with its source-last baseline."""
import argparse
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'tools')]

def evaluate(args):
    import tools.contour_visibility as contour
    from tools.source_last import source_last_payload
    from tools.evaluate_source_last import evaluate as run
    old=contour.classify_payload
    # source_last_payload already holds its original contour builder reference.
    contour.classify_payload=lambda f,a,c,v,grounding:source_last_payload(f,a,c,v)
    try:return run(args)
    finally:contour.classify_payload=old

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',required=True)
    raise SystemExit(evaluate(p.parse_args()))
