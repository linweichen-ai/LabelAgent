"""Same frozen model requests, resilient process-local progress publication."""
import argparse,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'tools')]

def evaluate(args):
    import tools.evaluate_target_first as transport
    from tools.atomic_progress import save_json
    from tools.evaluate_scene_absence import evaluate as evaluate_original
    # Only this process changes its JSON writer; live services are untouched.
    transport.save=save_json
    return evaluate_original(args)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',required=True)
    raise SystemExit(evaluate(p.parse_args()))
