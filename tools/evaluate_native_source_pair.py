"""Complete protected panel, unchanged greedy output, resilient private receipts."""
import argparse,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'tools')]

def evaluate(args):
    import tools.contour_visibility as contour
    import tools.source_last as source
    import tools.evaluate_source_last as runner
    import tools.evaluate_target_first as transport
    from tools.source_last import source_last_payload as baseline_payload
    from tools.native_source_pair import native_source_pair_payload
    from tools.evaluate_source_evidence import layout
    from tools.evaluate_contour_alpha import aggregate
    from tools.atomic_progress import save_json
    contour.classify_payload=lambda f,a,c,v,grounding:baseline_payload(f,a,c,v)
    source.source_last_payload=native_source_pair_payload
    runner.checked_layout=layout;runner.aggregate=aggregate;transport.save=save_json
    return runner.evaluate(args)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',required=True)
    raise SystemExit(evaluate(p.parse_args()))
