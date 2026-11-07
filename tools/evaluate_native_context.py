"""Fixed native-neighborhood comparison, original category-first output."""
import argparse,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'tools')]


def evaluate(args):
    import tools.contour_visibility as contour
    import tools.source_last as source
    import tools.evaluate_source_last as runner
    import tools.evaluate_vlm_grounding as grounding
    import tools.evaluate_target_first as transport
    from tools.source_last import source_last_payload as baseline_payload
    from tools.native_context import native_context_payload,validate_response,validate_backend
    from tools.evaluate_source_evidence import layout
    from tools.evaluate_contour_alpha import aggregate
    contour.classify_payload=lambda f,a,c,v,grounding:baseline_payload(f,a,c,v)
    source.source_last_payload=native_context_payload
    grounding.validated_response=validate_response
    transport.validate_backend=validate_backend
    runner.checked_layout=layout;runner.aggregate=aggregate
    return runner.evaluate(args)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',required=True)
    raise SystemExit(evaluate(p.parse_args()))
