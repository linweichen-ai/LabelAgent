"""Fixed 18-case evidence-first output comparison, archived raw wire intact."""
import argparse
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'tools')]


def evaluate(args):
    import tools.contour_visibility as contour
    import tools.source_last as source
    import tools.evaluate_source_last as runner
    import tools.evaluate_vlm_grounding as grounding
    from tools.source_last import source_last_payload as baseline_payload
    from tools.evidence_first import evidence_first_payload, validated_evidence_first
    from tools.evaluate_source_evidence import layout
    from tools.evaluate_contour_alpha import aggregate
    contour.classify_payload=lambda f,a,c,v,grounding:baseline_payload(f,a,c,v)
    source.source_last_payload=evidence_first_payload
    grounding.validated_response=validated_evidence_first
    runner.checked_layout=layout
    runner.aggregate=aggregate
    return runner.evaluate(args)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--plan',required=True)
    p.add_argument('--plan-sha256',required=True)
    p.add_argument('--output',required=True)
    raise SystemExit(evaluate(p.parse_args()))
