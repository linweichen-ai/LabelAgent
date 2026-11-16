"""Use the new classification builder with the unchanged frozen audit runner."""
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'tools')]


def evaluate(args):
    import tools.target_first as original
    from tools.contour_visibility import classify_payload
    from tools.evaluate_target_first import evaluate as frozen_evaluate
    previous = original.classify_payload
    original.classify_payload = classify_payload
    try:
        return frozen_evaluate(args)
    finally:
        original.classify_payload = previous


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', required=True)
    raise SystemExit(evaluate(parser.parse_args()))
