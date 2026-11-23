"""Build a strict, human-referenced external expert VLM A/B report."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.agent.expert_benchmark import build_expert_ab_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build external expert VLM A/B report")
    parser.add_argument("--input", required=True)
    args = parser.parse_args(argv)
    path = Path(args.input).resolve()
    if not path.is_file():
        raise SystemExit("A/B input file does not exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(build_expert_ab_report(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
