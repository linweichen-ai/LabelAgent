import sys,argparse
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1])]
if __name__=='__main__':
    import tools.evaluate_source_scene as runner
    from tools.source_scene_contract import scene_payload,reviewed_category,CATEGORIES
    runner.scene_payload=scene_payload;runner.reviewed_category=reviewed_category;runner.CATEGORIES=CATEGORIES
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',required=True);raise SystemExit(runner.evaluate(p.parse_args()))
