import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--size',choices=['2B','8B','32B'],required=True);p.add_argument('--models-root',required=True);a=p.parse_args()
    model=f'Qwen/Qwen3-VL-{a.size}-Instruct'
    pins=json.loads((Path(__file__).parent/'model_pins.json').read_text())
    target=Path(a.models_root)/model.split('/')[-1]
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=model,revision=pins[model],local_dir=target)
    print(json.dumps({'model':model,'revision':pins[model],'directory':str(target)}))


if __name__=='__main__':main()
