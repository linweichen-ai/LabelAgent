"""Inspect the installed real Qwen image processor, without loading model weights."""
import argparse
import re
import hashlib
import io
import json
import math
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['OMP_NUM_THREADS']='2'
import torch
from transformers import AutoProcessor
from PIL import Image

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model-dir',required=True)
parser.add_argument('--fixture-json',required=True)
parser.add_argument('--asset-root')
parser.add_argument('--output',required=True)
args=parser.parse_args()
fixture_path=Path(args.fixture_json).resolve()
fixture=json.loads(fixture_path.read_text(encoding='utf-8'))
root=Path(args.asset_root or fixture.get('asset_root',fixture_path.parent)).resolve()
model_dir=Path(args.model_dir).resolve()
config_raw=(model_dir/'config.json').read_bytes()
if json.loads(config_raw).get('model_type')!='qwen3_vl':
    raise ValueError('This probe requires the Qwen3-VL processor configuration')
output=Path(args.output).resolve()
output.mkdir(parents=True,exist_ok=False)
assert not torch.cuda.is_initialized()
processor=AutoProcessor.from_pretrained(str(model_dir),local_files_only=True)
merge=int(processor.image_processor.merge_size)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
rows=[]
for case in fixture['cases']:
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',case['id']):
        raise ValueError('Unsafe case ID in fixture')
    for index,asset in enumerate(case['views']['context']):
        path=(root/asset['path']).resolve()
        if not path.is_relative_to(root):raise ValueError('Source image escapes the asset root')
        raw=path.read_bytes()
        assert sha(raw)==asset['sha256']
        native=Image.open(io.BytesIO(raw)).convert('RGB')
        factor=math.sqrt(1024*32*32/(native.width*native.height))
        size=tuple(max(32,round(value*factor/32)*32) for value in native.size)
        common=native.resize(size,Image.Resampling.BICUBIC)
        target=output/(case['id']+f'_view_{index}_common.png')
        common.save(target,'PNG')
        counts={}
        grids={}
        for kind,image in [('native',native),('common_upsampled',common)]:
            for budget in (256,1024):
                processor.image_processor.size={'longest_edge':budget*32*32,'shortest_edge':64*32*32}
                result=processor.image_processor(images=[image],return_tensors='pt')
                grid=result['image_grid_thw'].tolist()
                assert len(grid)==1
                key=kind+'_'+str(budget)
                counts[key]=math.prod(grid[0])//merge**2
                grids[key]=grid[0]
        rows.append({'case_id':case['id'],'view_index':index,'role':case['role'],
            'original_sha256':asset['sha256'],'original_size':list(native.size),'common_size':list(common.size),
            'common_png_sha256':sha(target.read_bytes()),'image_tokens':counts,'grid_thw':grids})
assert not torch.cuda.is_initialized()
summary={'status':'complete','scope':'CPU_real_processor_only_no_model_inference_or_quality_result',
    'source_code_sha256':sha(Path(__file__).read_bytes()),'fixture_sha256':sha(fixture_path.read_bytes()),
    'model_config_sha256':sha(config_raw),'processor_type':type(processor.image_processor).__name__,'merge_size':merge,'views_checked':len(rows),
    'native_views_with_more_tokens_at_1024':sum(r['image_tokens']['native_1024']>r['image_tokens']['native_256'] for r in rows),
    'common_views_with_more_tokens_at_1024':sum(r['image_tokens']['common_upsampled_1024']>r['image_tokens']['common_upsampled_256'] for r in rows),
    'CUDA_initialized':False,'model_weights_loaded':False,'new_model_requests':0,'source_images_modified':False,
    'upsampling_adds_source_detail':False,'rows':rows}
(output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)
print(json.dumps([{'case':r['case_id'],'view':r['view_index'],'tokens':r['image_tokens']} for r in rows]),flush=True)
