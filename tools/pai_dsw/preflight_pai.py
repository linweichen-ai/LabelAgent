"""Read-only hardware/model checks for an allocated PAI DSW instance."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-gpus',type=int,default=2)
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--memory-utilization',type=float,default=.85)
    parser.add_argument('--hash-weights',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not 0<args.memory_utilization<1:parser.error('Use memory utilization strictly between zero and one')
    import torch
    report={'python':platform.python_version(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,
            'cuda_available':torch.cuda.is_available(),'gpu_count':torch.cuda.device_count(),'gpus':[]}
    total_visible_memory=0
    assert report['cuda_available'],'CUDA is unavailable; CPU fallback is not accepted'
    assert report['gpu_count']==args.expected_gpus,'Visible GPU count differs from allocation; do not override PAI device visibility'
    for i in range(report['gpu_count']):
        props=torch.cuda.get_device_properties(i)
        free,total=torch.cuda.mem_get_info(i)
        total_visible_memory+=total
        row={'index':i,'name':props.name,'total_GiB':round(total/2**30,2),'free_GiB':round(free/2**30,2)}
        report['gpus'].append(row)
        assert total>=44*2**30,'This profile expects nominal 48GB or larger GPUs'
        assert free>=total*args.memory_utilization,'Insufficient currently free memory; no other job was stopped'
    report['nvidia_smi']=subprocess.run(['nvidia-smi'],capture_output=True,text=True,check=True).stdout
    report['shm_GiB']=round(shutil.disk_usage('/dev/shm').total/2**30,2)
    assert report['shm_GiB']>=8,'Increase instance shared memory; 16GiB is recommended'
    if args.model_dir:
        model=args.model_dir.resolve();config=json.loads((model/'config.json').read_text())
        assert config.get('model_type')=='qwen3_vl','Expected dense Qwen3-VL model'
        assert not config.get('quantization_config'),'This guide uses BF16, not a prequantized checkpoint'
        index=model/'model.safetensors.index.json'
        shards=(set(json.loads(index.read_text())['weight_map'].values()) if index.exists()
                else {p.name for p in model.glob('*.safetensors')})
        assert shards,'Model weights not found'
        files=[model/name for name in sorted(shards)]
        assert all(p.is_file() and p.stat().st_size>0 for p in files),'Missing or empty weight shard'
        report['weight_shards']=len(files);report['weight_bytes']=sum(p.stat().st_size for p in files)
        assert report['weight_bytes']>=50*2**30,'Weight size is unexpectedly small for the requested BF16 32B model'
        assert report['weight_bytes']+8*2**30<total_visible_memory*args.memory_utilization, 'Model plus planning headroom exceeds this allocation; single 48GB is not the BF16 32B profile'
        report['config_sha256']=hashlib.sha256((model/'config.json').read_bytes()).hexdigest()
        if args.hash_weights:
            report['weight_sha256']={}
            for p in files:
                h=hashlib.sha256()
                with p.open('rb') as stream:
                    for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
                report['weight_sha256'][p.name]=h.hexdigest()
    report['read_only']=True
    text=json.dumps(report,ensure_ascii=False,indent=2)
    if args.output:args.output.write_text(text+'\n',encoding='utf-8')
    print(text)


if __name__=='__main__':main()
