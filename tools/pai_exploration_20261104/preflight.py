"""Cloud acceptance: actual CUDA allocation, BF16 and training imports."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    import torch
    import flash_attn
    from transformers import Qwen3VLForConditionalGeneration
    import peft
    assert torch.cuda.is_available(), 'CUDA unavailable'
    devices=[]
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            assert torch.cuda.is_bf16_supported(), 'BF16 unavailable'
            x=torch.ones((256,256),dtype=torch.bfloat16,device=f'cuda:{i}'); y=x@x
            torch.cuda.synchronize();assert float(y[0,0])==256
            free,total=torch.cuda.mem_get_info()
            devices.append({'index':i,'name':torch.cuda.get_device_name(i),'free_bytes':free,'total_bytes':total,'bf16_matmul_passed':True})
            del x,y
    # This process owns its temporary tensors only; no other process is stopped.
    d={'status':'CUDA_preflight_passed_NOT_training_verified','python':platform.python_version(),'torch':torch.__version__,
       'cuda':torch.version.cuda,'devices':devices,'packages':{k:importlib.metadata.version(k) for k in ['transformers','peft','flash-attn','deepspeed','accelerate']},
       'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.free','--format=csv'],text=True)}
    Path(a.out).write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2))


if __name__=='__main__':main()
