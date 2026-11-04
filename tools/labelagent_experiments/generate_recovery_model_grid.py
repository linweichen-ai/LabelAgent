"""Fixed image-only grid inference with the delivered trained checkpoint."""
import sys,json,io,hashlib,time,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'task_workspaces/labelagent_recovery_20261105';out=ROOT/'outputs/labelagent_recovery_20261105'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 cfg=read(work/'config.json');dest=out/'fixed_grid';dest.mkdir(exist_ok=True)
 if (dest/'completed.json').exists():raise FileExistsError('Completed grid is immutable')
 sys.path[:0]=[cfg['dependency_dir'],str(Path(cfg['package'])),cfg['source_snapshot']]
 import torch,numpy as np,psutil
 from PIL import Image
 from transformers import Mask2FormerForUniversalSegmentation,Mask2FormerImageProcessor
 from infer_single import infer
 from source.agent.edit_agent import _shape_from_mask
 for _ in range(120):
  if psutil.virtual_memory().available>=2*1024**3:break
  print('Waiting for 2 GiB available host memory; no process stopped',flush=True);time.sleep(5)
 else:raise RuntimeError('Host memory guard not satisfied')
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);torch.set_num_threads(3)
 package=Path(cfg['package']);weights=package/'model/model.safetensors';before=hashlib.sha256(weights.read_bytes()).hexdigest()
 assert before=='d11c2e7f86cd5189605b857c4697e55a5caead0a6888b52b2f5ff814150f5291'
 model=Mask2FormerForUniversalSegmentation.from_pretrained(package/'model',local_files_only=True).to('cpu').eval()
 processor=Mask2FormerImageProcessor.from_pretrained(package/'model',local_files_only=True);params=read(package/'inference_config.json')
 calls=[0];hook=model.register_forward_pre_hook(lambda *_:calls.__setitem__(0,calls[0]+1));rows=[]
 for cam in [0,1]:
  case=f'camera{cam}_frame76';source=ROOT/'outputs/labelagent_efficiency_partial_comparison_20261104/inputs'/(case+'.png');image=Image.open(source).convert('RGB');w,h=image.size
  for y in [0,341,682,1024]:
   for x in [0,426,853,1280]:
    key=f'x{x}_y{y}';folder=dest/case/key;folder.mkdir(parents=True,exist_ok=True)
    crop=image.crop((x,y,x+640,y+512));crop.save(folder/'input.png')
    ids,segments=infer(model,processor,crop,params['input_height_width'],'cpu',params['postprocessing'])
    rgb=np.stack([ids%256,ids//256%256,ids//65536%256],-1).astype('uint8');Image.fromarray(rgb).save(folder/'panoptic.png')
    shapes=[]
    for seg in segments:
     if seg['category_id']!=1:continue
     mapped=np.zeros((h,w),np.uint8);mapped[y:y+512,x:x+640]=(ids==seg['id']).astype(np.uint8)
     shape=_shape_from_mask(mapped,'pallet',len(shapes)+1)
     if shape:shape['score']=float(seg['score']);shapes.append(shape)
    save(folder/'prediction.json',{'segments_info':segments,'PAL_shapes':shapes,'model_forward_passes':1,'source_image_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'crop_xywh':[x,y,640,512]})
    row={'case':case,'tile':key,'PAL_candidates':len(shapes),'crop_xywh':[x,y,640,512]};rows.append(row);save(dest/'progress.json',{'rows':rows,'real_forwards':calls[0]});print(json.dumps(row),flush=True)
 hook.remove();assert calls[0]==32 and hashlib.sha256(weights.read_bytes()).hexdigest()==before
 save(dest/'completed.json',{'status':'complete','real_forwards':calls[0],'device':'CPU','same_checkpoint_sha256':before,'weights_unchanged':True,'no_reference_inputs':True,'rows':rows})
if __name__=='__main__':main()
