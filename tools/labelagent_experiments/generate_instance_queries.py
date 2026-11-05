"""Same trained checkpoint, paired contexts, original PAL query masks."""
import json,sys,hashlib,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'task_workspaces/labelagent_instance_separation_20261105';out=ROOT/'outputs/labelagent_instance_separation_20261105'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 cfg=read(work/'config.json');sys.path[:0]=[cfg['source_snapshot'],cfg['dependency_dir'],cfg['package']]
 import numpy as np,cv2,torch,psutil
 import torch.nn.functional as F
 from PIL import Image
 from transformers import Mask2FormerForUniversalSegmentation,Mask2FormerImageProcessor
 from calibrated_postprocess import postprocess
 from source.agent.edit_agent import _rasterize
 from instance_separation_geometry import predicted_groups,exact_binary_shape
 dest=out/'query_generation';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'completed.json').exists():raise FileExistsError('Frozen query generation')
 parents=[];inc=ROOT/'outputs/labelagent_context_comparison_20261105'
 for cam in [0,1]:
  case=f'camera{cam}_frame76';image=Image.open(inc/'inputs'/(case+'.png')).convert('RGB');shapes=read(inc/'C_hybrid'/case/'result.json')['shapes']
  for j,g in enumerate(predicted_groups(shapes,image)):
   folder=out/'parents'/case/g['key'];folder.mkdir(parents=True,exist_ok=True);Image.fromarray(g['mask']*255).save(folder/'mask.png');g={k:v for k,v in g.items() if k!='mask'};g.update(case=case,folder=str(folder));parents.append(g)
 save(out/'parents/completed.json',{'status':'complete','rows':parents,'source':'only incumbent predicted PAL masks and original images','reference_inputs':False})
 for _ in range(120):
  if psutil.virtual_memory().available>=2*1024**3:break
  print('Waiting for2GiB host memory; no process stopped',flush=True);time.sleep(5)
 else:raise RuntimeError('Host memory guard not met')
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);torch.set_num_threads(3);package=Path(cfg['package']);weight=package/'model/model.safetensors';before=hashlib.sha256(weight.read_bytes()).hexdigest();assert before=='d11c2e7f86cd5189605b857c4697e55a5caead0a6888b52b2f5ff814150f5291'
 model=Mask2FormerForUniversalSegmentation.from_pretrained(package/'model',local_files_only=True).to('cpu').eval();processor=Mask2FormerImageProcessor.from_pretrained(package/'model',local_files_only=True);params=read(package/'inference_config.json');calls=[0];hook=model.register_forward_pre_hook(lambda *_:calls.__setitem__(0,calls[0]+1));records=[];forwards=[]
 for p in parents:
  image=Image.open(inc/'inputs'/(p['case']+'.png')).convert('RGB');x1,y1,x2,y2=p['box']
  for pad in [32,96]:
   box=[max(0,x1-pad),max(0,y1-pad),min(1920,x2+pad),min(1536,y2+pad)];crop=image.crop(tuple(box));folder=dest/p['case']/(p['key']+'_'+str(pad));folder.mkdir(parents=True,exist_ok=True);crop.save(folder/'input.png');inputs=processor(images=crop.resize((576,448),Image.Resampling.BILINEAR),do_resize=False,return_tensors='pt')
   with torch.inference_mode():pred=model(**inputs)
   classprob=pred.class_queries_logits[0].softmax(-1);scores,labels=classprob.max(-1);queries=torch.where((labels==0)&(scores>.2))[0];ids,segments=postprocess(pred,(crop.height,crop.width),params['postprocessing']);rgb=np.stack([ids%256,ids//256%256,ids//65536%256],-1).astype('uint8');Image.fromarray(rgb).save(folder/'panoptic.png');meta={'parent_key':p['key'],'case':p['case'],'context_pad':pad,'source_box':box,'query_indices':queries.tolist(),'query_scores':[float(scores[q]) for q in queries],'standard_PAL_instances':sum(s['category_id']==1 for s in segments),'segments_info':segments,'exact_query_rejections':[]};raw_rows=[]
   for q in queries.tolist():
    probability=F.interpolate(pred.masks_queries_logits[:,q:q+1],size=(crop.height,crop.width),mode='bilinear',align_corners=False)[0,0].sigmoid().detach().numpy();local=(probability>=.5).astype('uint8');Image.fromarray(local*255).save(folder/f'query{q}_raw.png')
    border=int(local[0].any())+int(local[-1].any())+int(local[:,0].any())+int(local[:,-1].any())
    if not local.any() or border>=3:meta['exact_query_rejections'].append({'query':q,'reason':'empty or truncated at3 crop edges'});continue
    mask=np.zeros((1536,1920),np.uint8);mask[box[1]:box[3],box[0]:box[2]]=local;shape=exact_binary_shape(mask)
    if shape is None:meta['exact_query_rejections'].append({'query':q,'reason':'complete raster not representable under original geometry limits'});continue
    assert np.array_equal(mask,_rasterize(shape,1920,1536));key=hashlib.sha256(p['case'].encode()+mask.tobytes()).hexdigest();row={'key':key,'case':p['case'],'shape':shape,'score':float(scores[q]),'parent_key':p['key'],'context_pad':pad,'query_index':q,'source':'raw_trained_instance_query','source_prediction':str(folder/'prediction.json'),'raw_mask_file':str(folder/f'query{q}_raw.png'),'source_box':box,'whole_query_raster_sha256':hashlib.sha256(mask.tobytes()).hexdigest()};raw_rows.append(row);records.append(row)
   save(folder/'prediction.json',{**meta,'raw_query_hypotheses':raw_rows,'forward_passes':1,'input_height_width':[448,576],'original_checkpoint_sha256':before});row={'case':p['case'],'parent':p['key'],'context_pad':pad,'PAL_queries':len(queries),'standard_PAL_instances':meta['standard_PAL_instances'],'exact_hypotheses':len(raw_rows)};forwards.append(row);save(dest/'progress.json',{'forwards':forwards,'real_forwards':calls[0]});print(json.dumps(row),flush=True)
 hook.remove();assert calls[0]==2*len(parents) and hashlib.sha256(weight.read_bytes()).hexdigest()==before;save(dest/'all_hypotheses.json',records);save(dest/'completed.json',{'status':'complete','real_forwards':calls[0],'parents':len(parents),'raw_query_hypotheses':len(records),'same_checkpoint_sha256':before,'weights_unchanged':True,'reference_inputs':False,'forward_rows':forwards});print(json.dumps(read(dest/'completed.json')),flush=True)
if __name__=='__main__':main()
