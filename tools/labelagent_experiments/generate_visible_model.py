"""Eight same-checkpoint CPU forwards with aspect-preserving source inputs."""
import sys,json,hashlib,time
from pathlib import Path
from plan_visible_bases import ROOT,work,out,read,save
def main():
 cfg=read(work/'config.json');sys.path[:0]=[cfg['source_snapshot'],cfg['dependency_dir'],cfg['package']];import torch,numpy as np,psutil
 import torch.nn.functional as F
 from PIL import Image
 from transformers import Mask2FormerForUniversalSegmentation,Mask2FormerImageProcessor
 from calibrated_postprocess import postprocess
 from instance_separation_geometry import exact_binary_shape
 from source.agent.edit_agent import _rasterize
 dest=out/'aspect_model_generation';dest.mkdir(exist_ok=True)
 if (dest/'completed.json').exists():raise FileExistsError('Frozen model outputs')
 cohort=read(out/'planning/cohort.json');save(dest/'frozen_plan.json',{'source_cohort_sha256':hashlib.sha256((out/'planning/cohort.json').read_bytes()).hexdigest(),'forwards':8,'same_checkpoint':True,'input_height_width':[448,576],'input_aspect_ratio_preserved':True,'gray_padding_is_not_scene_data':True,'PAL_argmax_score_min':.2,'mask_probability_min':.5,'any_positive_mask_in_padding_rejects_whole_query':True,'post_budget_clipping':False,'reference_positions_used':False})
 for _ in range(120):
  if psutil.virtual_memory().available>=2*1024**3:break
  time.sleep(5)
 else:raise RuntimeError('2GiB RAM guard not met')
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);torch.set_num_threads(3);package=Path(cfg['package']);weight=package/'model/model.safetensors';h=hashlib.sha256(weight.read_bytes()).hexdigest();assert h=='d11c2e7f86cd5189605b857c4697e55a5caead0a6888b52b2f5ff814150f5291';model=Mask2FormerForUniversalSegmentation.from_pretrained(package/'model',local_files_only=True).to('cpu').eval();processor=Mask2FormerImageProcessor.from_pretrained(package/'model',local_files_only=True);params=read(package/'inference_config.json');calls=[0];hook=model.register_forward_pre_hook(lambda *_:calls.__setitem__(0,calls[0]+1));hyps=[];rows=[]
 for seed in cohort:
  image=Image.open(Path(seed['folder'])/'SAM_source_input.png').convert('RGB');box=seed['binding']['source_box'];scale=min(576/image.width,448/image.height);dw,dh=round(image.width*scale),round(image.height*scale);ox,oy=(576-dw)//2,(448-dh)//2;canvas=Image.new('RGB',(576,448),(127,127,127));canvas.paste(image.resize((dw,dh),Image.Resampling.BILINEAR),(ox,oy));folder=dest/seed['case']/f"seed{seed['index']:02d}";folder.mkdir(parents=True,exist_ok=True);canvas.save(folder/'actual_padded_input.png');image.save(folder/'source_input.png');inputs=processor(images=canvas,do_resize=False,return_tensors='pt')
  with torch.inference_mode():pred=model(**inputs)
  scores,labels=pred.class_queries_logits[0].softmax(-1).max(-1);qs=torch.where((labels==0)&(scores>.2))[0].tolist();padding=np.ones((448,576),bool);padding[oy:oy+dh,ox:ox+dw]=False;rejected=[];kept=[];ids,segments=postprocess(pred,(448,576),params['postprocessing']);Image.fromarray(np.stack([ids%256,ids//256%256,ids//65536%256],-1).astype('uint8')).save(folder/'panoptic_padded.png')
  for q in qs:
   logits=F.interpolate(pred.masks_queries_logits[:,q:q+1],size=(448,576),mode='bilinear',align_corners=False);binary=logits[0,0].sigmoid().cpu().numpy()>=.5
   if (binary&padding).any():rejected.append({'query':q,'reason':'positive PAL pixels in artificial padding; reject whole mask'});continue
   local=(F.interpolate(logits[:,:,oy:oy+dh,ox:ox+dw],size=(image.height,image.width),mode='bilinear',align_corners=False)[0,0].sigmoid().cpu().numpy()>=.5).astype('uint8');Image.fromarray(local*255).save(folder/f'query{q}_source_mask.png');mask=np.zeros((1536,1920),np.uint8);x1,y1,x2,y2=box;mask[y1:y2,x1:x2]=local;shape=exact_binary_shape(mask)
   if shape is None:rejected.append({'query':q,'reason':'whole raster cannot be represented under original geometry limits'});continue
   assert np.array_equal(mask,_rasterize(shape,1920,1536));key=hashlib.sha256(seed['case'].encode()+mask.tobytes()).hexdigest();row={'key':key,'case':seed['case'],'shape':shape,'score':float(scores[q]),'source':'aspect_preserving_trained_query','source_prediction':str(folder/'prediction.json'),'source_mask_file':str(folder/f'query{q}_source_mask.png'),'source_box':box,'query_index':q,'seed_key':seed['key'],'padding_positive_pixels':0,'whole_query_raster_sha256':hashlib.sha256(mask.tobytes()).hexdigest()};hyps.append(row);kept.append(row)
  record={'case':seed['case'],'index':seed['index'],'source_box':box,'resized_source_size':[dw,dh],'offset':[ox,oy],'PAL_queries':len(qs),'accepted_complete_hypotheses':len(kept),'rejected_queries':rejected,'segments_info':segments,'hypotheses':kept};save(folder/'prediction.json',record);rows.append({k:record[k] for k in ['case','index','PAL_queries','accepted_complete_hypotheses','rejected_queries']});save(dest/'progress.json',rows);print(json.dumps(rows[-1]),flush=True)
 hook.remove();assert calls[0]==8 and hashlib.sha256(weight.read_bytes()).hexdigest()==h;save(dest/'all_hypotheses.json',hyps);save(dest/'completed.json',{'status':'complete','real_CPU_forwards':calls[0],'same_checkpoint_sha256':h,'weights_unchanged':True,'valid_complete_outputs':len(hyps),'reference_inputs':False,'no_output_clipping':True});print(json.dumps(read(dest/'completed.json')),flush=True)
if __name__=='__main__':main()
