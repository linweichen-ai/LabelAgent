"""Frozen CPU query/gate diagnosis and four-quadrant PAL-only candidate."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
import sys,json,hashlib,importlib.util,copy,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'task_workspaces/labelagent_hybrid_repair_20261104'
OUT=ROOT/'outputs/labelagent_hybrid_repair_20261104'
PACK=ROOT/'outputs/mask2former_single_recovery_20261104/最新单模型候选包'
OLD=ROOT/'outputs/labelagent_hybrid_comparison_v3_20261104'
sys.path[:0]=[str(ROOT/'task_workspaces/cvat_mask2former_20261103/deps'),str(PACK),str(WORK/'source_snapshot')]

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    import numpy as np,torch,psutil,cv2
    from PIL import Image,ImageDraw,ImageFont
    from transformers import Mask2FormerForUniversalSegmentation,Mask2FormerImageProcessor
    from calibrated_postprocess import postprocess
    from source.agent.edit_agent import _shape_from_mask,_rasterize
    if (OUT/'mask_diagnosis.json').exists():raise FileExistsError('Completed diagnosis retained')
    if psutil.virtual_memory().available<3*2**30:raise RuntimeError('CPU diagnosis needs 3GiB initial RAM; no process stopped')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    OUT.mkdir(exist_ok=True);plan=json.loads((WORK/'repair_plan.json').read_text(encoding='utf-8'))
    assert sha(PACK/'model/model.safetensors')==plan['model_sha256']
    save(OUT/'repair_plan.json',plan)
    model=Mask2FormerForUniversalSegmentation.from_pretrained(PACK/'model',local_files_only=True).eval()
    processor=Mask2FormerImageProcessor.from_pretrained(PACK/'model',local_files_only=True)
    cfg=json.loads((PACK/'inference_config.json').read_text(encoding='utf-8'));params=cfg['postprocessing']
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    calls=0;rows=[];pictures=[];all_candidates={};started=time.monotonic()
    for camera in (0,1):
        case=f'camera{camera}_frame76';source=OLD/'inputs'/(case+'.png');image=Image.open(source).convert('RGB');w,h=image.size
        boxes=[('whole',(0,0,w,h)),('q1',(0,0,w//2,h//2)),('q2',(w//2,0,w,h//2)),('q3',(0,h//2,w//2,h)),('q4',(w//2,h//2,w,h))]
        additions=[];case_rows=[]
        for name,box in boxes:
            if psutil.virtual_memory().available<1024*2**20:raise RuntimeError('CPU memory guard; diagnosis caches retained')
            folder=OUT/'query_diagnosis'/case/name;crop=image.crop(box)
            inputs=processor(images=crop.resize((576,448),Image.Resampling.BILINEAR),do_resize=False,return_tensors='pt')
            with torch.inference_mode():output=model(**inputs)
            calls+=1;logits=output.class_queries_logits[0].float();prob=logits.softmax(-1);scores,labels=prob.max(-1)
            masks=output.masks_queries_logits[0].float().sigmoid()
            npz=folder/'raw_queries.npz';folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(npz,class_logits=logits.numpy(),mask_logits=output.masks_queries_logits[0].float().numpy())
            query_rows=[]
            for i in torch.argsort(prob[:,0],descending=True).tolist():
                top=int(labels[i]);query_rows.append({'query':i,'top_class':model.config.id2label.get(top,'null'),
                    'p_pallet':float(prob[i,0]),'p_null':float(prob[i,-1]),'top_score':float(scores[i]),
                    'mask_pixels_at_native_grid':int((masks[i]>=.5).sum()),
                    'class_gate':'PAL_retained_for_mask_competition' if top==0 and scores[i]>.2 else ('argmax_null' if top==7 else 'argmax_other_class' if top!=0 else 'below_PAL_score_threshold')})
            ids,segs=postprocess(output,(crop.height,crop.width),params)
            relaxed_ids,relaxed_segs=postprocess(output,(crop.height,crop.width),{**params,'promote_null_pallet':True})
            row={'case':case,'view':name,'crop':list(box),'input_sha256':sha(source),'forward':calls,
                 'top_class_counts':{model.config.id2label.get(k,'null'):int((labels==k).sum()) for k in range(8)},
                 'calibrated_PAL_count':sum(s['category_id']==1 for s in segs),
                 'diagnostic_null_promotion_PAL_count':sum(s['category_id']==1 for s in relaxed_segs),
                 'diagnostic_null_promotion_used_in_candidate':False,'queries_ranked_by_PAL':query_rows}
            save(folder/'diagnosis.json',row);rows.append(row);case_rows.append(row)
            rgb=np.stack([ids%256,ids//256%256,ids//65536%256],-1).astype('uint8');Image.fromarray(rgb).save(folder/'panoptic.png')
            for s in segs:
                if s['category_id']!=1 or name=='whole':continue
                shape=_shape_from_mask((ids==s['id']).astype('uint8'),'pallet',len(additions)+1)
                if not shape:continue
                for part in shape['parts']:
                    for ring in [part['points'],*(part.get('holes') or [])]:
                        for point in ring:point[0]+=box[0];point[1]+=box[1]
                shape['points']=shape['parts'][0]['points'];shape['holes']=shape['parts'][0].get('holes',[])
                shape['score']=s['score'];shape['provenance']={'model_sha256':plan['model_sha256'],'view':name,'query_native_resolution':[448,576]};additions.append(shape)
            canvas=Image.new('RGB',(1920,840),'#f2f5fa');d=ImageDraw.Draw(canvas)
            d.text((18,12),case+' / '+name+' · 实际网络查询诊断（热图不是标注）',font=font,fill='#142536')
            panels=[('原始裁剪',crop)]
            overlay=np.array(crop).copy()
            for s in segs:
                color=np.array([235,70,55] if s['category_id']==1 else [40,200,190] if s['category_id']==3 else [255,165,0]);m=ids==s['id'];overlay[m]=(overlay[m]*.5+color*.5).astype('uint8')
            panels.append(('原参数输出',Image.fromarray(overlay)))
            for q in query_rows[:2]:
                heat=cv2.applyColorMap((masks[q['query']].numpy()*255).astype('uint8'),cv2.COLORMAP_TURBO)[:,:,::-1]
                panels.append((f"PAL查询{q['query']} p={q['p_pallet']:.3f} / {q['top_class']}",Image.fromarray(heat)))
            for n,(title,im) in enumerate(panels):
                d.text((n*480+12,58),title,font=small,fill='#142536');canvas.paste(im.resize((480,640)),(n*480,100))
            d.text((18,760),f"固定参数检出托盘 {row['calibrated_PAL_count']}；仅诊断开启null提升 {row['diagnostic_null_promotion_PAL_count']}（不用于候选）。",font=font,fill='#142536')
            file=OUT/'assets'/(case+'_'+name+'_queries.jpg');file.parent.mkdir(exist_ok=True);canvas.save(file,quality=93);pictures.append(file.name)
            print(json.dumps({'case':case,'view':name,'PAL':row['calibrated_PAL_count'],'null_diag':row['diagnostic_null_promotion_PAL_count'],'maxPAL':query_rows[0]['p_pallet'],'forwards':calls}),flush=True)
            del output,masks,inputs
        # Freeze additions before any sparse-reference scoring; deterministic mask IoU NMS.
        kept=[];kept_masks=[]
        for shape in sorted(additions,key=lambda s:-float(s['score'])):
            mask=_rasterize(shape,w,h)>0
            if any(np.logical_and(mask,m).sum()/max(1,np.logical_or(mask,m).sum())>.5 for m in kept_masks):continue
            kept.append(shape);kept_masks.append(mask)
        baseline=json.loads((OLD/'seed_stages'/case/'mask_raw.json').read_text(encoding='utf-8'))
        final=copy.deepcopy(baseline)+kept
        for i,s in enumerate(final,1):s['id']=i
        all_candidates[case]=final;save(OUT/'mask_tiled'/case/'shapes.json',final)
        save(OUT/'mask_tiled'/case/'receipt.json',{'raw_PAL_additions':len(additions),'after_fixed_NMS':len(kept),'ground_truth_used_for_candidate':False})
    assert calls==10 and sha(PACK/'model/model.safetensors')==plan['model_sha256']
    summary={'status':'complete','real_forwards':calls,'device':'cpu','model_sha256':plan['model_sha256'],'weights_unchanged':True,'annotation_writes':0,
             'elapsed_seconds':time.monotonic()-started,'rows':[{k:v for k,v in r.items() if k!='queries_ranked_by_PAL'} for r in rows],
             'candidate_sha256':{k:sha(OUT/'mask_tiled'/k/'shapes.json') for k in all_candidates},'pictures':pictures}
    save(OUT/'mask_diagnosis.json',summary)

if __name__=='__main__':main()
