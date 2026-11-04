"""Same-context higher-resolution CPU control, diagnostic only."""
import diagnose_mask2former as D
import json,time

def main():
    import numpy as np,torch,psutil
    from PIL import Image,ImageDraw,ImageFont
    from transformers import Mask2FormerForUniversalSegmentation,Mask2FormerImageProcessor
    from calibrated_postprocess import postprocess
    out=D.OUT/'resolution_control'
    if (out/'completed.json').exists():raise FileExistsError('Diagnostic completed')
    if psutil.virtual_memory().available<3*2**30:raise RuntimeError('Need 3GiB initial RAM for CPU diagnostic; no process stopped')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    model=Mask2FormerForUniversalSegmentation.from_pretrained(D.PACK/'model',local_files_only=True).eval()
    processor=Mask2FormerImageProcessor.from_pretrained(D.PACK/'model',local_files_only=True)
    cfg=json.loads((D.PACK/'inference_config.json').read_text(encoding='utf-8'));rows=[]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
    def overlay(image,ids,segments):
        a=np.array(image).copy()
        for s in segments:
            color=np.array([235,70,55] if s['category_id']==1 else [40,200,190] if s['category_id']==3 else [255,165,0]);m=ids==s['id'];a[m]=(a[m]*.5+color*.5).astype('uint8')
        return Image.fromarray(a)
    for camera in (0,1):
        case=f'camera{camera}_frame76';source=D.OLD/'inputs'/(case+'.png');image=Image.open(source).convert('RGB')
        inputs=processor(images=image.resize((1152,896),Image.Resampling.BILINEAR),do_resize=False,return_tensors='pt')
        started=time.monotonic()
        with torch.inference_mode():output=model(**inputs)
        prob=output.class_queries_logits[0].softmax(-1);ids,segments=postprocess(output,(image.height,image.width),cfg['postprocessing'])
        row={'case':case,'input_size':[896,1152],'source_sha256':D.sha(source),'PAL_count':sum(s['category_id']==1 for s in segments),
             'PAL_argmax_queries':int((prob.argmax(-1)==0).sum()),'maximum_PAL_probability':float(prob[:,0].max()),'seconds':time.monotonic()-started,'used_for_candidate':False}
        rows.append(row);D.save(out/case/'diagnosis.json',row);D.save(out/case/'prediction.json',{'segments_info':segments})
        Image.fromarray(np.stack([ids%256,ids//256%256,ids//65536%256],-1).astype('uint8')).save(out/case/'panoptic.png')
        prior=np.array(Image.open(D.OUT/'query_diagnosis'/case/'whole'/'panoptic.png')).astype('uint32');prior=prior[:,:,0]+prior[:,:,1]*256+prior[:,:,2]*65536
        oldmeta=json.loads((D.OLD/'mask2former'/case/'prediction.json').read_text(encoding='utf-8'))['segments_info']
        canvas=Image.new('RGB',(1920,700),'#f1f5fa');draw=ImageDraw.Draw(canvas)
        draw.text((18,12),case+' · 保持整图上下文，仅将网络输入尺寸加倍',font=font,fill='#142536')
        for i,(name,im) in enumerate([('原图',image),('448×576原输入',overlay(image,prior,oldmeta)),('896×1152诊断输入',overlay(image,ids,segments))]):
            draw.text((640*i+15,58),name,font=font,fill='#142536');canvas.paste(im.resize((640,512)),(640*i,100))
        draw.text((18,636),f"托盘 {row['PAL_count']}；此控制只用于归因，不纳入候选或调阈值。",font=font,fill='#142536')
        file=D.OUT/'assets'/(case+'_resolution_control.jpg');canvas.save(file,quality=94)
        print(json.dumps(row),flush=True);del inputs,output,prob
    D.save(out/'completed.json',{'status':'complete','real_forwards':2,'device':'cpu','rows':rows,'used_for_candidate':False,'annotation_writes':0,'weights_sha256':D.sha(D.PACK/'model/model.safetensors')})

if __name__=='__main__':main()
