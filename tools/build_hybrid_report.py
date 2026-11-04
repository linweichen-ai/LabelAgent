"""Render saved real comparison stages and clearly labelled result-replay videos."""
import argparse, hashlib, io, json, sys
from pathlib import Path
from fractions import Fraction


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);cfg=read(p.parse_args().config)
    out=Path(cfg['output']);result=read(out/'completed.json');assert result['status']=='complete'
    sys.path.insert(0,cfg['source_snapshot'])
    import numpy as np,cv2,av
    from PIL import Image,ImageDraw,ImageFont,ImageOps
    from source.agent.edit_agent import _rasterize
    labels=['pallet','person','floor','slope','ego','other_robot','cargo_floor']
    zh=['托盘','人员','普通地面','坡板','本车','其他机器人','车厢地板']
    palette=[(235,85,60),(255,165,0),(40,200,190),(240,220,35),(230,85,200),(135,90,230),(45,120,230)]
    priority=['floor','slope','cargo_floor','ego','other_robot','pallet','person']
    colors=dict(zip(labels,palette));titlefont=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    assets=out/'assets';assets.mkdir(exist_ok=True);videos=out/'videos';videos.mkdir(exist_ok=True)
    def panoptic(image,shapes,directory=None):
        w,h=image.size;ids=np.zeros((h,w),np.uint32);meta={};counter=0;stuff={}
        ordered=sorted(shapes,key=lambda s:(priority.index(s['label']),float(s.get('score') or 0),s['id']))
        for shape in ordered:
            mask=_rasterize(shape,w,h)>0;label=shape['label']
            if label in ('floor','slope','cargo_floor') and label in stuff:sid=stuff[label]
            else:
                counter+=1;sid=counter;meta[sid]={'id':sid,'label':label,'source_shape_ids':[]}
                if label in ('floor','slope','cargo_floor'):stuff[label]=sid
            meta[sid]['source_shape_ids'].append(shape['id']);ids[mask]=sid
        original=np.asarray(image);rendered=original.copy();segments=[]
        for sid in np.unique(ids):
            if not sid:continue
            mask=ids==sid;segment=dict(meta[int(sid)],area=int(mask.sum()));segments.append(segment)
            rendered[mask]=(original[mask]*.5+np.array(colors[segment['label']])*.5).astype('uint8')
            if segment['label'] in ('pallet','person','ego','other_robot'):
                contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rendered,contours,-1,(255,255,255),1)
        assert int((ids>0).sum())==sum(s['area'] for s in segments)
        if directory:
            directory.mkdir(parents=True,exist_ok=True)
            rgb=np.stack([ids%256,ids//256%256,ids//65536%256],-1).astype('uint8');Image.fromarray(rgb).save(directory/'panoptic.png')
            semantic=np.full((h,w),255,np.uint8)
            for s in segments:semantic[ids==s['id']]=labels.index(s['label'])
            Image.fromarray(semantic).save(directory/'semantic.png')
            save(directory/'segments.json',{'segments':segments,'unknown_pixels':int((ids==0).sum()),'presentation_priority':priority,'human_truth':False,'primary_scores_computed_before_this_export':True})
        return Image.fromarray(rendered)
    def legend(draw,y):
        for i,(label,name) in enumerate(zip(labels,zh)):
            x=15+(i%4)*475;yy=y+(i//4)*32;draw.rectangle((x,yy,x+19,yy+19),fill=colors[label]);draw.text((x+28,yy-1),name,font=small,fill='#162434')
    def stage_card(case,title,image,view,file,left_caption='原图'):
        canvas=Image.new('RGB',(1920,960),'#f1f5fa');d=ImageDraw.Draw(canvas)
        d.text((18,13),case+' · '+title,font=titlefont,fill='#162434')
        d.text((18,55),'左：'+left_caption+'；右：实际保存的预测。候选未确认为真值。',font=font,fill='#45566a')
        canvas.paste(image.resize((960,768)),(0,90));canvas.paste(view.resize((960,768)),(960,90));legend(d,875)
        canvas.save(file,quality=94);return file
    comparisons=[];replays=[];gallery=[];refinement=read(out/'completed.json')['refinement']
    for frame,camera in cfg['cases']:
        case=f'camera{camera}_frame{frame}';image=Image.open(out/'inputs'/(case+'.png')).convert('RGB');w,h=image.size
        seed=read(out/'seed_stages'/case/'all_arms.json');raw=read(out/'seed_stages'/case/'mask_raw.json');refined=read(out/'seed_stages'/case/'mask_refined.json')
        arms={name:read(out/name/case/'result.json')['shapes'] for name in ('A_current','B_sam7','C_hybrid')}
        rendered={name:panoptic(image,shapes,out/'dense_exports'/name/case) for name,shapes in arms.items()}
        stages=[('01 原始图像',image),('02 实际训练交付包初稿',panoptic(image,raw)),
                ('03 SAM3掩码与点细化',panoptic(image,refined)),('04 SAM3缺失类别补漏',panoptic(image,seed['C_hybrid'])),
                ('05 Agent阶段结果（失败则保留候选）',rendered['C_hybrid'])]
        for index,(title,view) in enumerate(stages):
            file=assets/(case+f'_stage_{index+1:02d}.jpg');stage_card(case,title,image,view,file);replays.append((file,4));gallery.append(file)
        panels=[('原图',image),('A 现有两类',rendered['A_current']),('B 七类SAM3',rendered['B_sam7']),('C 混合流程',rendered['C_hybrid'])]
        def comparison(rect,name):
            canvas=Image.new('RGB',(1920,590),'#f1f5fa');d=ImageDraw.Draw(canvas)
            d.text((18,10),case+' · '+name,font=titlefont,fill='#162434')
            for i,(label,im) in enumerate(panels):
                d.text((i*480+12,53),label,font=font,fill='#162434');canvas.paste(im.crop(rect).resize((480,384)),(i*480,88))
            legend(d,490);return canvas
        file=assets/(case+'_three_arms.jpg');comparison((0,0,w,h),'同图三方案').save(file,quality=95);comparisons.append((file,8));gallery.append(file)
        for i,rect in enumerate([(0,0,w//2,h//2),(w//2,0,w,h//2),(0,h//2,w//2,h),(w//2,h//2,w,h)],1):
            file=assets/(case+f'_quadrant_{i}.jpg');comparison(rect,'固定四分区放大 '+str(i)).save(file,quality=95);gallery.append(file)
        for row in [r for r in refinement if r['case']==case]:
            shape_id=row['shape_id'];before=[s for s in raw if s['id']==shape_id];after=[s for s in refined if s['id']==shape_id]
            first=panoptic(image,before);second=panoptic(image,after);draw=ImageDraw.Draw(second)
            for point in row['clicks']:
                x,y=point['x'],point['y'];color='#39ff6e' if point['label']==1 else '#ff3355';draw.ellipse((x-10,y-10,x+10,y+10),outline=color,width=4)
            file=assets/(case+f'_sam_refine_{shape_id}.jpg');caption='SAM3细化 '+row['label']+(' · 技术返回有效' if row.get('accepted_technical_output') else ' · 拒绝/保留原初稿')
            stage_card(case,caption,first,second,file,left_caption='细化前候选');gallery.append(file)
    scores=result['common_class_scores'];compact=[]
    for arm in ('A_current','B_sam7','C_hybrid'):
        p=scores[arm]['pallet'];h=scores[arm]['person'];r=p['radii']['3'];hr=h['radii']['3']
        compact.append({'arm':arm,'pallet_instances_supported':r['objects_matched_at_support_threshold'],
            'reference_pallet_instances':p['supported_reference_objects'],'pallet_points_covered':r['covered_projected_points'],
            'reference_pallet_points':p['reference_projected_points'],'person_points_covered_by_pallet':hr['points_covered_by_other_class'],
            'person_instances_supported':hr['objects_matched_at_support_threshold'],'reference_person_instances':h['supported_reference_objects'],
            'person_points_covered':hr['covered_projected_points'],'reference_person_points':h['reference_projected_points']})
    summary=Image.new('RGB',(1920,1080),'#132334');d=ImageDraw.Draw(summary)
    d.text((70,65),'同图三方案 · 共同类别参考一致性',font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',42),fill='white')
    y=180
    for row in compact:
        line=f"{row['arm']}: 托盘支持 {row['pallet_instances_supported']}/{row['reference_pallet_instances']}；参考点 {row['pallet_points_covered']}/{row['reference_pallet_points']}；人员误覆盖点 {row['person_points_covered_by_pallet']}"
        d.text((70,y),line,font=titlefont,fill='#b9f1e6')
        person=f"人员支持 {row['person_instances_supported']}/{row['reference_person_instances']}；人员参考点 {row['person_points_covered']}/{row['reference_person_points']}"
        d.text((90,y+43),person,font=font,fill='#d6eaf4');y+=112
    y+=30
    for line in [f"Agent仅{sum(bool(r['converged']) for r in result['agent_runs'])}/6组收敛；七类4组失败，整套流程未通过验收。",
                 '仅2张已暴露同采集图；这些不是像素准确率。','地面等类别没有完整二维真值，不能据此宣称七类提升。',
                 '实际使用训练交付包 d11c2e7f…；SAM3补漏单独记账。','原始标注未写入；模型权重和原图保持。','本片是实际阶段结果回放，不是连续工作台录屏。']:
        d.text((70,y),line,font=titlefont,fill='#e0e9f4');y+=66
    summary_file=assets/'comparison_summary.jpg';summary.save(summary_file,quality=95);gallery.append(summary_file);comparisons.append((summary_file,8))
    receipts=[]
    def video(name,frames):
        target=videos/name;temporary=target.with_suffix('.pending.mp4');container=av.open(str(temporary),'w',format='mp4',options={'movflags':'+faststart'});stream=container.add_stream('libx264',rate=12)
        stream.width,stream.height,stream.pix_fmt=1920,1080,'yuv420p';stream.codec_context.thread_count=2;stream.options={'crf':'20','preset':'veryfast'};count=0
        for file,seconds in frames:
            with Image.open(file) as source:im=ImageOps.contain(source.convert('RGB'),(1840,930))
            canvas=Image.new('RGB',(1920,1080),'#101e2a');canvas.paste(im,((1920-im.width)//2,45+(930-im.height)//2));draw=ImageDraw.Draw(canvas)
            draw.text((40,1010),'真实阶段结果回放 · 非连续录屏 · 已压缩等待 · 候选待人工确认',font=font,fill='#97dfd1')
            frame=av.VideoFrame.from_image(canvas).reformat(format='yuv420p')
            for _ in range(seconds*12):
                frame.pts=count;frame.time_base=Fraction(1,12);frame.pict_type=av.video.frame.PictureType.NONE
                for packet in stream.encode(frame):container.mux(packet)
                count+=1
        for packet in stream.encode():container.mux(packet)
        container.close();temporary.replace(target)
        with av.open(str(target)) as f:decoded=sum(1 for _ in f.decode(video=0))
        assert decoded==count
        receipt={'path':str(target.relative_to(out)),'duration_seconds':count/12,'decoded_frames':decoded,
                 'width':1920,'height':1080,'type':'real_saved_stage_replay_not_continuous_recording',
                 'sha256':hashlib.sha256(target.read_bytes()).hexdigest()};receipts.append(receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)
    video('01_真实阶段结果回放.mp4',replays)
    video('02_同图三方案对照.mp4',comparisons)
    save(out/'media_verification.json',{'status':'PASS','videos':receipts,'pictures':len(gallery),'unconfirmed_candidates':True})
    converged=sum(bool(r['converged']) for r in result['agent_runs'])
    failed=sum(r['stop_reason']=='error' for r in result['agent_runs'])
    trace=read(out/'model_trace.json');public={'status':'comparison_recorded_agent_review_incomplete' if failed else 'comparison_recorded',
        'agent_runs_converged':converged,'agent_runs_failed':failed,
        'end_to_end_agent_review_passed':all(bool(r['converged']) for r in result['agent_runs']),
        'images':2,'mask2former_checkpoint_sha256':result['model_sha256'],
        'weights_unchanged':result['weights_unchanged'],'source_images_unchanged':result['source_images_unchanged'],
        'protected_state_before_after_equal':result['protected_state_before_after_equal'],'annotation_writes':0,
        'mask2former_source_real_forwards':2,'mask2former_fresh_forwards_this_run':result['fresh_mask2former_forwards_this_run'],
        'SAM_source_requests':sum(r.get('service')=='SAM 3' for r in trace),'VLM_real_requests':sum(r.get('service')=='GPT' for r in trace),
        'VLM_protocol_valid_replies':sum(r.get('service')=='GPT' and r.get('status')=='succeeded' for r in trace),
        'common_class_comparison':compact,'agent_runs':[{k:v for k,v in r.items() if k!='case'} for r in result['agent_runs']],
        'SAM_refinement_attempts':len(refinement),'SAM_refinement_technically_accepted':sum(bool(r.get('accepted_technical_output')) for r in refinement),
        'dense_pixel_accuracy_evaluated':False,'scope':result['scope'],'pictures':len(gallery),'videos':len(receipts),'default_workbench_changed':False}
    public['hole_review_batch_size']=cfg.get('hole_review_batch_size')
    public['same_operation_cap_per_whole_agent_round']=3
    save(out/'public_summary.json',public)
    rows=['<!doctype html><meta charset="utf-8"><title>Mask2Former→SAM3→Agent 同图对照</title>',
      '<style>body{max-width:1450px;margin:30px auto;padding:20px;font:17px/1.7 "Microsoft YaHei",sans-serif;background:#f3f6fa;color:#152438}img,video{width:100%}section{padding:24px;background:white;border-radius:12px;margin:20px 0}td,th{padding:12px;border:1px solid #cdd5df}table{border-collapse:collapse}</style>',
      '<h1>实际训练交付包 → SAM3 → Agent：同图三方案对照</h1>',
      f'<section><strong>真实复核记录：{converged}/6组收敛，{failed}/6组调用或协议失败。失败组保留候选，不表示Agent纠错成功，整套流程尚未通过验收。</strong></section>',
      '<section><p>本次使用CVAT项目实际单模型交付包，权重d11c2e7f…，448×576及包内专用阈值。此前R5/R11的3图试跑保留为旧版诊断，不代替此模型结果。</p>',
      '<p>这份单模型确实完成3960次更新，但原项目交付记录也注明：人物和箱内地板有改善，小托盘仍有退步，尚未替换原默认模型。当前两张PNG的模型初稿同样未检出托盘；C方案托盘来自SAM3补漏。不能把本次失败解释为模型没有训练过，也不能仅据两图判断整个模型无效。</p>',
      '<p>A为现有两类流程；B为七类SAM3控制；C为训练模型初稿、SAM3掩码点细化、缺失类别补漏、Agent复核。B/C使用相同探索性七类规则，未假造人工七类参考。三组统一2048回复预算、最多3轮。</p>',
      '<p>首轮768 token导致孔洞判断JSON截断，第二轮出现CUDA显存失败，记录分别保留。最终版每次最多10个孔洞、全部覆盖；每个Agent轮次仍最多3项实际编辑。复用真实模型与SAM缓存，没有调权重或阈值。原始PNG一致，原标注未写入。</p></section>',
      '<section><h2>共同类别结果</h2><p>仅两张已暴露同采集图，按原V2稀疏投影参考指标计分；不是二维像素准确率。</p><table><tr><th>方案</th><th>托盘支持</th><th>托盘参考点覆盖</th><th>人员支持</th><th>人员参考点覆盖</th><th>人员被托盘误覆盖点↓</th></tr>']
    for r in compact:rows.append(f"<tr><td>{r['arm']}</td><td>{r['pallet_instances_supported']}/{r['reference_pallet_instances']}</td><td>{r['pallet_points_covered']}/{r['reference_pallet_points']}</td><td>{r['person_instances_supported']}/{r['reference_person_instances']}</td><td>{r['person_points_covered']}/{r['reference_person_points']}</td><td>{r['person_points_covered_by_pallet']}</td></tr>")
    rows.append('</table><p>地面等类别没有完整二维真值。互斥全景图按事先列明的同一优先级导出，仅供检查；主要计分仍使用导出前的原候选，不用渲染结果冒充提升。</p></section><section><h2>Agent实际行为</h2><table><tr><th>图</th><th>方案</th><th>轮数</th><th>结束原因</th><th>实际编辑</th></tr>')
    for r in result['agent_runs']:rows.append(f"<tr><td>{r['case']}</td><td>{r['arm']}</td><td>{r['rounds']}</td><td>{r['stop_reason']}</td><td>{r['applied_edits']}</td></tr>")
    rows.append('</table><p>零编辑不代表纠错成功；未收敛或错误如实保留。SAM3补漏与模型本身检出分开解释。</p></section><section><h2>视频</h2>')
    for r in receipts:rows.append(f"<p>{r['path']} · {r['duration_seconds']}秒</p><video controls preload='metadata' src='{r['path']}'></video>")
    rows.append('</section><section><h2>同图、阶段和固定分区细节</h2>')
    for file in gallery:relative=file.relative_to(out).as_posix();rows.append(f'<p>{file.stem}</p><a href="{relative}"><img loading="lazy" src="{relative}"></a>')
    rows.append('</section><section><p>原图、权重、输入清单及保护状态以completed.json、frozen_plan.json、cache_source_binding.json、model_trace.json核对。所有输出仍未确认为真值；没有自动采用或写回CVAT。</p></section>')
    (out/'index.html').write_text('\n'.join(rows),encoding='utf-8');print(json.dumps({'report_complete':True,'pictures':len(gallery),'videos':len(receipts)}),flush=True)


if __name__=='__main__':main()
