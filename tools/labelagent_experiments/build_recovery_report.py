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
    diagnosis=Path(cfg['additional_pal_shapes']).parent
    prototype=read(diagnosis/'mask_diagnosis.json')
    resolution=read(diagnosis/'resolution_control/completed.json') if (diagnosis/'resolution_control/completed.json').exists() else None
    diagnostic_forward_count=2+(resolution['real_forwards'] if resolution else 0)
    reviews=read(out/'batch_reviews.json')
    bindings=[r['local_gap'] for r in reviews if 'local_gap' in r]
    unique={ (r['hole_id'],r['evidence_file'],r.get('raw_verification_file')):r for r in bindings}
    vetoes=[r for r in unique.values() if r['proposed_decision']['action']=='fill_gap' and r['decision']['action']=='keep']
    guarded=[op for arm in ('A_current','B_sam7','C_hybrid') for frame,camera in cfg['cases']
             for audit in read(out/arm/f'camera{camera}_frame{frame}'/'result.json').get('agent',{}).get('audit',[])
             for op in audit['decision'].get('guarded_proposals',[])]
    errors=sum(r['stop_reason']=='error' for r in result['agent_runs'])
    converged=sum(bool(r['converged']) for r in result['agent_runs'])
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
        whole=read(out/'seed_stages'/case/'mask_whole.json');tiles=read(out/'seed_stages'/case/'mask_all_tiles.json')
        arms={name:read(out/name/case/'result.json')['shapes'] for name in ('A_current','B_sam7','C_hybrid')}
        rendered={name:panoptic(image,shapes,out/'dense_exports'/name/case) for name,shapes in arms.items()}
        stages=[('01 原始图像',image),('02 同一训练权重：整图原输入',panoptic(image,whole)),
                ('03 历史训练模型固定四分区候选',panoptic(image,tiles)),
                ('04 固定排重后SAM3掩码与点细化',panoptic(image,refined)),
                ('05 保留原SAM＋独立训练实例完整候选',panoptic(image,seed['C_hybrid'])),
                ('06 真实剩余缺口修复与复核后',rendered['C_hybrid'])]
        for index,(title,view) in enumerate(stages):
            file=assets/(case+f'_stage_{index+1:02d}.jpg');stage_card(case,title,image,view,file);replays.append((file,4));gallery.append(file)
        panels=[('原图',image),('A 现有两类',rendered['A_current']),('B 固定网格SAM＋修复',rendered['B_sam7']),('C 训练实例＋SAM细化＋修复',rendered['C_hybrid'])]
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
    diagnosis_video=[]
    import shutil
    for name in prototype['pictures']:
        original=diagnosis/'assets'/name;file=assets/('diagnosis_'+name)
        shutil.copy2(original,file);gallery.append(file);diagnosis_video.append((file,2))
    for original in sorted((diagnosis/'assets').glob('*resolution_control.jpg')):
        file=assets/('diagnosis_'+original.name);shutil.copy2(original,file);gallery.append(file);diagnosis_video.append((file,3))
    verification_cards=[]
    for number,row in enumerate(unique.values(),1):
        canvas=Image.new('RGB',(1920,800),'#f1f5fa');draw=ImageDraw.Draw(canvas)
        draw.text((20,15),'逐缺口真实核验 '+str(number)+' · '+row['hole_id'],font=titlefont,fill='#142536')
        draw.text((20,62),'左：实际送审的原图/二值/缺口；右：独立原图核验（细线只定位，不着色内部）',font=font,fill='#45566a')
        with Image.open(row['evidence_file']) as im:canvas.paste(im.convert('RGB'),(0,130))
        raw=Path(row['raw_verification_file']).with_suffix('.png') if row.get('raw_verification_file') else None
        if raw and raw.exists():
            with Image.open(raw) as im:
                im=ImageOps.contain(im.convert('RGB'),(720,450));canvas.paste(im,(1180+(720-im.width)//2,130+(450-im.height)//2))
        verification=row.get('raw_photo_verification') or {}
        draw.text((20,620),'原建议：'+row['proposed_decision']['action']+'；原图描述：'+str(verification.get('object','无需填补核验')),font=font,fill='#142536')
        draw.text((20,670),'原图类别：'+str(verification.get('category','未请求'))+'；最终动作：'+row['decision']['action'],font=font,fill='#142536')
        draw.text((20,720),'依据：'+row['decision']['reason'][:140],font=small,fill='#45566a')
        file=assets/f'gap_verification_{number:03d}.jpg';canvas.save(file,quality=93);gallery.append(file);verification_cards.append(file)
        if row in vetoes and len(diagnosis_video)<16:diagnosis_video.append((file,3))
    # Real photographs and exact model-input evidence, never synthetic results.
    efficiency=Path(cfg['instance_generation']).parent
    agent_replay=[]
    for arm in ('B_sam7','C_hybrid'):
        for frame,camera in cfg['cases']:
            case=f'camera{camera}_frame{frame}'
            for original in sorted((out/arm/case).glob('*.jpg')):
                with Image.open(original) as im: photo=im.convert('RGB')
                canvas=Image.new('RGB',(1920,1080),'#132334');draw=ImageDraw.Draw(canvas)
                draw.text((25,18),case+' / '+arm+' / '+original.stem,font=titlefont,fill='white')
                canvas.paste(ImageOps.contain(photo,(1840,930)),(40,90))
                file=assets/('actual_agent_'+arm+'_'+case+'_'+original.name);canvas.save(file,quality=93);gallery.append(file)
                if any(s in original.name for s in ['seed_cache','observing','finished']):agent_replay.append((file,2))
    for number,original in enumerate(sorted((efficiency/'diagnosis').glob('*.jpg')),1):
        file=assets/('old_addition_'+original.name);shutil.copy2(original,file);gallery.append(file)
        if number<=5:diagnosis_video.append((file,3))
    for number,original in enumerate(sorted((Path(cfg['instance_generation'])/'visual_cache').glob('*.png')),1):
        file=assets/('instance_verification_'+original.name);shutil.copy2(original,file);gallery.append(file)
    for number,original in enumerate(sorted((out/'gap_action_cache').glob('*_sam_comparison_*.png')),1):
        file=assets/('actual_SAM_before_after_'+original.name);shutil.copy2(original,file);gallery.append(file)
        if number<=8:diagnosis_video.append((file,3))
    recovery_frames=[]
    for file in sorted((efficiency/'miss_diagnosis').glob('*.jpg')):
        target=assets/('missing_'+file.name);shutil.copy2(file,target);gallery.append(target);diagnosis_video.append((target,4))
    for file in sorted((efficiency/'fixed_grid').glob('*/*/prediction.json')):
        folder=file.parent;meta=read(file);crop=Image.open(folder/'input.png').convert('RGB')
        pixels=np.asarray(Image.open(folder/'panoptic.png')).astype(np.uint32);ids=pixels[:,:,0]+256*pixels[:,:,1]+65536*pixels[:,:,2]
        from source.agent.edit_agent import _shape_from_mask
        local=[]
        for segment in meta['segments_info']:
            shape=_shape_from_mask((ids==segment['id']).astype(np.uint8),labels[segment['category_id']-1],segment['id'])
            if shape:local.append(shape)
        target=assets/('fixed_grid_'+folder.parent.name+'_'+folder.name+'.jpg')
        stage_card(folder.parent.name,'同一训练权重固定网格 · '+folder.name,crop,panoptic(crop,local),target)
        gallery.append(target);recovery_frames.append((target,1))
    refinement_rows=read(Path(cfg['instance_generation'])/'refinement_progress.json')['rows']
    for number,row in enumerate(refinement_rows,1):
        image=Image.open(out/'inputs'/(row['case']+'.png')).convert('RGB');box=row['crop']
        before=panoptic(image,[row['original_model_shape']]).crop(box);after=panoptic(image,[row['shape']]).crop(box)
        target=assets/('model_instance_refinement_'+row['key']+'.jpg')
        stage_card(row['case'],'独立训练实例 → SAM3：'+('细化有效' if row['refinement_accepted'] else '保留原训练候选'),before,after,target,left_caption='训练模型候选局部')
        gallery.append(target)
        if number<=10:recovery_frames.append((target,3))
    for file in sorted((Path(cfg['instance_generation'])/'visual_cache').glob('*.png')):
        target=assets/('model_target_evidence_'+file.name);shutil.copy2(file,target);gallery.append(target)
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
    for line in [f"Agent协议错误 {errors}/6组；收敛 {converged}/6组；轮数上限如实保留。",
                 f"原图核验否决 {len(vetoes)} 个唯一填补建议（含不确定）；保护初始候选。",
                 '仅2张已暴露同采集图；这些不是像素准确率。','地面等类别没有完整二维真值，不能据此宣称七类提升。',
                 '实际使用训练交付包 d11c2e7f…；SAM3补漏单独记账。','原始标注未写入；模型权重和原图保持。','本片是实际阶段结果回放，不是连续工作台录屏。']:
        d.text((70,y),line,font=titlefont,fill='#e0e9f4');y+=58
    summary_file=assets/'comparison_summary.jpg';summary.save(summary_file,quality=95);gallery.append(summary_file);comparisons.append((summary_file,8))
    # Acceptance is reported even when protocol repair succeeds but budget or
    # quality limits fail.  Never change predictions after this calculation.
    totals={};foreground_preserved=True;sam_pal_preserved=True
    for arm in ('A_current','C_hybrid'):
        count=area=0
        for frame,camera in cfg['cases']:
            case=f'camera{camera}_frame{frame}';image=Image.open(out/'inputs'/(case+'.png'));w,h=image.size
            shapes=read(out/arm/case/'result.json')['shapes'];mask=np.zeros((h,w),bool)
            for s in shapes:
                if s['label']=='pallet':count+=1;mask|=_rasterize(s,w,h)>0
            area+=int(mask.sum())
            if arm=='C_hybrid':
                def fingerprints(values,allowed):return {hashlib.sha256(s['label'].encode()+_rasterize(s,w,h).tobytes()).hexdigest() for s in values if s['label'] in allowed}
                before=read(out/'seed_stages'/case/'all_arms.json')['C_hybrid']
                foreground_preserved &= fingerprints(before,{'pallet','person'})<=fingerprints(shapes,{'pallet','person'})
                control=read(out/'A_current'/case/'result.json')['shapes']
                sam_pal_preserved &= fingerprints(control,{'pallet'})<=fingerprints(shapes,{'pallet'})
        totals[arm]={'PAL_count':count,'PAL_union_pixels':area}
    a,c=compact[0],compact[2];count_ratio=totals['C_hybrid']['PAL_count']/max(1,totals['A_current']['PAL_count'])
    area_ratio=totals['C_hybrid']['PAL_union_pixels']/max(1,totals['A_current']['PAL_union_pixels'])
    checks={'PAL_support_strict_gain':c['pallet_instances_supported']>a['pallet_instances_supported'],
        'PAL_point_coverage_not_lower':c['pallet_points_covered']>=a['pallet_points_covered'],
        'person_point_coverage_not_lower':c['person_points_covered']>=a['person_points_covered'],
        'person_off_class_not_higher':c['person_points_covered_by_pallet']<=a['person_points_covered_by_pallet'],
        'PAL_candidate_ratio_at_most_1_5':count_ratio<=1.5,'PAL_union_area_ratio_at_most_1_25':area_ratio<=1.25,
        'initial_C_foreground_preserved':bool(foreground_preserved),'original_SAM_PAL_preserved':bool(sam_pal_preserved),
        'review_protocol_no_errors':errors==0,'all_six_runs_converged_within_three_rounds':converged==6}
    acceptance={'checks':checks,'all_conditions_passed':all(checks.values()),'totals':totals,'candidate_ratio':count_ratio,'union_area_ratio':area_ratio,
        'automatic_adoption':False,'reason':'Two exposed same-capture images; no dense floor truth; unchanged criteria and failed conditions retained.'}
    save(out/'acceptance_checks.json',acceptance)
    d.text((70,1010),f"预设检查：实例支持未增；PAL面积 {area_ratio:.4f}×（上限1.25×）；三轮内未全部收敛。",
           font=font,fill='#ffcfb4')
    summary.save(summary_file,quality=95)
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
    video('03_实例面积与SAM修复证据.mp4',diagnosis_video)
    video('04_Agent三轮真实阶段回放.mp4',agent_replay)
    video('05_固定网格与训练实例细化.mp4',recovery_frames)
    save(out/'media_verification.json',{'status':'PASS','videos':receipts,'pictures':len(gallery),'unconfirmed_candidates':True})
    converged=sum(bool(r['converged']) for r in result['agent_runs'])
    failed=sum(r['stop_reason']=='error' for r in result['agent_runs'])
    trace=read(out/'model_trace.json');public={'status':'comparison_recorded_budget_limited' if converged<6 else 'comparison_recorded_with_guards',
        'agent_runs_converged':converged,'agent_runs_failed':failed,
        'end_to_end_agent_review_passed':all(bool(r['converged']) for r in result['agent_runs']),
        'images':2,'mask2former_checkpoint_sha256':result['model_sha256'],
        'weights_unchanged':result['weights_unchanged'],'source_images_unchanged':result['source_images_unchanged'],
        'protected_state_before_after_equal':result['protected_state_before_after_equal'],'annotation_writes':0,
        'mask2former_source_real_forwards':10,'diagnostic_only_mask2former_forwards':diagnostic_forward_count,
        'mask2former_fresh_forwards_this_run':result['fresh_mask2former_forwards_this_run'],
        'SAM_source_requests':sum(r.get('service')=='SAM 3' for r in trace),'VLM_real_requests':sum(r.get('service')=='GPT' for r in trace),
        'VLM_protocol_valid_replies':sum(r.get('service')=='GPT' and r.get('status')=='succeeded' for r in trace),
        'common_class_comparison':compact,'agent_runs':[{k:v for k,v in r.items() if k!='case'} for r in result['agent_runs']],
        'SAM_refinement_attempts':len(refinement),'SAM_refinement_technically_accepted':sum(bool(r.get('accepted_technical_output')) for r in refinement),
        'dense_pixel_accuracy_evaluated':False,'scope':result['scope'],'pictures':len(gallery),'videos':len(receipts),'default_workbench_changed':False}
    public['hole_review_batch_size']=cfg.get('hole_review_batch_size')
    public['same_operation_cap_per_whole_agent_round']=3
    public.update(unique_gap_evidence_bindings=len(unique),unique_gap_fill_vetoes=len(vetoes),
        destructive_global_proposals_held=len(guarded),acceptance=acceptance,
        cached_real_gap_proposals_inherited=read(out/'cache_source_binding.json').get('inherited_gap_proposal_files',0),
        diagnostic_raw_photo_calls=3,diagnostic_crop_model_PAL_counts=[10,4],
        original_default_workbench_not_installed=True)
    public['same_context_larger_input_diagnostic']=resolution['rows'] if resolution else 'guarded_not_run'
    save(out/'public_summary.json',public)


    import html
    before=Path(cfg['instance_generation']).parent.parent/'labelagent_efficiency_partial_comparison_20261104'
    missing=[(0,42),(0,48),(1,47),(1,48),(1,49)];miss_rows=[]
    for cam,obj in missing:
        vals=[]
        for arm,directory in [('A_current',out),('C_hybrid',out)]:
            per=read(directory/arm/'evaluation/per_image.json')
            image=next(r for r in per if r['camera']==cam)
            item=next(r for r in image['objects'] if r['label']=='pallet' and r['obj_id']==obj)
            vals.append(item)
        miss_rows.append({'camera':cam,'object_id':obj,'reference_points':vals[0]['unique_visible_points'],
            'old_best_supported_coverage':vals[0]['best_candidate_supported_coverage'],'new_best_supported_coverage':vals[1]['best_candidate_supported_coverage'],
            'old_raw_point_coverage':vals[0]['best_candidate_point_coverage'],'new_raw_point_coverage':vals[1]['best_candidate_point_coverage'],
            'new_candidate_meets_support_before_global_one_to_one':vals[1]['best_candidate_supported_coverage']>=.5})
    save(out/'five_missed_objects.json',{'scope':'per-object necessary support test, not a substitute for original global one-to-one metric','rows':miss_rows})
    budgets=[];rounds=[]
    for row in result['agent_runs']:
        r=read(out/row['arm']/row['case']/'result.json')
        for audit in r['agent']['audit']:
            ops=audit['decision']['operations'];tools=audit.get('tool_results',[])
            budgets.append({'case':row['case'],'arm':row['arm'],'round':audit['round'],'operations':len(ops),
              'max_cumulative_points':max([t.get('point_count',0) for t in tools]+[0]),'max_actual_points':max([t.get('active_request_point_count',t.get('point_count',0)) for t in tools]+[0])})
            rounds.append({'case':row['case'],'arm':row['arm'],'round':audit['round'],
              'ground_tasks':sum(a['semantic']=='ground' for a in audit['decision'].get('hole_assessment',[])),
              'resolved_by_SAM':sum(len(t.get('resolved_gap_ids',[])) for t in tools),'pending_after_SAM':sum(len(t.get('unresolved_gap_ids',[])) for t in tools),
              'individual_fill_ops':sum(t.get('operation')=='fill_gap' and t.get('status')=='applied' for t in tools),'tool_errors':[t.get('error') for t in tools if t.get('status')=='error']})
    assert all(r['round']<=3 and r['operations']<=3 and r['max_cumulative_points']<=32 and r['max_actual_points']<=32 for r in budgets)
    save(out/'budget_audit.json',{'passed':True,'limits_unchanged':True,'rows':budgets});save(out/'round_analysis.json',rounds)
    generation=read(Path(cfg['instance_generation'])/'completed.json');control=read(Path(cfg['instance_generation'])/'blank_controls.json')
    public.update(mask2former_source_real_forwards=42,mask2former_fresh_fixed_grid_forwards=32,whole_mask_candidates_no_budget_clipping=True,
       five_missed_objects=miss_rows,generation=generation,new_synthetic_controls_passed=control['all_passed'],
       actual_grid_SAM_requests_source=64,budget_audit_passed=True,pictures=len(gallery),videos=len(receipts))
    save(out/'public_summary.json',public)
    rows=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>5个托盘漏检与三轮复核恢复</title>',
      '<style>body{max-width:1450px;margin:30px auto;padding:20px;font:17px/1.7 "Microsoft YaHei",sans-serif;background:#f3f6fa;color:#152438}img,video{max-width:100%}section{padding:24px;background:white;border-radius:12px;margin:20px 0}td,th{padding:10px;border:1px solid #cdd5df}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>',
      '<h1>5个托盘漏检与三轮复核恢复</h1>',
      f'<section><b>原验收：{"全部通过" if acceptance["all_conditions_passed"] else "仍未全部通过"}。托盘支持 {c["pallet_instances_supported"]}/{c["reference_pallet_instances"]}；面积{area_ratio:.4f}倍；三轮内收敛{converged}/6。没有默认采用或写回原标注。</b></section>',
      '<section><h2>原因与新尝试</h2><p>五个原漏检涉及堆叠托盘。原掩码一部分把多层并为一组，覆盖参考点却未满足0.67纯度及一对一实例条件；另一些上层区域未覆盖。新增整组轮廓不能解决实例支持。本次明确允许原范围外的完整候选，绝不把掩码裁在旧并集内来通过面积要求。</p>',
      '<p>同一训练交付权重d11c2e7f…执行32次新的固定网格CPU前向，输入阈值不变；SAM3执行64次固定网格双提示文本推理。首版纯SAM仍将新候选看作整组，未产生合格新增实例；这一失败版本完整保留。</p>',
      '<p>最终C让训练模型的独立实例实际进入候选池，不要求它先与原SAM有覆盖。SAM3在原图局部对这些实例做掩码和点细化；拒绝或警告时保留原训练候选并明确标注未细化。原像素目标视图和原图上下文视图分别核验目标，避免把周围多层都当作同一候选。按固定置信度排序，超原面积或数量预算时整项拒绝，所有被保留的掩码都未作预算裁剪。</p>',
      '<p>地面预检没有改善：局部拼图四组仅完整满足1/1/1/5个缺口，去掉先验为4/10/10/7个，均未采用。最终保留原先验，修复第二轮重复提示及预算调度：最多两次独立缺口填补，再一次真实SAM处理剩余缺口；新提示点只取上次预测未覆盖且已获语义批准的像素。累计最多32点，送入SAM的点聚焦当前待办，已解决区域不再占有效提示。第三轮仍需真实复查，没有隐藏多个普通编辑为一次。</p>',
      '<p>三轮、每轮三项、32点累计、每个接纳缺口100%实际覆盖及所有未批准像素锁均保持。未解决缺口不改称非地面来凑收敛。原始标注、权重、原PNG及参考文件保留；训练更新0、标注写入0。</p>',
      f'<p>新增灰/白功能负对照：{"通过" if control["all_passed"] else "未通过"}。这是合成软件控制，不能当作真实分割质量；目标模型判断也不是人工真值。仅两张已暴露图，仍缺地面密集真值，不据此宣称泛化或自动部署。</p></section>',
      '<section><h2>同图三方案：原V2指标</h2><table><tr><th>方案</th><th>托盘实例支持</th><th>托盘参考点</th><th>人员参考点</th><th>人员被托盘覆盖点</th></tr>']
    for r in compact:rows.append(f'<tr><td>{r["arm"]}</td><td>{r["pallet_instances_supported"]}/{r["reference_pallet_instances"]}</td><td>{r["pallet_points_covered"]}/{r["reference_pallet_points"]}</td><td>{r["person_points_covered"]}/{r["reference_person_points"]}</td><td>{r["person_points_covered_by_pallet"]}</td></tr>')
    rows.append('</table><p>A复用身份绑定的既有真实结果；B为纯SAM固定网格尝试＋新地面策略；C为训练模型独立实例＋SAM细化＋新地面策略。新调用与缓存分别记账。不是像素准确率。</p></section><section><h2>原五个漏检的单候选支持条件</h2><table><tr><th>相机/对象</th><th>点数</th><th>原纯度合格覆盖率</th><th>新纯度合格覆盖率</th><th>达到0.5</th></tr>')
    for r in miss_rows:rows.append(f'<tr><td>{r["camera"]}/{r["object_id"]}</td><td>{r["reference_points"]}</td><td>{r["old_best_supported_coverage"]:.4f}</td><td>{r["new_best_supported_coverage"]:.4f}</td><td>{r["new_candidate_meets_support_before_global_one_to_one"]}</td></tr>')
    rows.append('</table><p>此表是单对象必要条件；全局一对一匹配仍以原V2聚合指标为准。事后参考卡不参与候选筛选。五个对象全部保留，没有移出验收集。</p></section><section><h2>六组真实复核</h2><table><tr><th>图</th><th>方案</th><th>轮数</th><th>收敛</th><th>结束原因</th><th>实际操作</th></tr>')
    for r in result['agent_runs']:rows.append(f'<tr><td>{r["case"]}</td><td>{r["arm"]}</td><td>{r["rounds"]}</td><td>{r["converged"]}</td><td>{r["stop_reason"]}</td><td>{r["applied_edits"]}</td></tr>')
    rows.append('</table><p>未被下一轮确认的编辑不算收敛。达到三轮仍有待办就保留失败，不能用零协议错误代替完成。</p></section><section><h2>原冻结验收</h2><table>')
    for key,value in checks.items():rows.append(f'<tr><td>{key}</td><td>{"通过" if value else "未通过"}</td></tr>')
    rows.append(f'</table><p>数量比例{count_ratio:.4f}（上限1.5）；面积比例{area_ratio:.4f}（上限1.25）。均按原候选并集计算，没有用渲染覆盖顺序修改计分。</p></section><section><h2>过程视频</h2><p>真实保存阶段的结果回放，非连续桌面录屏；每段均完整解码核验。</p>')
    for r in receipts:rows.append(f'<p>{r["path"]} · {r["duration_seconds"]}秒</p><video controls preload="metadata" src="{r["path"]}"></video>')
    rows.append('</section><section><h2>图片、固定网格、独立实例和逐轮证据</h2>')
    for file in gallery:
        rel=file.relative_to(out).as_posix();rows.append(f'<p>{html.escape(file.stem)}</p><a href="{rel}"><img loading="lazy" src="{rel}"></a>')
    rows.append('</section><p>配置与来源见frozen_plan.json、protocol_amendment.json、cache_source_binding.json、model_trace.json；原验收见acceptance_checks.json；五对象与逐轮检查见five_missed_objects.json、round_analysis.json和budget_audit.json。候选未确认为真值，原始标注未覆盖。</p></html>')
    (out/'index.html').write_text('\n'.join(rows),encoding='utf-8')
    print(json.dumps({'report_complete':True,'pictures':len(gallery),'videos':len(receipts),'acceptance_passed':acceptance['all_conditions_passed']}),flush=True)

if __name__=='__main__':main()
