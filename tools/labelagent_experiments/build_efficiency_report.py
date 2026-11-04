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
                ('05 原SAM掩码保留＋范围内实例分离',panoptic(image,seed['C_hybrid'])),
                ('06 修复边界与批量缺口核验后',rendered['C_hybrid'])]
        for index,(title,view) in enumerate(stages):
            file=assets/(case+f'_stage_{index+1:02d}.jpg');stage_card(case,title,image,view,file);replays.append((file,4));gallery.append(file)
        panels=[('原图',image),('A 现有两类',rendered['A_current']),('B SAM局部实例＋修复',rendered['B_sam7']),('C 训练模型裁剪提示＋修复',rendered['C_hybrid'])]
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
    for number,original in enumerate(sorted((efficiency/'diagnosis').glob('*.jpg')),1):
        file=assets/('old_addition_'+original.name);shutil.copy2(original,file);gallery.append(file)
        if number<=5:diagnosis_video.append((file,3))
    for number,original in enumerate(sorted((Path(cfg['instance_generation'])/'visual_cache').glob('*.png')),1):
        file=assets/('instance_verification_'+original.name);shutil.copy2(original,file);gallery.append(file)
    for number,original in enumerate(sorted((out/'gap_action_cache').glob('*_sam_comparison_*.png')),1):
        file=assets/('actual_SAM_before_after_'+original.name);shutil.copy2(original,file);gallery.append(file)
        if number<=8:diagnosis_video.append((file,3))
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
    old=read(efficiency/'diagnosis/completed.json')
    pilot=read(out.parent/'labelagent_efficiency_comparison_20261104/completed.json')
    budget_rows=[]
    for arm in ('A_current','B_sam7','C_hybrid'):
        for frame,camera in cfg['cases']:
            case=f'camera{camera}_frame{frame}'
            r=read(out/arm/case/'result.json')
            for audit in r['agent']['audit']:
                ops=audit['decision']['operations']
                budget_rows.append({'case':case,'arm':arm,'round':audit['round'],'operations':len(ops),
                    'SAM_points':max([len(o.get('points',[])) for o in ops]+[0])})
    assert all(r['operations']<=3 and r['SAM_points']<=32 and r['round']<=3 for r in budget_rows)
    save(out/'budget_audit.json',{'passed':True,'rows':budget_rows,'limits_unchanged':True})
    public['generation']=read(Path(cfg['instance_generation'])/'completed.json')
    public['negative_hint_pilot_converged']=sum(bool(r['converged']) for r in pilot['agent_runs'])
    public['reference_used_for_instance_selection']=False
    public['diagnostic_reference_used_only_retrospectively']=True
    public['all_warning_rejection_changed_only_for_complete_approved_gaps']=True
    save(out/'public_summary.json',public)
    rows=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>托盘实例、面积与三轮收敛优化</title>',
      '<style>body{max-width:1450px;margin:30px auto;padding:20px;font:17px/1.7 "Microsoft YaHei",sans-serif;background:#f3f6fa;color:#152438}img,video{max-width:100%}section{padding:24px;background:white;border-radius:12px;margin:20px 0}td,th{padding:10px;border:1px solid #cdd5df}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>',
      '<h1>托盘实例、面积与三轮收敛优化</h1>',
      f'<section><b>未放宽验收：全部条件 {"通过" if acceptance["all_conditions_passed"] else "未通过"}。托盘支持 {c["pallet_instances_supported"]}/{c["reference_pallet_instances"]}，PAL并集面积 {area_ratio:.4f}倍（上限1.25），三轮内收敛 {converged}/6。没有自动采用。</b></section>',
      '<section><h2>原因与修复</h2>',
      '<p>旧版11个新增托盘掩码没有增加新的合格参考对象，部分增加了同一对象外围范围。候选数量不等于新实例支持；按原V2规则还必须满足单实例点覆盖0.5、点纯度0.67和一对一匹配。没有把未标注二维区域当成误检真值。</p>',
      '<p>新实例分离使用37次真实局部SAM3调用、16次原图类别核验。B使用原SAM候选裁剪，C再加入训练过的Mask2Former固定四分区裁剪提示。新掩码裁在原SAM托盘并集内，原掩码全部保留；按冻结规则排重、限量。它能控制新增面积，但这种保守方法无法恢复原并集外的空间漏检。原图核验来自模型，不是人工真值。</p>',
      f'<p>旧33次填补中，{old["operations_with_incomplete_target_polygon"]}次操作框没有完整包含目标，{old["operations_leaving_reviewed_target_pixels_missing"]}次实际编辑后仍留目标像素。缺口框右、下界是闭区间，旧算法误按半开区间处理，留下边条并在后续重复。新算法补齐闭区间；原有分数和验收阈值不变。</p>',
      '<p>首轮已批准的地面任务为8、18、14、26个，旧每轮最多3次编辑，本来就可能超出三轮。新流程用一次真实SAM3多点操作处理一组已逐项看过的缺口，只有被实际SAM掩码100%包含的完整缺口才可写入候选。部分覆盖的缺口整块不接纳并显式待办；所有未批准区域保持像素锁。下一轮必须由真实模型比较前后图。</p>',
      '<p>第一次批处理把“不确定、保留”误作为负提示，触发负点排除失败；这次改成不确定区域锁定、不给负提示。预检只基于实际SAM满足缺口数选择正提示模式，不使用3D参考分数。正点未全覆盖警告原样保留，不能使整批冒充成功；其他警告仍拒绝。少量剩余任务继续按原单项工具修复，每轮仍最多3项，最多3轮，SAM累计点最多32。</p>',
      '<p>原始两类A复用身份绑定的既有真实结果；B/C重新运行修复后的Agent。训练模型权重、原图、原始参考标注未改，标注写入0、训练更新0。CPU模型历史推理与新SAM/VLM调用分别留痕；没有把缓存重放计成新的模型推理。</p></section>',
      '<section><h2>共同类别参考一致性</h2><p>仅两张已暴露同采集图；原V2半径3、至少5点、纯度0.67、支持0.5及一对一规则均未改变。这不是二维像素准确率；地面缺少完整二维真值。</p><table><tr><th>方案</th><th>托盘支持</th><th>托盘参考点</th><th>人员参考点</th><th>人员被托盘覆盖点</th></tr>']
    for r in compact:rows.append(f'<tr><td>{r["arm"]}</td><td>{r["pallet_instances_supported"]}/{r["reference_pallet_instances"]}</td><td>{r["pallet_points_covered"]}/{r["reference_pallet_points"]}</td><td>{r["person_points_covered"]}/{r["reference_person_points"]}</td><td>{r["person_points_covered_by_pallet"]}</td></tr>')
    rows.append('</table></section><section><h2>六组真实复核</h2><table><tr><th>图</th><th>方案</th><th>轮数</th><th>终止原因</th><th>实际编辑</th></tr>')
    for r in result['agent_runs']:rows.append(f'<tr><td>{r["case"]}</td><td>{r["arm"]}</td><td>{r["rounds"]}</td><td>{r["stop_reason"]}</td><td>{r["applied_edits"]}</td></tr>')
    rows.append('</table><p>首次负提示试跑也完整保留；它仅2/6收敛。达到三轮上限不能算通过；零协议错误不能代替收敛。没有为了凑齐通过而提高轮数或把多次普通填补伪装成一次。</p></section><section><h2>原验收条件</h2><table><tr><th>条件</th><th>结果</th></tr>')
    for key,value in checks.items():rows.append(f'<tr><td>{key}</td><td>{"通过" if value else "未通过"}</td></tr>')
    rows.append(f'</table><p>数量比例{count_ratio:.4f}；并集面积比例{area_ratio:.4f}。本轮候选不部署为默认，不写回CVAT。</p></section><section><h2>过程视频</h2><p>下列为真实保存阶段和送审图片的结果回放，不是连续桌面录屏，没有合成预测证据。</p>')
    for r in receipts:rows.append(f'<p>{r["path"]} · {r["duration_seconds"]}秒</p><video controls preload="metadata" src="{r["path"]}"></video>')
    rows.append('</section><section><h2>图片与局部细节</h2>')
    for file in gallery:
        rel=file.relative_to(out).as_posix();rows.append(f'<p>{html.escape(file.stem)}</p><a href="{rel}"><img loading="lazy" src="{rel}"></a>')
    rows.append('</section><section><h2>复现与核验</h2><p>frozen_plan.json、protocol_amendment.json、cache_source_binding.json、model_trace.json记录配置、来源及真实调用。completed.json、acceptance_checks.json、budget_audit.json保留原验收及所有失败项。dense_exports仅为有明确优先级的全景候选展示，不是训练真值。</p></section></html>')
    (out/'index.html').write_text('\n'.join(rows),encoding='utf-8')
    print(json.dumps({'report_complete':True,'pictures':len(gallery),'videos':len(receipts),'acceptance_passed':acceptance['all_conditions_passed']}),flush=True)

if __name__=='__main__':main()
