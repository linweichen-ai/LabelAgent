"""Real source-anchor, new gap-instance and matched reviewer-layout evidence."""
import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
from run_rgb_layers import ROOT,read,save,sha
from experiment_report_media import Report,overlay
from instance_identity_consistency import single_physical_base
from semantic_subject_guard import compatible_pallet_subject
import run_anchor_bracket_prior as original
import run_source_gap_instances as gap
import review_structure_layout as layout
BASE=original.BASE;ONE=original.OUT;TWO=BASE/'labelagent_anchor_image_extent_prior_20261108';OUT=BASE/'labelagent_anchor_gap_report_20261108'
def main():
    assert all(read(p/'independent_readback.json')['passed'] for p in [ONE,TWO,gap.OUT]);review=read(layout.OUT/'completed.json');lp=read(layout.OUT/'frozen_plan.json');assert all(sha(f)==h for f,h in lp['protected'].items());checks=[]
    for r in review['rows']:
        d=read(r['response']);assert sha(r['response'])==r['response_sha256'] and d['local_vlm']['model_config_sha256']==lp['model_config_sha256'] and d['local_vlm']['actual_image_token_counts']==[256] and json.loads(d['choices'][0]['message']['content'])==r['decision'] and single_physical_base(r['decision'])==r['single_base'];request=next(x for x in lp['requests'] if x['request_sha256']==r['request_sha256']);assert hashlib.sha256(json.dumps(request['payload'],sort_keys=True).encode()).hexdigest()==r['request_sha256'];checks.append({'layout_request':r['request_sha256'],'raw_reply_model_tokens_and_original_parser_exact':True})
    for binding in lp['bindings']:
        group=[r for r in review['rows'] if r['candidate']==binding['candidate'] and r['view']==binding['view']]
        for kind in ['real','gray','white']:
            square=next(r for r in group if r['kind']==kind and r['layout']=='square');aspect=next(r for r in group if r['kind']==kind and r['layout']=='aspect');np.testing.assert_array_equal(np.asarray(layout.render(Image.open(square['image']).convert('RGB'),binding)),np.asarray(Image.open(aspect['image']).convert('RGB')))
        a,b,c,d=binding['crop'];x,y,z,w=binding['full_original_photo_rectangle_in_square'];assert a<=x and b<=y and c>=z and d>=w
    sem=read(gap.OUT/'semantic/completed.json');sp=read(gap.OUT/'semantic/frozen_plan.json');assert all(sha(f)==h for f,h in sp['protected'].items());assert not any(c['all_semantic_stages_passed'] for c in sem['candidates']) and not any(r['stage']=='extent' for r in sem['rows'])
    for r in sem['rows']:
        reply=read(r['response']);assert sha(r['response'])==r['response_sha256'] and reply['local_vlm']['model_config_sha256']==sp['model_config_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256]*len(r['images'])
    g1=read(ONE/'generation_completed.json');g2=read(TWO/'generation_completed.json');gg=read(gap.OUT/'SAM/generation_completed.json');fresh=g1['actual_new_SAM_calls']+g2['actual_new_SAM_calls']+gg['actual_new_SAM_calls'];vlm=read(gap.OUT/'review_completed.json')['actual_new_VLM_calls']+sem['fresh_this_execution']+review['actual_new_VLM_calls'];assert fresh==11 and vlm==32
    all_native=[];audits={}
    for trial,generation,arms in [(ONE,g1,['original','bracket']),(TWO,g2,['original','bracket']),(gap.OUT,gg,['point_only','RGB_gap_prior'])]:
        for r in generation['rows']:all_native.append((trial,r))
        for arm in arms:
            d=read(trial/'audits'/arm/'support_area_audit.json');assert all(r['old_matches']==r['geometry_area_person_valid_matches']==r['unfiltered_all_matches'] for c in d['cases'] for r in c['radii']);audits[(str(trial),arm)]=d
    binding=read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'];assert all(sha(ROOT/p)==h for p,h in binding.items());cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');protected=read(ROOT/'outputs/labelagent_soft_prior_20261108/frozen_plan.json')['protected'];assert all(sha(cfg[k])==protected[cfg[k]] for k in ['reference_annotations','reference_description']);report=Report(OUT)
    p1=read(ONE/'frozen_plan.json');p2=read(TWO/'frozen_plan.json')
    for detail in p2['parents']:
        n=detail['parent'];old=next(d for d in p1['parents'] if d['parent']==n);rows=[r for r in g2['rows'] if r['parent']==n];a=next(r for r in rows if r['arm']=='original');b=next(r for r in rows if r['arm']=='bracket');photo=Image.open(a['input']).convert('RGB');card=report.canvas(f'跨裁剪源锚点 {n}：先验可增加可见区域，不按面积反推')
        report.panel(card,0,Image.open(TWO/'parents'/f'{n:02d}'/'source_RGB_anchor_paths.png').convert('RGB'),'源正点、邻候选锚点和RGB路径',[f"原框路径{'可行' if not old['skip_reason'] else '不可行，保留失败'}",f"原图估计斜率{detail['estimated_slope']:.3f}",'锚点实例身份未人工确认'])
        for k,row in enumerate([a,b],1):
            prior=np.asarray(Image.open(row['prior']))>0;report.panel(card,k,overlay(photo,prior),'原完整源先验' if k==1 else '全原图范围的层间路径先验',[f'先验像素{int(prior.sum())}',f"新增{detail['prior_pixels_added_outside_original_source_mask']}、移除{detail['original_prior_pixels_removed']}源先验像素" if k==2 else '原源模型掩码未改','这些是输入假设，完整SAM输出另页验收'])
        report.add('anchors',f'anchor_source_{n:02d}',card);card=report.canvas(f'完整原生返回 {n}：同RGB、同点、同模型，只有输入先验不同')
        report.panel(card,0,photo,'实际未标记源输入',['邻候选仅作为层间假设','没有用参考坐标生成新点或先验','所有原结果保留'])
        for k,row in enumerate([a,b],1):
            x1,y1,x2,y2=row['box'];mask=np.load(Path(row['folder'])/'complete_full_mask.npy',allow_pickle=False)[y1:y2,x1:x2];audit=audits[(str(TWO),row['arm'])];candidate=next(r for r in audit['candidates'] if r['key']==row['key']);r3=next(r for c in audit['cases'] if c['case']==row['case'] for r in c['radii'] if r['radius']==3);own=next(r['ownership'] for r in r3['new_mask_ownership'] if r['key']==row['key']);notes=[f"原生面积{row['raw_pixels']} / 新增{candidate['extra_pixels']}",f"面积通过{candidate['area_fits']} / 点约束{row['native_prompt_guard']['valid']}"]+[f"obj{v['object_id']} {v['points']}点 / 纯度{v['purity']:.3f}" for v in own[:4]];report.panel(card,k,overlay(photo,mask),'原先验 → 完整原生返回' if k==1 else '层间先验 → 完整原生返回',notes)
        report.add('anchors',f'anchor_native_{n:02d}',card)
    card=report.canvas('原候选外框限制的先验对照：3组路径不可行，1组真实执行')
    for k,row in enumerate(g1['rows']):
        photo=Image.open(row['input']).convert('RGB');x1,y1,x2,y2=row['box'];m=np.load(Path(row['folder'])/'complete_full_mask.npy',allow_pickle=False)[y1:y2,x1:x2];report.panel(card,k,overlay(photo,m),row['arm'],[f"完整原生面积{row['raw_pixels']}",'仅2次新SAM；其余3组未编造返回'])
    report.panel(card,2,Image.open(ONE/'parents/01/source_RGB_anchor_paths.png').convert('RGB'),'失败原因：先验搜索域与倾斜锚点冲突',['原框下界在58列无可行路径','后续改为搜索完整原图，未放宽验收','所有失败计划与图片保留']);report.add('anchors','anchor_v1_failure',card)
    gp=read(gap.OUT/'frozen_plan.json');gr=read(gap.OUT/'review_completed.json')
    for proposal in gp['rows']:
        n=proposal['index'];photo=Image.open(proposal['input']).convert('RGB');v=gr['rows'][n-1];card=report.canvas(f'源掩码层次间隙 {n}：新正点未取自任何参考坐标');report.panel(card,0,Image.open(gap.OUT/'proposals'/f'{n:02d}'/'source_gap_anchors.png').convert('RGB'),'父候选覆盖、两端子候选未覆盖的源间隙',[f"新源点{proposal['source_point']}",'正点7×7内两子候选均没有前景','负点仍是源实例假设，不是人工真值'])
        for k,b in enumerate(proposal['new_point_views'][0]['bindings'],1):report.panel(card,k,Image.open(b['image']).convert('RGB'),'原中性近景' if k==1 else '原中性上下文',[v['decision']['object'],'同一原模型、提示、双图与实际256+256预算'])
        report.add('gaps',f'gap_source_{n:02d}',card);card=report.canvas(f'源间隙完整SAM对照 {n}：不得以稀疏命中代替实例验收');report.panel(card,0,photo,'实际未标记原图',['同一个新正点及两个源端点负点','两臂仅改变输入mask先验','原三轮及32点上限未增加'])
        for k,row in enumerate([r for r in gg['rows'] if r['parent']==n],1):
            x1,y1,x2,y2=row['box'];mask=np.load(Path(row['folder'])/'complete_full_mask.npy',allow_pickle=False)[y1:y2,x1:x2];report.panel(card,k,overlay(photo,mask),row['arm'],[f"完整面积{row['raw_pixels']} / 原生点约束{row['native_prompt_guard']['valid']}",'完整像素原样保存，未裁剪','负点中心空但7×7内有9前景，原守卫拒收' if not row['valid'] else '面积与提示通过仍不等于新增实例'])
        report.add('gaps',f'gap_native_{n:02d}',card)
    for n,candidate in enumerate(sem['candidates'],1):
        rows=[r for r in sem['rows'] if r['candidate']==candidate['key']];row=next(r for r in gg['rows'] if r['key']==candidate['key']);card=report.canvas(f'原完整语义复核 {n}：先冻结全部源有效候选，再运行原门槛')
        choices=[next(r for r in rows if r['stage']=='neutral' and r['kind']=='candidate' and r['name']==name) for name in ['near','context']]
        for name in ['pointer','corners']:choices.append(next((r for r in rows if r['stage']=='structure' and r['name']==name),None))
        choices += [next(r for r in rows if r['stage']=='neutral' and r['kind']==kind) for kind in ['gray','white']]
        for k,r in enumerate(choices):
            if r is None:report.panel(card,k,Image.open(row['input']).convert('RGB'),'上游复核未过，没有请求结构阶段',['这里只展示原照片，没有模型答复'],True)
            else:report.panel(card,k,Image.open(r['images'][0]).convert('RGB'),r['stage']+' / '+r['kind']+' / '+r['name'],[str(r['decision'])[:160]],True)
        report.add('review',f'gap_semantic_{n:02d}',card)
    for n,b in enumerate(lp['bindings'],1):
        rows=[r for r in review['rows'] if r['candidate']==b['candidate'] and r['view']==b['view']];card=report.canvas('原单基座复核：只移除均匀留白，原256视觉预算、提示与否决规则不变')
        order=[('real','square'),('real','aspect'),('gray','square'),('gray','aspect'),('white','square'),('white','aspect')]
        for k,(kind,form) in enumerate(order):
            r=next(r for r in rows if r['kind']==kind and r['layout']==form);d=r['decision'];report.panel(card,k,Image.open(r['image']).convert('RGB'),kind+' / '+form+' / '+b['view'],[f"枚举{d['category']}/{d['instance_status']}",f"原单基座判定{r['single_base']}；重复证据{r['repeated_evidence']}",d['evidence'][:80]],True)
        report.add('review',f'structure_layout_{n:02d}',card)
    # Reference projection is used ONLY in this after-generation diagnostic.
    sys.path[:0]=[cfg['source_snapshot'],str(Path(cfg['source_snapshot'])/'tools'),cfg['dependency_dir']]
    import evaluate_projected_support as ev
    from source.agent.edit_agent import _rasterize
    from projection_audit_core import frame_data,cameras
    assert sha(Path(cfg['source_snapshot'])/'tools/evaluate_projected_support.py')=='93c5152169d04d43b25dbf4a4139b12676daf78750a7f059837121656bb644c2';dataset=Path(cfg['dataset']);manifest=read(dataset/'import_manifest.json');data=frame_data(dataset,manifest,76,ev);cams=cameras(dataset,manifest,ev);ledger=[]
    for cam,ids in [(0,[42,48]),(1,[47,49])]:
        case=f'camera{cam}_frame76';source_indices,pix=ev.project_visible(data['xyz'],cams[cam]);owners=data['owner'][source_indices];sourcefile=ROOT/'outputs/labelagent_context_comparison_20261105/inputs'/(case+'.png');photo=Image.open(sourcefile).convert('RGB');old=read(ROOT/'outputs/labelagent_context_comparison_20261105/C_hybrid'/case/'result.json')['shapes'];old_masks=[_rasterize(s,1920,1536) for s in old if s['label']=='pallet'];old_union=np.maximum.reduce(old_masks)
        for oid in ids:
            idx=next(i for i,b in enumerate(data['boxes']) if b['obj_id']==oid);points=pix[owners==idx+1];assert len(points)>=5;x1,y1=np.maximum(0,points.min(0)-48);x2,y2=np.minimum([1920,1536],points.max(0)+49);box=[int(x1),int(y1),int(x2),int(y2)];crop=photo.crop(tuple(box));marked=crop.copy();draw=ImageDraw.Draw(marked)
            for x,y in points:draw.ellipse((int(x-x1)-2,int(y-y1)-2,int(x-x1)+2,int(y-y1)+2),fill='yellow')
            diagnostics=[];seen=set()
            for trial,row in all_native:
                if row['case']!=case or not row['new_SAM_request'] or row['key'] in seen:continue
                seen.add(row['key']);audit=audits[(str(trial),row['arm'])];r3=next(r for c in audit['cases'] if c['case']==case for r in c['radii'] if r['radius']==3);ownership=next(r['ownership'] for r in r3['new_mask_ownership'] if r['key']==row['key']);v=next((v for v in ownership if v['object_id']==oid),None)
                if v:diagnostics.append({'trial':trial.name,'key':row['key'],'mask':str(Path(row['folder'])/'complete_full_mask.npy'),'points':v['points'],'coverage':v['coverage'],'purity':v['purity'],'source_valid':row['valid'],'qualified_support':v['qualified_support']})
            best=max(diagnostics,key=lambda d:(d['points'],d['purity'])) if diagnostics else None;ledger.append({'case':case,'object_id':oid,'original_visible_reference_points':len(points),'diagnostic_crop_not_used_for_generation':box,'reference_is_not_dense_human_2D_truth':True,'new_generated_masks_with_any_r3_target_hits':len(diagnostics),'diagnostic_best_raw_hits_only_not_selection_or_adoption':best,'new_qualified_instance':False});card=report.canvas(f'四个剩余实例逐项诊断：{case} / 原参考obj{oid}，投影局部未用于生成')
            report.panel(card,0,crop,'原照片参考投影局部（诊断用）',['按原参考范围展示，非模型输入','不把此框当作人工二维真值','没有用参考点提出提示或裁分割输出']);report.panel(card,1,marked,'原参考的可见点',[f'原可见点{len(points)}个','原8像素/0.15米可见性及投影规则不变','黄色点只作事后诊断'])
            if best:
                mask=np.load(best['mask'],allow_pickle=False);report.panel(card,2,overlay(photo,mask).crop(tuple(box)),'本轮新返回的最大原始命中（仅诊断）',[f"命中{best['points']}/{len(points)} / 纯度{best['purity']:.3f}",f"源有效{best['source_valid']} / 支持{best['qualified_support']:.3f}",'不能以此诊断排名挑模型或绕过验收'])
            else:report.panel(card,2,overlay(photo,old_union).crop(tuple(box)),'本轮无对应新命中；显示原C结果',['源层次候选没有覆盖该遗漏','这是当前生成器的覆盖盲点','不声称这次已优化或恢复该实例'])
            report.add('ledger',f'case_camera{cam}_obj{oid}',card)
    report.text_card('ledger','summary','本轮结论：新种子与视图诊断均未恢复剩余实例',['11次新SAM、32次新VLM；没有新增训练或Mask2Former前向','跨裁剪锚点：原框3组路径无解，完整原图边界搜索可执行4组','新先验可增加原候选外的可见区域，但完整返回仍混入相邻层','源层次间隙提出2个新正点，原主体复核通过；完整实例复核仍拒收','一例接口成功但负点7×7有9前景，原生守卫正确拒绝，未裁输出','结构排版出现single枚举与stack证据冲突，原重复证据规则拦截','原13/17与历史6/6保留；历史Agent未重跑、原标注不变'])
    report.text_card('ledger','limits','接下来的重点：源候选缺失与可见单位身份，不再只抖动蓝色点',['新方法仍依赖已存在的PAL父/子候选，对另外3个遗漏没有新命中','需要为缺候选的实例建立独立源图证据，不能用参考坐标补种子','本轮先验输入可以增减像素，模型最终返回始终完整保留','灰白带定位标记的两种排版均拒判托盘，没有因此放宽真实候选判定','结构排版仅是离线诊断，不能认定原Agent已修复或上线','同批稀疏投影与图像语义仍有边界，不能伪装成独立密集真值'])
    videos=report.videos(['anchors','gaps','review','ledger']);summary={'status':'complete_rejected_source_anchors_gaps_and_matched_structure_layout','actual_new_SAM_requests':fresh,'actual_new_VLM_calls':vlm,'new_CPU_Mask2Former_forwards':0,'new_training':False,'logical_SAM_captures':len(all_native),'exact_SAM_control_reuses':g2['exact_previous_requests_reused'],'source_gap_points':len(gp['rows']),'source_gap_neutral_fresh_VLM':gr['actual_new_VLM_calls'],'full_semantic_fresh_VLM':sem['fresh_this_execution'],'structure_layout_fresh_VLM':review['actual_new_VLM_calls'],'structure_layout_original_real_replies_reused':review['exact_prior_requests_reused'],'structure_layout_matched_controls_all_reject_pallet':all(v['all_matched_controls_reject_pallet_identity'] for c in review['comparisons'] for v in c['layouts'].values()),'source_gap_native_point_guard_failures':sum(not r['native_prompt_guard']['valid'] for r in gg['rows']),'full_extent_review_requests':0,'source_gap_full_semantic_passes':0,'new_Agent_runs':0,'Agent_round_limit':3,'all_original_gates_unchanged':True,'no_output_mask_clipping':True,'retained_support':'13/17','retained_historical_Agent':'6/6','historical_Agent_not_rerun':True,'remaining_misses':4,'new_accepted_instances':0,'adopted':False,'pictures':len(report.cards),'videos':len(videos),'saved_stage_replay_not_screen_recording':True};save(OUT/'summary.json',summary);save(OUT/'four_remaining_case_ledger.json',{'cases':ledger,'diagnostic_after_generation_only':True,'not_an_oracle_selection_rule':True});save(OUT/'independent_readback.json',{'passed':True,'checks':checks,'layout_photo_and_locator_regions_retained_and_reconstructed':True,'original_six_and_references_exact':True,'counts':{'new_SAM':fresh,'new_VLM':vlm},'all_comparison_radii_counts_unchanged':True});text='本轮11新SAM、32新VLM，跨裁剪锚点、源层次间隙和等预算结构排版均未恢复新实例。新先验可补原候选外区域，但最终完整返回仍混邻层；一例接口成功却负点7×7残留9前景，原生守卫拒收。新正点主体复核通过，完整单基座复核仍失败；全范围阶段没有请求。结构排版中single枚举与stack证据矛盾被原规则拦截，带定位标记的灰白对照在两种排版均未误判托盘。全部原生像素保留、未裁输出、未采用，原13/17与历史6/6保留（Agent未重跑），剩4个。四例参考投影仅作事后诊断，未用于生成；本轮只对蓝色遗漏有新命中，另外三例仍是源候选覆盖盲点。视频为实际保存阶段回放。';(OUT/'本轮结论.txt').write_text(text,encoding='utf-8');report.index('跨裁剪锚点、源层次新实例与结构复核诊断',text,videos);print(summary,flush=True)
if __name__=='__main__':main()
