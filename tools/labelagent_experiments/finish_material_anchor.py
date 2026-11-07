"""Read back cue pixels, actual caption calls, exact cache replay and old raw mask."""
import hashlib,json,os,textwrap
from pathlib import Path
import av,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from run_material_anchor import ROOT,OUT,SHORT,old
from run_visual_budget import payload
from material_anchor_policy import propose
from semantic_subject_guard import compatible_pallet_subject

COUNTER=Path(os.environ.get('LABELAGENT_COUNTER_OUTPUT',str(ROOT/'outputs/labelagent_counter_material_20261108')))


def main():
    if (OUT/'summary.json').exists():raise FileExistsError('Preserve completed material report')
    p=old.read(OUT/'frozen_plan.json');s=old.read(OUT/'semantic_completed.json');cache=old.read(OUT/'caption_cross_study_overlap_audit.json');replay=old.read(OUT/'cache_first_replay.json');g=old.read(OUT/'generation_completed.json');audit=old.read(OUT/'support_area_audit.json');cp=old.read(COUNTER/'frozen_plan.json');cs=old.read(COUNTER/'semantic_completed.json');cg=old.read(COUNTER/'generation_completed.json');sourceplan=old.read(SHORT/'frozen_plan.json');assert all(old.digest(f)==h for f,h in p['protected'].items()) and all(old.digest(f)==h for f,h in cp['protected'].items());checks=[]
    for row,model in [(r,p['model_config_sha256']) for r in s['rows']]+[(r,cp['model_config_sha256']) for r in cs['rows']]:
        image=Path(row['semantic_image']).read_bytes();assert hashlib.sha256(image).hexdigest()==row['semantic_image_sha256'];body=payload(image,256);assert hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()==row['request_sha256'];reply=old.read(row['response']);assert old.digest(row['response'])==row['response_sha256'] and reply['local_vlm']['model_config_sha256']==model and reply['local_vlm']['actual_image_token_counts']==[256];assert json.loads(reply['choices'][0]['message']['content'])==row['decision'] and compatible_pallet_subject(row['decision'])==row['PAL_guard'];photo=Image.open(row['source_image']).convert('RGB');box=row['semantic_canvas_box'];clip=row['semantic_clip_box'];canvas=Image.new('RGB',(129,129),(127,127,127));canvas.paste(photo.crop(tuple(clip)),(clip[0]-box[0],clip[1]-box[1]));canvas=canvas.resize((512,512),Image.Resampling.LANCZOS);np.testing.assert_array_equal(np.asarray(canvas),np.asarray(Image.open(row['semantic_image']).convert('RGB')));roi=photo.crop(tuple(row['box']));np.testing.assert_array_equal(np.asarray(roi),np.asarray(Image.open(row['input']).convert('RGB')));checks.append({'key':row['key'],'request_model_reply_source_pixels_and_ROI_exact':True})
    for diag in p['diagnostics']:
        if 'seed' not in diag:continue
        seed=next(x for x in sourceplan['seeds'] if x['index']==diag['seed']);rows=[r for r in p['proposals'] if r['seed']==diag['seed']];image=Image.open(rows[0]['input']).convert('RGB');anchor=[seed['initial_points'][0][0]-seed['binding']['source_box'][0],seed['initial_points'][0][1]-seed['binding']['source_box'][1]];generated,_,_=propose(np.asarray(image),anchor);assert [r['points'] for r in generated]==[r['points'] for r in rows]
    assert replay['new_model_calls']==0 and replay['cache_verified_logical_views']==15 and replay['all_PAL_guard_decisions_preserved'];assert g['unique_real_SAM_calls']==0 and g['reused_exact_SAM_calls']==1 and cg['unique_real_SAM_calls']==0 and not cg['rows'];cached=g['rows'][0];a=np.load(cached['raw_array_file'],allow_pickle=False);meta=old.read(cached['metadata_file']);raw=a[f"{cached['prediction_call']}_masks"][cached['native_index']].squeeze();full=np.load(Path(cached['folder'])/'complete_full_mask.npy',allow_pickle=False);x1,y1,x2,y2=cached['box'];np.testing.assert_array_equal(full[y1:y2,x1:x2],raw);assert int(full.sum())==int((raw!=0).sum()) and hashlib.sha256(full.tobytes()).hexdigest()==cached['mask_sha256'];assert meta['selected_index']==cached['native_index'] and meta['image_sha256']==cached['input_sha256'] and meta['points']==cached['points'] and meta['prior_mask_sha256'] is None;np.testing.assert_array_equal(a[f"{cached['prediction_call']}_point_coords"],[[q['x'],q['y']] for q in cached['points']]);assert all(old.digest(f)==h for f,h in old.read(OUT/'generation_frozen_plan.json')['protected_cached_files'].items())
    binding=old.read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'];assert all(old.digest(ROOT/f)==h for f,h in binding.items());assets=OUT/'assets';assets.mkdir();font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',26);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20);cards=[]
    def canvas(title):
        c=Image.new('RGB',(1920,1080),'#eef3f7');ImageDraw.Draw(c).text((25,25),title,font=font,fill='#16364f');return c
    def panel(c,k,photo,title,notes):
        x=25+k*475;d=ImageDraw.Draw(c);d.text((x,100),title,font=small,fill='#16364f');photo=ImageOps.contain(photo,(445,500));c.paste(photo,(x,145));yy=690
        for note in notes:
            for line in textwrap.wrap(str(note),35 if str(note).isascii() else 21):d.text((x,yy),line,font=small,fill='#16364f');yy+=27
    def marked(photo,rows):
        im=photo.copy();d=ImageDraw.Draw(im)
        for r in rows:
            for q in r['points']:d.ellipse((q['x']-3,q['y']-3,q['x']+3,q['y']+3),fill='#ffcc00' if q['label'] else '#ff00cc')
        return im
    def store(c,name):
        f=assets/name;c.save(f,quality=94);cards.append(f)
    for seed in sorted({r['seed'] for r in s['rows']}):
        rows=[r for r in s['rows'] if r['seed']==seed];c=canvas(f'种子{seed:02d}：原图材质选点与实际中性回复');photo=Image.open(rows[0]['input']).convert('RGB');panel(c,0,marked(photo,rows),'源照片与提示点',['最多2组图像假设','原输入范围完整保留','颜色区域只用于选点','不代表物理实例分界']);cue=Image.open(OUT/f'seed{seed:02d}/RGB_cue_not_an_annotation.png');panel(c,1,cue,'输入颜色区域',['用于选择原图像素点','全部SAM输出单独保存','该图不参与面积验收','没有按209像素预算选点'])
        for k,r in enumerate(rows):panel(c,k+2,Image.open(r['semantic_image']),r['cue'].replace('_material_interior',''),[r['decision']['object'],f'词面托盘证据 {r["PAL_guard"]}','129原生像素放大到512','没有箭头或类别词写入图片'])
        store(c,f'seed{seed:02d}_material_cues_and_neutral_replies.jpg')
    original_card_count=len(cards)
    c=canvas('唯一通过的提示与历史完全一致：复用原SAM选中结果，未重新推理');photo=Image.open(cached['input']).convert('RGB');rgb=np.asarray(photo).copy();rgb[raw!=0]=(.5*rgb[raw!=0]+.5*np.array([0,215,165])).astype('uint8');detail=audit['candidates'][0];panel(c,0,marked(photo,[cached]),'原SAM输入',['原图、点、无先验路径一致','实际完整载荷散列已比较','新SAM调用0，精确复用1']);panel(c,1,Image.fromarray(rgb),'原服务选中完整mask',[f'完整面积 {cached["raw_pixels"]}',f'新增面积 {detail["extra_pixels"]}',f'面积通过 {detail["area_fits"]}',f'人员冲突 {detail["person_conflict"]}','全3份原生mask仍保存','历史捕获未单独保存低分辨率logits']);panel(c,2,Image.open(cached['semantic_image']),'实际中性输入',[cached['decision']['object'],'类别词出现不等于实例新增','此区域没有合格PAL参考点命中','不修改旧参考或原mask']);store(c,'cached_original_selected_mask_no_new_SAM.jpg')
    for seed in sorted({r['seed'] for r in cs['rows']}):
        rr=[r for r in cs['rows'] if r['seed']==seed];c=canvas(f'种子{seed:02d} 纹理点/背景负点假设：中性检查未过，未调用SAM');photo=Image.open(rr[0]['input']).convert('RGB');panel(c,0,photo,'原始源照片',['原标注与旧点保持','这组仅是新提示假设','背景负点尚未确认']);panel(c,1,marked(photo,rr),'冻结提示假设',['黄正点、紫负点','负点是未确认的背景假设','每组≤2点，最多2组','没有实际应用到SAM或标签'])
        for k,r in enumerate(rr):panel(c,k+2,Image.open(r['semantic_image']),r['cue'],[r['decision']['object'],f'词面托盘证据 {r["PAL_guard"]}',f'新模型调用 {r["new_model_call"]}','未过初筛，分割调用0'])
        store(c,f'seed{seed:02d}_textured_counter_hypotheses_rejected.jpg')
    c=canvas('本批未形成新实例：对应、材质颜色与完整基座须分别核验');d=ImageDraw.Draw(c);notes=['颜色提示15组：仅1组有词面托盘证据，且输入与旧SAM完全相同','实际13次VLM，其中4个载荷与历史重复；同批另复用2条，不隐藏开销','已补缓存优先路径：15视图精确重放0请求、原判定一致，5项检查通过','纹理纠偏4逻辑视图：2次新VLM、2次精确复用，均未获托盘证据','颜色可能连到地面或墙面；局部纹理也不能单独保证物理部位正确','新SAM0、旧SAM精确复用1，新独立mask0、新Agent0，未采用','仍13/17、旧6/6、剩4，原标注与0/3/6、面积、纯度等门槛不改','后续需要保持同一点的结构上下文，继续区分近邻托盘与实际目标部位']
    for i,n in enumerate(notes):d.text((40,150+i*105),n,font=font,fill='#16364f')
    store(c,'summary_material_counter_and_cache_limitations.jpg')
    videos=[]
    for filename,photos in [('01_八组材质提示与实际复核.mp4',cards[:original_card_count]),('02_旧原生掩码复用与纹理假设拒绝.mp4',cards[original_card_count:])]:
        with av.open(str(OUT/filename),'w') as container:
            stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p'
            for f in photos:
                arr=np.asarray(Image.open(f).convert('RGB'))
                for _ in range(40):
                    for packet in stream.encode(av.VideoFrame.from_ndarray(arr,format='rgb24')):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        cap=cv2.VideoCapture(str(OUT/filename));count=0
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
        cap.release();assert count==40*len(photos);videos.append({'path':filename,'frames':count,'sha256':old.digest(OUT/filename)})
    summary={'status':'complete_rejected_material_and_counter_hypotheses','material_logical_views':s['logical_views'],'material_actual_new_VLM_calls':s['unique_real_VLM_calls'],'material_accidental_historical_unique_repeats':cache['accidental_repeated_historical_unique_payloads'],'material_within_batch_reuses':s['same_batch_reuses'],'material_PAL_cues':s['eligible_for_SAM'],'counter_logical_views':cs['logical_views'],'counter_actual_new_VLM_calls':cs['unique_real_VLM_calls'],'counter_exact_reused_views':cs['exact_cache_reused_views'],'counter_PAL_cues':cs['eligible_for_SAM'],'actual_new_VLM_calls':s['unique_real_VLM_calls']+cs['unique_real_VLM_calls'],'new_SAM_calls':0,'reused_exact_source_selected_SAM_calls':1,'new_independent_masks':0,'cached_native_mask_area':cached['raw_pixels'],'cached_native_mask_extra':detail['extra_pixels'],'cached_capture_low_res_logits_not_available':True,'cache_first_exact_replay_views':15,'cache_first_replay_new_calls':0,'cache_first_original_guards_preserved':True,'cache_engineering_tests':5,'no_output_mask_clipping':True,'all_originals_and_old_six_exact':True,'retained_support':'13/17','retained_original_Agent':'6/6','remaining_misses':4,'Agent_round_limit':3,'new_Agent_runs':0,'adopted':False,'no_dense_human_truth_or_independent_generalization':True,'pictures':len(cards),'videos':len(videos)};old.save(OUT/'summary.json',summary);old.save(OUT/'independent_readback.json',{'passed':True,'checks':checks,'all_cue_points_reproduced_from_RGB_only':True,'source_selected_cached_mask_complete_and_exact':True,'original_raw_mask_count':int(a[f"{cached['prediction_call']}_masks"].shape[0]),'historical_low_res_logits_not_claimed':True,'cached_replay_did_not_generate_new_model_responses':True,'all_original_six_exact':True});old.save(OUT/'media_verification.json',{'passed':True,'pictures':len(cards),'videos':videos,'fully_decoded_frames':sum(v['frames'] for v in videos),'saved_stage_replay_not_screen_recording':True});text='颜色提示15组只有1组有词面托盘证据，且该输入与旧SAM请求完全一致，直接复用原服务选中的9362像素完整掩码（新增134、原面积及人员保护通过，但没有PAL参考实例命中）。没有新SAM或独立掩码。中性复核实际发送13个不同载荷，其中4个与历史重复，另有2条同批精确复用；这是缓存遗漏造成的重复调用，不包装成新输入或预先设计的重复实验。缓存优先入口已补齐并完成15视图零请求重放，模型/载荷/图像/256令牌均绑定，5项检查通过，旧实际13条回复仍保留。纹理点与未确认背景负点的4组假设另有2新VLM+2缓存，全部未获得托盘证据，未实际调用SAM或更改标签。颜色连通到地面/墙面、局部纹理不保证物理部位，相关失败全部保存。原13/17、旧6/6、原标注、三轮及全套验收不改，新Agent0，未采用。旧原生捕获没有单独保存低分辨率logits，不能声称已补齐该历史数据。视频为实测阶段回放。';(OUT/'本轮结论.txt').write_text(text,encoding='utf-8');(OUT/'index.html').write_text('<meta charset="utf-8"><title>材质点及纹理纠偏完整对照</title><h1>选点、物理证据与完整掩码检查</h1><p>'+text+'</p><a href="summary.json">实测结论</a> · <a href="independent_readback.json">独立回读</a> · <a href="caption_cross_study_overlap_audit.json">重复开销审计</a> · <a href="cache_first_replay.json">零请求缓存重放</a>'+''.join('<p><video controls width="1100" src="'+v['path']+'"></video></p>' for v in videos)+''.join('<p><img width="1100" src="'+f.relative_to(OUT).as_posix()+'"></p>' for f in cards),encoding='utf-8');print(summary,flush=True)


if __name__=='__main__':main()
